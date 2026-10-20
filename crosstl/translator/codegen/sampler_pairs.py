"""OpenGL combined image samplers for textures that CrossGL samples through explicit samplers.

CrossGL keeps a texture and the sampler state it is filtered with apart, as HLSL (``Texture2D`` + ``SamplerState``),
MSL (``texture2d`` + ``sampler``) and Vulkan GLSL (``texture2D`` + ``sampler``) do: ``texture(t, s, P)``. OpenGL GLSL
has no separate sampler objects in the language; a ``sampler2D`` uniform names a texture unit, and the host binds one
texture and one sampler object to that unit (OpenGL 4.6 section 8.2, "Sampler Objects"). GLSL output therefore drops
the explicit sampler and samples the texture's uniform, which is faithful only while every texture is filtered by
one sampler state. A texture read through two samplers (``albedo.Sample(pointSampler, uv)`` and
``albedo.Sample(linearSampler, uv)``) collapsed into two identical samples.

:func:`lower_combined_sampler_pairs` gives each distinct (texture, sampler state) pair of such a texture its own
combined uniform, as SPIRV-Cross does for its GLSL output: the texture's uniform and binding keep the pair of a
combined texture's own sampler state, or else the pair its first non-comparison use names, and every other pair is a copy declared after all other declarations,
without explicit binding attributes, so the emitter assigns it a free unit and leaves every other unit in place. A comparison pair also takes the shadow sampler type, since a
depth comparison needs ``sampler2DShadow`` while a plain sample of the same texture needs ``sampler2D``. Separate
texture types (``texture2D``) become the combined type of their pair, because OpenGL GLSL has none.

The pairs reach a texture through helper functions as well: a texture parameter takes the pair its body (and its
callees) sample it with, and each call site passes the uniform of that pair. A parameter that the body samples
through more than one sampler state would need a specialized copy of the function per pair, and a texture sampled
through a sampler chosen at run time (``s[i]``) has no OpenGL equivalent; both are refused. Sources indexed through
``nonuniformEXT`` keep their separate Vulkan GLSL form, which the emitter preserves for descriptor indexing.

The returned records list every pair of each split texture, in the ``// crosstl-combined-samplers:`` artifact
header, so a host can bind the same texture with each sampler object to the units of those uniforms.
"""

from __future__ import annotations

import json
from copy import copy, deepcopy

from ..artifact_contracts import (
    COMBINED_SAMPLER_PREFIX,
    with_artifact_contract_header,
)
from ..ast import (
    ArrayAccessNode,
    ArrayType,
    ASTNode,
    FunctionCallNode,
    IdentifierNode,
    LiteralNode,
    NamedType,
)
from .image_access_contracts import (
    TEXTURE_COMPARE_INTRINSIC_NAMES,
    TEXTURE_GATHER_COMPARE_INTRINSIC_NAMES,
    TEXTURE_IMPLICIT_SAMPLER_INTRINSIC_NAMES,
    TEXTURE_QUERY_LOD_INTRINSIC_NAMES,
    TEXTURE_TEXEL_FETCH_INTRINSIC_NAMES,
)
from .lowering_refusals import refuse_unsupported

__all__ = [
    "canonicalize_separate_textures",
    "lower_combined_sampler_pairs",
    "with_combined_sampler_header",
]

_SEPARATE_TO_COMBINED = {
    "texture1D": "sampler1D",
    "texture1DArray": "sampler1DArray",
    "texture2D": "sampler2D",
    "texture2DArray": "sampler2DArray",
    "texture3D": "sampler3D",
    "textureCube": "samplerCube",
    "textureCubeArray": "samplerCubeArray",
}
_SEPARATE_TO_COMBINED.update(
    {
        f"{prefix}{separate}": f"{prefix}{combined}"
        for prefix in ("i", "u")
        for separate, combined in list(_SEPARATE_TO_COMBINED.items())
    }
)
_SHADOW_OF = {
    "sampler1D": "sampler1DShadow",
    "sampler1DArray": "sampler1DArrayShadow",
    "sampler2D": "sampler2DShadow",
    "sampler2DArray": "sampler2DArrayShadow",
    "samplerCube": "samplerCubeShadow",
    "samplerCubeArray": "samplerCubeArrayShadow",
}
_PLAIN_OF = {shadow: plain for plain, shadow in _SHADOW_OF.items()}
_FILTERABLE = (
    set(_SEPARATE_TO_COMBINED)
    | set(_SEPARATE_TO_COMBINED.values())
    | set(_PLAIN_OF)
    | {"sampler3D"}
)
_SAMPLER_TYPES = {
    "sampler",
    "comparison_sampler",
    "SamplerState",
    "SamplerComparisonState",
}
_COMPARE_OPS = (
    TEXTURE_COMPARE_INTRINSIC_NAMES
    | TEXTURE_GATHER_COMPARE_INTRINSIC_NAMES
    | {"textureGatherCompareOffsets"}
)
_FILTERED_OPS = TEXTURE_IMPLICIT_SAMPLER_INTRINSIC_NAMES | _COMPARE_OPS
_FETCH_OPS = TEXTURE_TEXEL_FETCH_INTRINSIC_NAMES
_BINDING_ATTRIBUTES = {"binding", "register", "texture", "sampler", "set", "space"}

# Sampler-state sources besides declared samplers: a combined texture's own state, and a texel fetch, which reads
# no sampler state but needs a non-shadow type.
_OWN = ("own",)
_FETCH = ("fetch",)
_DYNAMIC = ("dynamic",)
_PARALLEL = "elementwise"


def _type_of(node):
    return getattr(node, "var_type", None) or getattr(
        node, "param_type", getattr(node, "vtype", None)
    )


def _base_type_name(type_node):
    while isinstance(type_node, ArrayType):
        type_node = type_node.element_type
    if isinstance(type_node, NamedType):
        return type_node.name
    if type_node is None:
        return ""
    text = getattr(type_node, "name", None) or str(type_node)
    return str(text).split("[", 1)[0].split("<", 1)[0].strip()


def _with_base_type(type_node, base_name):
    if isinstance(type_node, ArrayType):
        updated = copy(type_node)
        updated.element_type = _with_base_type(type_node.element_type, base_name)
        return updated
    if isinstance(type_node, NamedType):
        return NamedType(base_name, list(type_node.generic_args))
    text = str(type_node)
    return base_name + text[len(text.split("[", 1)[0]) :]


def _set_type(node, type_node):
    for attribute in ("var_type", "param_type", "vtype"):
        if hasattr(node, attribute):
            setattr(node, attribute, type_node)


def _call_name(call):
    function = getattr(call, "function", None)
    if isinstance(function, IdentifierNode):
        return function.name
    return function if isinstance(function, str) else None


def _walk(root):
    """Every AST node under ``root`` in source order, each once."""
    seen = set()
    stack = [root]
    while stack:
        value = stack.pop()
        if isinstance(value, (list, tuple)):
            stack.extend(reversed(value))
            continue
        if isinstance(value, dict):
            stack.extend(reversed(list(value.values())))
            continue
        if not isinstance(value, ASTNode) or id(value) in seen:
            continue
        seen.add(id(value))
        yield value
        stack.extend(
            reversed([child for key, child in vars(value).items() if key != "parent"])
        )


def _uses_nonuniform(expression):
    return any(
        isinstance(node, FunctionCallNode) and _call_name(node) == "nonuniformEXT"
        for node in _walk(expression)
    )


def _signature(value):
    """A structural key of an expression, so two spellings of one index compare equal."""
    if isinstance(value, (list, tuple)):
        return tuple(_signature(item) for item in value)
    if not isinstance(value, ASTNode):
        return repr(value)
    return (type(value).__name__,) + tuple(
        (key, _signature(child))
        for key, child in sorted(vars(value).items())
        if key not in {"parent", "source_location", "annotations"}
    )


def _literal_index(expression):
    if isinstance(expression, LiteralNode) and isinstance(expression.value, int):
        return expression.value
    try:
        return int(str(getattr(expression, "value", None)))
    except (TypeError, ValueError):
        return None


class _Resource:
    """A texture or sampler declaration: a module or stage variable, an entry parameter, or a helper parameter."""

    def __init__(self, node, kind, container=None, function=None):
        self.node = node
        self.name = node.name
        self.kind = kind  # "texture" or "sampler"
        self.container = container  # list holding the declaration, for roots
        self.function = function  # helper function owning a parameter
        self.is_root = function is None
        self.base_type = _base_type_name(_type_of(node))
        self.keys = []  # distinct (sampler source, shadow) pairs, first use first
        self.sites = []  # (key, operand list, slot) of each read through this resource
        self.site_ids = set()
        self.preserved = False
        self.reaching = []  # sampler parameters: the sampler sources callers bind

    @property
    def is_shadow_type(self):
        return self.base_type in _PLAIN_OF

    def add_key(self, key):
        if key not in self.keys:
            self.keys.append(key)

    def add_site(self, key, operands, slot):
        identity = (key, id(operands), slot)
        if identity in self.site_ids:
            return False
        self.site_ids.add(identity)
        self.sites.append((key, operands, slot))
        self.add_key(key)
        return True


class _Lowering:
    def __init__(self, ast, target):
        self.ast = ast
        self.target = target
        self.functions = []
        self.function_ids = set()
        self.scopes = {}  # id(function) -> {name: _Resource}
        self.by_name = {}
        self.calls = []  # (caller, call, callee)
        self.records = []
        self.taken_names = set()

    def refuse(self, detail):
        refuse_unsupported(self.target, "explicit sampler", detail, None)

    # Collection -------------------------------------------------------------------------------------------------

    def classify(self, node):
        base = _base_type_name(_type_of(node))
        if base in _SAMPLER_TYPES:
            return "sampler"
        if base in _FILTERABLE:
            return "texture"
        return None

    def resources(self, nodes, container, function=None):
        scope = {}
        for node in nodes or []:
            kind = self.classify(node)
            if kind and getattr(node, "name", None):
                scope[node.name] = _Resource(node, kind, container, function)
        return scope

    def collect(self):
        module = self.resources(self.ast.global_variables, self.ast.global_variables)
        entries = []
        for stage in getattr(self.ast, "stages", {}).values():
            stage_scope = dict(module)
            stage_scope.update(
                self.resources(stage.local_variables, stage.local_variables)
            )
            if stage.entry_point is not None:
                entries.append((stage.entry_point, stage_scope))
            for function in stage.local_functions or []:
                self.add_function(function, stage_scope, entry=False)
        for function in self.ast.functions or []:
            self.add_function(function, module, entry=False)
        for function, scope in entries:
            self.add_function(function, scope, entry=True)
        for node in _walk(self.ast):
            name = getattr(node, "name", None)
            if isinstance(name, str):
                self.taken_names.add(name)

    def add_function(self, function, outer_scope, entry):
        if id(function) in self.function_ids:
            return
        self.function_ids.add(id(function))
        self.functions.append(function)
        scope = dict(outer_scope)
        parameters = function.parameters
        if entry:
            # Entry parameters are bound by the host like module resources.
            scope.update(self.resources(parameters, parameters))
        else:
            scope.update(self.resources(parameters, None, function))
        self.scopes[id(function)] = scope
        self.by_name.setdefault(function.name, []).append(function)

    def has_separate_resources(self):
        """Whether any sampler or separate texture is declared: otherwise every texture samples its own state."""
        return any(
            resource.kind == "sampler" or resource.base_type in _SEPARATE_TO_COMBINED
            for scope in self.scopes.values()
            for resource in scope.values()
        )

    def resolve(self, function, expression):
        """The resource an operand names, with a sampler-array element index, or ``None``."""
        index = None
        if isinstance(expression, ArrayAccessNode):
            index = expression.index_expr
            expression = expression.array_expr
        if not isinstance(expression, IdentifierNode):
            return None, None
        return self.scopes[id(function)].get(expression.name), index

    # Analysis ---------------------------------------------------------------------------------------------------

    def sampler_source(self, function, operand, texture_operand=None):
        resource, index = self.resolve(function, operand)
        if resource is None or resource.kind != "sampler":
            return None
        if _uses_nonuniform(operand):
            return "preserved"
        if index is None:
            return (resource,)
        if (
            isinstance(texture_operand, ArrayAccessNode)
            and isinstance(operand, ArrayAccessNode)
            and _signature(texture_operand.index_expr) == _signature(index)
        ):
            # ``texture(t[i], s[i], P)``: element i of the textures is filtered by element i of the samplers,
            # which a host expresses by binding sampler i to the unit of texture i.
            return (resource, _PARALLEL)
        literal = _literal_index(index)
        return _DYNAMIC if literal is None else (resource, literal)

    def scan(self):
        for function in self.functions:
            for call in _walk(function.body):
                if not isinstance(call, FunctionCallNode):
                    continue
                name = _call_name(call)
                args = call.args
                if name in _FILTERED_OPS or name in _FETCH_OPS:
                    self.scan_texture_operation(function, call, name, args)
                elif name in self.by_name:
                    for callee in self.by_name[name]:
                        if len(callee.parameters) == len(args):
                            self.calls.append((function, call, callee))

    def scan_texture_operation(self, function, call, name, args):
        if not args:
            return
        texture, _ = self.resolve(function, args[0])
        if texture is None or texture.kind != "texture":
            return
        if _uses_nonuniform(args[0]):
            texture.preserved = True
            return
        sampler = (
            self.sampler_source(function, args[1], args[0]) if len(args) > 1 else None
        )
        if sampler == "preserved":
            texture.preserved = True
            return
        if name in _FETCH_OPS:
            key = (_FETCH, False)
        elif sampler is None:
            # The texture's own sampler state: its declared type already says whether it compares.
            key = (_OWN, texture.is_shadow_type)
        elif name in TEXTURE_QUERY_LOD_INTRINSIC_NAMES:
            # A level-of-detail query reads the sampler's filters and LOD clamps and accepts either type.
            key = (sampler, None)
        else:
            key = (sampler, name in _COMPARE_OPS)
        texture.add_site(key, call.args, 0)

    def propagate(self):
        """Carry each helper parameter's pairs to the operands its callers pass, until nothing changes."""
        changed = True
        while changed:
            changed = False
            for caller, call, callee in self.calls:
                callee_scope = self.scopes[id(callee)]
                for index, parameter in enumerate(callee.parameters):
                    resource = callee_scope.get(parameter.name)
                    if resource is None or resource.kind != "texture":
                        continue
                    changed |= self.propagate_parameter(
                        caller, call, callee, index, resource
                    )

    def propagate_parameter(self, caller, call, callee, index, parameter):
        argument = call.args[index]
        texture, _ = self.resolve(caller, argument)
        if texture is None or texture.kind != "texture":
            return False
        changed = False
        if parameter.preserved and not texture.preserved:
            texture.preserved = True
            changed = True
        for key in parameter.keys:
            mapped = self.map_key(caller, call, callee, key)
            changed |= texture.add_site(mapped, call.args, index)
        return changed

    def bind_sampler_parameters(self):
        """Record, for each helper sampler parameter, the sampler sources its callers pass."""
        changed = True
        while changed:
            changed = False
            for caller, call, callee in self.calls:
                callee_scope = self.scopes[id(callee)]
                for index, parameter in enumerate(callee.parameters):
                    resource = callee_scope.get(parameter.name)
                    if resource is None or resource.kind != "sampler":
                        continue
                    source = self.sampler_source(caller, call.args[index])
                    if source in (None, "preserved"):
                        sources = [_DYNAMIC]
                    elif source[0] is not _DYNAMIC[0] and not source[0].is_root:
                        sources = list(source[0].reaching)
                    else:
                        sources = [source]
                    for item in sources:
                        if item not in resource.reaching:
                            resource.reaching.append(item)
                            changed = True

    def root_sampler_key(self, key):
        """A root texture's pair with a helper's sampler parameter, as the one sampler its callers bind."""
        source, shadow = key
        if source in (_OWN, _FETCH, _DYNAMIC) or source[0].is_root:
            return key
        reaching = source[0].reaching
        if not reaching:
            # No call binds the parameter: the helper never runs with this texture.
            return None
        if len(reaching) > 1 and _DYNAMIC not in reaching:
            return (("several", source[0]), shadow)
        if len(reaching) != 1 or reaching[0] == _DYNAMIC:
            return (_DYNAMIC, shadow)
        bound = reaching[0]
        if len(source) == 2:
            bound = (bound[0], source[1]) if len(bound) == 1 else _DYNAMIC
        return (bound, shadow)

    def map_key(self, caller, call, callee, key):
        source, shadow = key
        if len(source) == 2 and isinstance(source[0], _Resource):
            resource, literal = source
        elif len(source) == 1 and isinstance(source[0], _Resource):
            resource, literal = source[0], None
        else:
            return key
        if resource.is_root:
            return key
        if resource.function is not callee:
            return (_DYNAMIC, shadow)
        position = next(
            i for i, p in enumerate(callee.parameters) if p.name == resource.name
        )
        mapped = self.sampler_source(caller, call.args[position])
        if mapped in (None, "preserved"):
            return (_DYNAMIC, shadow)
        if literal is not None:
            if len(mapped) != 1:
                return (_DYNAMIC, shadow)
            mapped = (mapped[0], literal)
        return (mapped, shadow)

    # Lowering ---------------------------------------------------------------------------------------------------

    def resource_label(self, resource):
        if resource.is_root:
            return f"'{resource.name}'"
        return f"parameter '{resource.name}' of '{resource.function.name}'"

    def lower(self):
        textures = []
        seen = set()
        for function in self.functions:
            for resource in self.scopes[id(function)].values():
                if resource.kind == "texture" and id(resource) not in seen:
                    seen.add(id(resource))
                    textures.append(resource)
        for resource in textures:
            if not resource.preserved:
                self.lower_texture(resource)
        return self.ast

    def normalize_keys(self, resource):
        """Resolve sampler parameters of root textures and give LOD queries the pair of their sampler."""
        sites = [
            (self.root_sampler_key(key) if resource.is_root else key, operands, slot)
            for key, operands, slot in resource.sites
        ]
        sites = [site for site in sites if site[0] is not None]
        typed = [key for key, _operands, _slot in sites if key[1] is not None]

        def settle(key):
            if key[1] is not None:
                return key
            match = next((item for item in typed if item[0] == key[0]), None)
            return match or (key[0], resource.is_shadow_type)

        resource.sites = [
            (settle(key), operands, slot) for key, operands, slot in sites
        ]
        resource.keys = []
        for key, _operands, _slot in resource.sites:
            resource.add_key(key)

    def lower_texture(self, resource):
        self.normalize_keys(resource)
        filtered = [key for key in resource.keys if key[0] != _FETCH]
        several = [source[1] for source, _ in filtered if source[0] == "several"]
        if several:
            parameter = several[0]
            self.refuse(
                f"texture {self.resource_label(resource)} is sampled through sampler parameter "
                f"'{parameter.name}' of '{parameter.function.name}', which its callers bind to several "
                "samplers; OpenGL needs a specialized copy of the function per sampler"
            )
            return
        if any(source == _DYNAMIC for source, _shadow in filtered):
            self.refuse(
                f"texture {self.resource_label(resource)} is sampled through a sampler selected at run time; "
                "OpenGL binds one sampler state to each texture unit"
            )
            return
        variants = self.variant_keys(resource, filtered)
        if variants is None:
            return
        if resource.is_root:
            self.lower_root(resource, variants)
        else:
            self.retype(resource.node, resource, variants[0][1])

    def variant_keys(self, resource, filtered):
        """``[(key, shadow)]`` per uniform the texture needs; the first keeps the declaration."""
        needs_fetch = any(key[0] == _FETCH for key in resource.keys)
        plain = [key for key in filtered if not key[1]]
        shadow = [key for key in filtered if key[1]]
        if not resource.is_root:
            if len(filtered) > 1:
                self.refuse(
                    f"texture {self.resource_label(resource)} is sampled through {len(filtered)} sampler "
                    "states; OpenGL needs one combined sampler parameter per pair"
                )
                return None
            if filtered and filtered[0][1] and needs_fetch:
                self.refuse(
                    f"texture {self.resource_label(resource)} is both depth-compared and texel-fetched; "
                    "OpenGL needs a shadow and a non-shadow sampler parameter"
                )
                return None
            key = filtered[0] if filtered else (_FETCH, False)
            return [(key, key[1])]
        # The declaration's own sampler state stays on its uniform; explicit pairs follow in first-use order.
        ordered = sorted(plain + shadow, key=lambda key: key[0] != _OWN)
        if not ordered:
            return [((_FETCH, False), False)]
        if not plain and needs_fetch:
            ordered.insert(0, (_FETCH, False))
        return [(key, key[1]) for key in ordered]

    def combined_type(self, resource, shadow):
        base = _SEPARATE_TO_COMBINED.get(resource.base_type, resource.base_type)
        if shadow:
            if base.startswith(("i", "u")) or base == "sampler3D":
                self.refuse(
                    f"texture {self.resource_label(resource)} of type {resource.base_type} cannot take a depth "
                    "comparison in GLSL"
                )
                return None
            return _SHADOW_OF.get(base, base)
        return _PLAIN_OF.get(base, base)

    def retype(self, node, resource, shadow):
        if resource.base_type not in _SEPARATE_TO_COMBINED and not any(
            key[0] not in (_OWN, _FETCH) for key in resource.keys
        ):
            # Sampled only through its own state (or fetched): the declaration is the source's.
            return
        base = self.combined_type(resource, shadow)
        if base is not None and base != _base_type_name(_type_of(node)):
            _set_type(node, _with_base_type(_type_of(node), base))

    def lower_root(self, resource, variants):
        names = {variants[0][0]: resource.name}
        primary_key = variants[0][0]
        for key, shadow in variants[1:]:
            names[key] = self.variant_name(resource, key)
            clone = deepcopy(resource.node)
            clone.name = names[key]
            clone.attributes = [
                attribute
                for attribute in getattr(clone, "attributes", []) or []
                if str(getattr(attribute, "name", "")).lower()
                not in _BINDING_ATTRIBUTES
            ]
            self.retype(clone, resource, shadow)
            # After every declaration, so that the units the emitter assigns to the source's resources stay put.
            resource.container.append(clone)
        self.retype(resource.node, resource, variants[0][1])
        for key, operands, slot in resource.sites:
            self.rename_site(operands, slot, names.get(key, names[primary_key]))
        if len(variants) > 1:
            for key, shadow in variants:
                self.records.append(self.record(resource, key, names[key], shadow))

    def variant_name(self, resource, key):
        source, shadow = key
        if source in (_OWN, _FETCH):
            suffix = source[0]
        else:
            suffix = source[0].name
            if len(source) == 2 and source[1] != _PARALLEL:
                suffix = f"{suffix}_{source[1]}"
        base = f"{resource.name}_{suffix}"
        if shadow:
            base = f"{base}_shadow" if f"{base}" in self.taken_names else base
        name = base
        counter = 2
        while name in self.taken_names:
            name = f"{base}_{counter}"
            counter += 1
        self.taken_names.add(name)
        return name

    @staticmethod
    def rename_site(arguments, slot, name):
        operand = arguments[slot]
        if isinstance(operand, ArrayAccessNode):
            replacement = copy(operand)
            replacement.array_expr = replacement.array = IdentifierNode(name)
            arguments[slot] = replacement
        elif isinstance(operand, IdentifierNode) and operand.name != name:
            arguments[slot] = IdentifierNode(name)

    @staticmethod
    def record(resource, key, name, shadow):
        source = key[0]
        record = {"uniform": name, "texture": resource.name, "shadow": bool(shadow)}
        if source == _FETCH:
            record["sampler"] = None
        elif source == _OWN:
            record["sampler"] = resource.name
        else:
            record["sampler"] = source[0].name
            if len(source) == 2 and source[1] == _PARALLEL:
                record["elementwise"] = True
            elif len(source) == 2:
                record["samplerIndex"] = source[1]
        return record


def lower_combined_sampler_pairs(ast, target):
    """Return ``(ast, records)``: ``ast`` with one combined uniform per (texture, sampler state) pair.

    ``target`` is the label refusals are reported under (``"OpenGL"``, ``"WebGL"``). The caller's AST is not
    mutated. ``records`` describe the uniforms of every texture that was split, for
    :func:`with_combined_sampler_header`.
    """
    if not isinstance(ast, ASTNode) or not hasattr(ast, "global_variables"):
        return ast, []
    survey = _Lowering(ast, target)
    survey.collect()
    if not survey.has_separate_resources():
        return ast, []
    lowering = _Lowering(deepcopy(ast), target)
    lowering.collect()
    lowering.scan()
    lowering.propagate()
    lowering.bind_sampler_parameters()
    return lowering.lower(), lowering.records


def canonicalize_separate_textures(ast):
    """Return ``ast`` with separate texture types spelled as the combined CrossGL type of the same image.

    HLSL and MSL textures are separate objects whatever CrossGL spelling they come from: ``sampler2D`` with
    explicit samplers is how the HLSL and MSL importers write ``Texture2D`` and ``texture2d``. A ``texture2D``
    (Vulkan GLSL, SPIR-V) therefore emits as the same native texture, and as the depth texture (``sampler2DShadow``,
    MSL ``depth2d``) when every sample of it is a depth comparison, as the HLSL importer infers. The caller's AST is
    not mutated.
    """
    if not isinstance(ast, ASTNode) or not hasattr(ast, "global_variables"):
        return ast
    survey = _Lowering(ast, None)
    survey.collect()
    if not any(
        resource.base_type in _SEPARATE_TO_COMBINED
        for scope in survey.scopes.values()
        for resource in scope.values()
    ):
        return ast
    lowering = _Lowering(deepcopy(ast), None)
    lowering.collect()
    lowering.scan()
    lowering.propagate()
    seen = set()
    for scope in lowering.scopes.values():
        for resource in scope.values():
            if id(resource) in seen or resource.base_type not in _SEPARATE_TO_COMBINED:
                continue
            seen.add(id(resource))
            sampled = [key for key in resource.keys if key[0] != _FETCH]
            shadow = (
                bool(sampled)
                and all(key[1] for key in sampled)
                and len(sampled) == len(resource.keys)
                and resource.base_type[0] not in "iu"
                and resource.base_type != "texture3D"
            )
            base = _SEPARATE_TO_COMBINED[resource.base_type]
            base = _SHADOW_OF.get(base, base) if shadow else base
            _set_type(resource.node, _with_base_type(_type_of(resource.node), base))
    return lowering.ast


def with_combined_sampler_header(source, records):
    """Prefix the artifact with the combined-sampler pairs a host binds, when any texture was split.

    GLSL ES (WebGL 2) requires ``#version`` before any comment (GLSL ES 3.00 section 3.4), so an ES artifact
    carries no header; its pairs still follow the ``<texture>_<sampler>`` naming.
    """
    if not records:
        return source
    first_line = source.lstrip("\n").split("\n", 1)[0].split("//", 1)[0].split()
    if first_line[:1] == ["#version"] and first_line[-1:] == ["es"]:
        return source
    payload = {"schemaVersion": 1, "pairs": records}
    header = COMBINED_SAMPLER_PREFIX + json.dumps(
        payload, sort_keys=True, separators=(",", ":")
    )
    return with_artifact_contract_header(source, header)
