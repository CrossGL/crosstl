"""The GLSL language version a generated program needs, read from the program text.

The OpenGL emitter writes the source's ``#version`` line, or a default, before it knows which constructs the
program body will need. A layout qualifier, a built-in function or a shader stage that the declared version does
not have makes the artifact fail to compile, so once the program is rendered this module lists every requirement
it uses and the emitter raises the version line to the lowest one that satisfies them all. A requirement that an
extension the program enables already satisfies needs no raise. A requirement the profile cannot meet at any
version (a construct GLSL ES does not have, or any raise in a profile whose version is fixed, such as WebGL 2) is
returned as a refusal reason instead.

The minimums are those of the GLSL 4.60 and GLSL ES 3.20 specifications (the version that introduced each
built-in function in chapter 8 and each layout qualifier in section 4.4). Location qualifiers depend on the stage
and direction: vertex inputs and fragment outputs have them from GLSL 3.30 and GLSL ES 3.00, every other stage
interface from GLSL 4.10 and GLSL ES 3.10 (section 4.4.1 and 4.4.2).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

__all__ = [
    "GLSLRequirement",
    "VARIABLE_GATHER_OFFSET",
    "ES_OPAQUE_TYPES",
    "generated_program_requirements",
    "parse_version_line",
    "resolve_version_line",
    "stage_requirements",
    "with_es_default_precisions",
]


@dataclass(frozen=True)
class GLSLRequirement:
    """A construct and the language versions or extensions that provide it.

    ``desktop`` and ``es`` are the minimum ``#version`` numbers; ``None`` means the profile has no such construct
    at any version. ``desktop_extensions`` and ``es_extensions`` satisfy the requirement below that version.
    """

    feature: str
    desktop: int | None
    es: int | None
    desktop_extensions: tuple[str, ...] = ()
    es_extensions: tuple[str, ...] = ()

    def minimum(self, is_es):
        return self.es if is_es else self.desktop

    def satisfied_by(self, version, is_es, extensions):
        minimum = self.minimum(is_es)
        if minimum is not None and version >= minimum:
            return True
        allowed = self.es_extensions if is_es else self.desktop_extensions
        return any(extension in extensions for extension in allowed)

    def describe(self, is_es):
        minimum = self.minimum(is_es)
        profile = "GLSL ES" if is_es else "GLSL"
        extensions = self.es_extensions if is_es else self.desktop_extensions
        options = [f"{profile} {_dotted(minimum)}"] if minimum is not None else []
        options.extend(extensions)
        if not options:
            return f"{self.feature} does not exist in {profile}"
        return f"{self.feature} requires {' or '.join(options)}"


def _dotted(version):
    return f"{version // 100}.{version % 100:02d}"


# Built-in functions, keyed by name: the version of their first appearance in chapter 8.
_GPU_SHADER5 = ("GL_ARB_gpu_shader5",)
_ES_GPU_SHADER5 = ("GL_EXT_gpu_shader5", "GL_OES_gpu_shader5")
_PACKING = ("GL_ARB_shading_language_packing",)
_IMAGE_LOAD_STORE = ("GL_ARB_shader_image_load_store",)
_COMPUTE = ("GL_ARB_compute_shader",)
_DERIVATIVE_CONTROL = ("GL_ARB_derivative_control",)
_BUILTIN_FUNCTION_GROUPS = (
    (
        ("textureGather", "textureGatherOffset"),
        (400, 310, ("GL_ARB_texture_gather",) + _GPU_SHADER5, ()),
    ),
    (("textureGatherOffsets",), (400, 320, _GPU_SHADER5, _ES_GPU_SHADER5)),
    # Minimum-LOD clamped sampling exists only through GL_ARB_sparse_texture_clamp, which GLSL ES does not have.
    (
        (
            "textureClampARB",
            "textureOffsetClampARB",
            "textureGradClampARB",
            "textureGradOffsetClampARB",
        ),
        (None, None, ("GL_ARB_sparse_texture_clamp",), ()),
    ),
    (("textureQueryLod",), (400, None, ("GL_ARB_texture_query_lod",), ())),
    (("textureQueryLevels",), (430, None, ("GL_ARB_texture_query_levels",), ())),
    (
        ("textureSamples", "imageSamples"),
        (450, None, ("GL_ARB_shader_texture_image_samples",), ()),
    ),
    (
        (
            "bitfieldExtract",
            "bitfieldInsert",
            "bitfieldReverse",
            "bitCount",
            "findLSB",
            "findMSB",
            "uaddCarry",
            "usubBorrow",
            "umulExtended",
            "imulExtended",
            "frexp",
            "ldexp",
        ),
        (400, 310, _GPU_SHADER5, ()),
    ),
    (("fma",), (400, 320, _GPU_SHADER5, _ES_GPU_SHADER5)),
    (("packUnorm2x16", "unpackUnorm2x16"), (400, 300, _PACKING, ())),
    (
        ("packUnorm4x8", "packSnorm4x8", "unpackUnorm4x8", "unpackSnorm4x8"),
        (400, 310, _PACKING + _GPU_SHADER5, ()),
    ),
    (
        ("packSnorm2x16", "unpackSnorm2x16", "packHalf2x16", "unpackHalf2x16"),
        (420, 300, _PACKING, ()),
    ),
    (
        ("floatBitsToInt", "floatBitsToUint", "intBitsToFloat", "uintBitsToFloat"),
        (330, 300, ("GL_ARB_shader_bit_encoding",) + _GPU_SHADER5, ()),
    ),
    (
        ("interpolateAtCentroid", "interpolateAtSample", "interpolateAtOffset"),
        (400, 320, _GPU_SHADER5, ("GL_OES_shader_multisample_interpolation",)),
    ),
    (
        (
            "dFdxFine",
            "dFdyFine",
            "dFdxCoarse",
            "dFdyCoarse",
            "fwidthFine",
            "fwidthCoarse",
        ),
        (450, None, _DERIVATIVE_CONTROL, ()),
    ),
    (("memoryBarrier",), (420, 310, _IMAGE_LOAD_STORE, ())),
    (
        (
            "memoryBarrierShared",
            "memoryBarrierBuffer",
            "memoryBarrierImage",
            "groupMemoryBarrier",
        ),
        (430, 310, _COMPUTE, ()),
    ),
    (("imageLoad", "imageStore"), (420, 310, _IMAGE_LOAD_STORE, ())),
    (("imageSize",), (430, 310, ("GL_ARB_shader_image_size",), ())),
    (
        (
            "imageAtomicAdd",
            "imageAtomicMin",
            "imageAtomicMax",
            "imageAtomicAnd",
            "imageAtomicOr",
            "imageAtomicXor",
            "imageAtomicExchange",
            "imageAtomicCompSwap",
        ),
        (420, 320, _IMAGE_LOAD_STORE, ("GL_OES_shader_image_atomic",)),
    ),
    (
        ("atomicCounter", "atomicCounterIncrement", "atomicCounterDecrement"),
        (420, 310, ("GL_ARB_shader_atomic_counters",), ()),
    ),
    (
        (
            "atomicAdd",
            "atomicMin",
            "atomicMax",
            "atomicAnd",
            "atomicOr",
            "atomicXor",
            "atomicExchange",
            "atomicCompSwap",
        ),
        (430, 310, ("GL_ARB_shader_storage_buffer_object",) + _COMPUTE, ()),
    ),
)
# A non-constant ``textureGatherOffset`` offset (GLSL 4.60 and GLSL ES 3.20, section 8.9.4); the emitter records
# it while it renders, since the program text does not show whether an offset is a constant expression.
VARIABLE_GATHER_OFFSET = GLSLRequirement(
    "a non-constant textureGatherOffset offset", 400, 320, _GPU_SHADER5, _ES_GPU_SHADER5
)
BUILTIN_FUNCTION_REQUIREMENTS = {
    name: GLSLRequirement(f"built-in function '{name}'", *requirement)
    for names, requirement in _BUILTIN_FUNCTION_GROUPS
    for name in names
}
_BUILTIN_CALL = re.compile(
    r"\b("
    + "|".join(sorted(BUILTIN_FUNCTION_REQUIREMENTS, key=len, reverse=True))
    + r")\s*\("
)

# Opaque and 64-bit types whose first version is above the GLSL 3.30 and GLSL ES 3.00 baseline.
_FP64 = (400, None, ("GL_ARB_gpu_shader_fp64",), ())
_TYPE_GROUPS = (
    (
        ("double", "dvec2", "dvec3", "dvec4")
        + tuple(f"dmat{shape}" for shape in ("2", "3", "4"))
        + tuple(f"dmat{c}x{r}" for c in "234" for r in "234"),
        _FP64,
    ),
    (
        tuple(f"{prefix}samplerCubeArray" for prefix in ("", "i", "u"))
        + ("samplerCubeArrayShadow",),
        (
            400,
            320,
            ("GL_ARB_texture_cube_map_array",),
            ("GL_EXT_texture_cube_map_array", "GL_OES_texture_cube_map_array"),
        ),
    ),
    (
        tuple(f"{prefix}sampler2DMS" for prefix in ("", "i", "u")),
        (150, 310, ("GL_ARB_texture_multisample",), ()),
    ),
    (
        tuple(f"{prefix}sampler2DMSArray" for prefix in ("", "i", "u")),
        (
            150,
            320,
            ("GL_ARB_texture_multisample",),
            ("GL_OES_texture_storage_multisample_2d_array",),
        ),
    ),
    (
        tuple(f"{prefix}samplerBuffer" for prefix in ("", "i", "u")),
        (140, 320, ("GL_ARB_texture_buffer_object",), ("GL_EXT_texture_buffer",)),
    ),
    (
        tuple(
            f"{prefix}image{shape}"
            for prefix in ("", "i", "u")
            for shape in ("2D", "3D", "Cube", "2DArray")
        ),
        (420, 310, _IMAGE_LOAD_STORE, ()),
    ),
    (("atomic_uint",), (420, 310, ("GL_ARB_shader_atomic_counters",), ())),
)
TYPE_REQUIREMENTS = {
    name: GLSLRequirement(f"type '{name}'", *requirement)
    for names, requirement in _TYPE_GROUPS
    for name in names
}
_TYPE_USE = re.compile(
    r"\b(" + "|".join(sorted(TYPE_REQUIREMENTS, key=len, reverse=True)) + r")\b"
)

_EXPLICIT_LOCATION = (
    "GL_ARB_explicit_attrib_location",
    "GL_ARB_separate_shader_objects",
)
_SEPARATE_SHADER_OBJECTS = ("GL_ARB_separate_shader_objects",)
_STAGE_LABELS = {
    "vertex": "vertex",
    "fragment": "fragment",
    "geometry": "geometry",
    "tessellation_control": "tessellation control",
    "tessellation_evaluation": "tessellation evaluation",
}
_LAYOUT_REQUIREMENTS = {
    "uniform_location": GLSLRequirement(
        "layout(location) on a uniform", 430, 310, ("GL_ARB_explicit_uniform_location",)
    ),
    "index": GLSLRequirement(
        "layout(index) on a fragment output",
        330,
        None,
        _EXPLICIT_LOCATION + ("GL_ARB_blend_func_extended",),
        ("GL_EXT_blend_func_extended",),
    ),
    "component": GLSLRequirement(
        "layout(component)", 440, None, ("GL_ARB_enhanced_layouts",)
    ),
    "binding": GLSLRequirement(
        "layout(binding)", 420, 310, ("GL_ARB_shading_language_420pack",)
    ),
    "buffer": GLSLRequirement(
        "a shader storage block", 430, 310, ("GL_ARB_shader_storage_buffer_object",)
    ),
    "local_size": GLSLRequirement(
        "a compute workgroup size layout", 430, 310, _COMPUTE
    ),
    # Interface declarations and uniform blocks with a layout keep the GLSL 3.30 floor the emitter has always
    # used: its stage interfaces are written in the 3.30 style.
    "interface": GLSLRequirement(
        "an in/out stage interface declaration", 330, 300, _EXPLICIT_LOCATION
    ),
    "uniform_block_layout": GLSLRequirement(
        "a uniform block layout", 330, 300, ("GL_ARB_uniform_buffer_object",)
    ),
}
_STAGE_REQUIREMENTS = {
    "geometry": GLSLRequirement(
        "the geometry stage",
        150,
        320,
        (),
        ("GL_EXT_geometry_shader", "GL_OES_geometry_shader"),
    ),
    "tessellation_control": GLSLRequirement(
        "the tessellation control stage",
        400,
        320,
        ("GL_ARB_tessellation_shader",),
        ("GL_EXT_tessellation_shader", "GL_OES_tessellation_shader"),
    ),
    "tessellation_evaluation": GLSLRequirement(
        "the tessellation evaluation stage",
        400,
        320,
        ("GL_ARB_tessellation_shader",),
        ("GL_EXT_tessellation_shader", "GL_OES_tessellation_shader"),
    ),
    "compute": GLSLRequirement("the compute stage", 430, 310, _COMPUTE),
}

_UNKNOWN_STAGE = object()
_INTERFACE_QUALIFIERS = {
    "flat",
    "smooth",
    "noperspective",
    "centroid",
    "sample",
    "patch",
    "invariant",
    "precise",
    "highp",
    "mediump",
    "lowp",
    "readonly",
    "writeonly",
    "coherent",
    "volatile",
    "restrict",
    "shared",
}
_STORAGE_QUALIFIERS = ("in", "out", "uniform", "buffer")
_PLAIN_INTERFACE = re.compile(
    r"^(?:(?:" + "|".join(sorted(_INTERFACE_QUALIFIERS)) + r")\s+)*"
    r"(in|out)\s+[A-Za-z_]\w*\s+[A-Za-z_]\w*\s*(?:\[[^\]]*\]\s*)*;$"
)
_PLAIN_BUFFER_BLOCK = re.compile(
    r"^(?:(?:" + "|".join(sorted(_INTERFACE_QUALIFIERS)) + r")\s+)*buffer\s+\w+\s*\{?"
)
_COMMENTS = re.compile(r"//[^\n]*|/\*.*?\*/", re.S)
_EXTENSION_LINE = re.compile(
    r"^\s*#\s*extension\s+([A-Za-z_]\w*)\s*:\s*(?:enable|require|warn)\b", re.M
)


def parse_version_line(version_line):
    """Return ``(number, profile)`` for a ``#version`` line, or ``(None, None)`` when it has no number."""
    parts = str(version_line or "").strip().split()
    if len(parts) < 2 or parts[0] != "#version":
        return None, None
    try:
        number = int(parts[1])
    except ValueError:
        return None, None
    return number, (parts[2] if len(parts) > 2 else None)


def stage_io_location_requirement(stage, direction):
    """The ``layout(location)`` requirement of an ``in`` or ``out`` declaration of ``stage``."""
    if (stage, direction) in {("vertex", "in"), ("fragment", "out")}:
        return GLSLRequirement(
            f"layout(location) on a {stage} {direction}put",
            330,
            300,
            _EXPLICIT_LOCATION,
        )
    label = _STAGE_LABELS.get(stage) if stage is not _UNKNOWN_STAGE else None
    feature = (
        f"layout(location) on a {label} {direction}put"
        if label
        else f"layout(location) on a stage {direction}put"
    )
    return GLSLRequirement(
        feature, 410, 310, _SEPARATE_SHADER_OBJECTS, ("GL_EXT_separate_shader_objects",)
    )


def stage_requirements(stage_names):
    """The requirements of the shader stages a program defines, in a stable order."""
    return [
        _STAGE_REQUIREMENTS[stage]
        for stage in sorted(
            stage for stage in stage_names if stage in _STAGE_REQUIREMENTS
        )
    ]


def _split_layout(stripped):
    """Split ``layout(...) rest`` into the qualifier ids and the declaration after the closing parenthesis."""
    depth = 0
    for index in range(len("layout"), len(stripped)):
        char = stripped[index]
        if char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
            if depth == 0:
                inner = stripped[len("layout(") : index]
                ids = {
                    part.partition("=")[0].strip()
                    for part in inner.split(",")
                    if part.strip()
                }
                return ids, stripped[index + 1 :].strip()
    return None, None


def _storage_qualifier(declaration):
    for token in declaration.replace(";", " ; ").replace("{", " { ").split():
        if token in _STORAGE_QUALIFIERS:
            return token
        if token not in _INTERFACE_QUALIFIERS:
            return None
    return None


def _layout_line_requirements(ids, declaration, stage):
    storage = _storage_qualifier(declaration)
    is_block = "{" in declaration or (
        storage in {"uniform", "buffer"} and not declaration.rstrip().endswith(";")
    )
    requirements = []
    if "location" in ids:
        if storage in {"in", "out"}:
            requirements.append(stage_io_location_requirement(stage, storage))
        elif storage == "uniform" and not is_block:
            requirements.append(_LAYOUT_REQUIREMENTS["uniform_location"])
    for qualifier in ("index", "component", "binding"):
        if qualifier in ids:
            requirements.append(_LAYOUT_REQUIREMENTS[qualifier])
    if storage == "buffer":
        requirements.append(_LAYOUT_REQUIREMENTS["buffer"])
    elif storage == "uniform" and is_block:
        requirements.append(_LAYOUT_REQUIREMENTS["uniform_block_layout"])
    elif storage in {"in", "out"} and declaration.rstrip(";").strip() not in {
        "in",
        "out",
    }:
        requirements.append(_LAYOUT_REQUIREMENTS["interface"])
    if any(qualifier.startswith("local_size_") for qualifier in ids):
        requirements.append(_LAYOUT_REQUIREMENTS["local_size"])
    return requirements


def _stage_scanner(guard_stages, default_stage):
    """Yield the stage of each line, following the emitter's ``#ifdef GL_<STAGE>_SHADER`` guards."""
    stack = []

    def stage_of(stripped):
        if stripped.startswith("#"):
            directive = stripped[1:].strip()
            keyword = directive.split(None, 1)[0] if directive else ""
            if keyword == "ifdef":
                macro = directive.split()[1] if len(directive.split()) > 1 else ""
                inherited = stack[-1] if stack else default_stage
                stack.append(guard_stages.get(macro, inherited))
            elif keyword in {"if", "ifndef"}:
                stack.append(stack[-1] if stack else default_stage)
            elif keyword in {"else", "elif"} and stack:
                stack[-1] = _UNKNOWN_STAGE
            elif keyword == "endif" and stack:
                stack.pop()
            return None
        return stack[-1] if stack else default_stage

    return stage_of


def generated_program_requirements(code, stage_names=(), guard_macros=None):
    """Every requirement a rendered GLSL program uses, in order of first appearance without duplicates.

    ``stage_names`` are the stages the program defines and ``guard_macros`` maps a stage name to the macro the
    emitter guards that stage's declarations with when one artifact carries several stages.
    """
    stage_names = [stage for stage in stage_names if stage is not None]
    default_stage = stage_names[0] if len(stage_names) == 1 else _UNKNOWN_STAGE
    guard_stages = {macro: stage for stage, macro in (guard_macros or {}).items()}
    stage_of = _stage_scanner(guard_stages, default_stage)

    found = {}
    body = _COMMENTS.sub("", str(code or ""))
    for line in body.splitlines():
        stripped = line.strip()
        stage = stage_of(stripped)
        if stage is None:
            continue
        if stripped.startswith("layout"):
            ids, declaration = _split_layout(stripped)
            if ids is not None:
                for requirement in _layout_line_requirements(ids, declaration, stage):
                    found.setdefault(requirement, None)
        elif _PLAIN_INTERFACE.match(stripped):
            found.setdefault(_LAYOUT_REQUIREMENTS["interface"], None)
        elif _PLAIN_BUFFER_BLOCK.match(stripped):
            found.setdefault(_LAYOUT_REQUIREMENTS["buffer"], None)
        for match in _BUILTIN_CALL.finditer(stripped):
            found.setdefault(BUILTIN_FUNCTION_REQUIREMENTS[match.group(1)], None)
        for match in _TYPE_USE.finditer(stripped):
            found.setdefault(TYPE_REQUIREMENTS[match.group(1)], None)
    for requirement in stage_requirements(stage_names):
        found.setdefault(requirement, None)
    return list(found)


def program_extensions(code):
    """The extensions a rendered program enables with ``enable``, ``require`` or ``warn``."""
    return set(_EXTENSION_LINE.findall(str(code or "")))


def resolve_version_line(version_line, requirements, extensions, can_raise=True):
    """Return ``(version_line, refusals)``: the lowest version line meeting ``requirements``.

    The profile is kept: GLSL ES rises within GLSL ES and desktop GLSL within its profile (``core`` once the
    version has profiles). ``refusals`` lists the requirements no version of the profile meets, and every unmet
    one when ``can_raise`` is false.
    """
    number, profile = parse_version_line(version_line)
    if number is None:
        return version_line, []
    is_es = profile == "es"
    target = number
    refusals = []
    for requirement in requirements:
        if requirement.satisfied_by(number, is_es, extensions):
            continue
        minimum = requirement.minimum(is_es)
        if minimum is None or not can_raise:
            refusals.append(
                f"{requirement.describe(is_es)}; the target is {version_line}"
            )
            continue
        target = max(target, minimum)
    if target == number:
        return version_line, refusals
    if is_es:
        return f"#version {target} es", refusals
    if target < 150:
        return f"#version {target}", refusals
    return f"#version {target} {profile or 'core'}", refusals


# Opaque types with no default precision in GLSL ES (GLSL ES 3.20 section 4.7.4). ``sampler2D`` and
# ``samplerCube`` default to ``lowp``; they are listed too because a CrossGL texture read returns full-precision
# values, which ``lowp`` does not guarantee.
ES_OPAQUE_TYPES = tuple(
    f"{prefix}{kind}{shape}"
    for prefix in ("", "i", "u")
    for kind in ("sampler", "image")
    for shape in (
        "2D",
        "3D",
        "Cube",
        "2DArray",
        "2DMS",
        "2DMSArray",
        "CubeArray",
        "Buffer",
    )
) + (
    "sampler2DShadow",
    "samplerCubeShadow",
    "sampler2DArrayShadow",
    "samplerCubeArrayShadow",
    "atomic_uint",
)
_ES_OPAQUE_USE = re.compile(
    r"\b(" + "|".join(sorted(ES_OPAQUE_TYPES, key=len, reverse=True)) + r")\b"
)
_PRECISION_STATEMENT = re.compile(
    r"^\s*precision\s+(?:highp|mediump|lowp)\s+([A-Za-z_]\w*)\s*;", re.M
)


def with_es_default_precisions(code):
    """Declare ``highp`` defaults for every type a GLSL ES program uses without one.

    CrossGL ``float`` and ``int`` are 32-bit, and GLSL ES fragment shaders have no default ``float`` precision and
    a ``mediump`` default for ``int``; opaque types other than ``sampler2D`` and ``samplerCube`` have no default
    in any stage. ``highp`` meets the precision of every narrower qualifier, so a declared default only ever
    widens it. Existing ``precision`` statements are kept.
    """
    text = str(code or "")
    lines = text.splitlines()
    if not lines:
        return code
    declared = set(_PRECISION_STATEMENT.findall(text))
    body = _COMMENTS.sub("", text)
    used = {match.group(1) for match in _ES_OPAQUE_USE.finditer(body)}
    missing = [kind for kind in ("float", "int") if kind not in declared]
    missing.extend(
        kind for kind in ES_OPAQUE_TYPES if kind in used and kind not in declared
    )
    if not missing:
        return code
    insert_at = next(
        (
            index + 1
            for index, line in enumerate(lines)
            if line.strip().startswith("#version")
        ),
        0,
    )
    while insert_at < len(lines) and lines[insert_at].strip().startswith("#extension"):
        insert_at += 1
    lines[insert_at:insert_at] = [f"precision highp {kind};" for kind in missing]
    return "\n".join(lines) + ("\n" if text.endswith("\n") else "")
