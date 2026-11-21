"""WGSL types (W3C WGSL, section 6) for the WGSL importer, and their CrossGL spellings.

:class:`TypeResolver` reads a type written as a :class:`~.WgslAst.Ident` (``vec3<f32>``, ``array<f32, 4>``,
``texture_storage_2d<rgba8unorm, write>``) into one of the frozen dataclasses below, following ``alias``
declarations and the predeclared aliases of section 6.2.10 (``vec3f``, ``mat2x2h``). :func:`crossgl_type` writes a
type the way CrossGL spells it, with the array extents a declarator takes separately (:func:`crossgl_declarator`).
A type CrossGL has no spelling for is refused with :class:`WgslImportError`.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from crosstl.translator.diagnostics import CODE_PREFIX, TranslationDiagnostic

from . import WgslAst as A

__all__ = [
    "ABSTRACT_FLOAT",
    "ABSTRACT_INT",
    "BOOL",
    "F16",
    "F32",
    "I32",
    "U32",
    "Array",
    "Atomic",
    "Matrix",
    "Pointer",
    "ResultStruct",
    "Sampler",
    "Scalar",
    "StructType",
    "Texture",
    "Vector",
    "WgslImportError",
    "concretize",
    "crossgl_declarator",
    "crossgl_type",
    "element_scalar",
    "TypeResolver",
    "array_suffix",
    "crossgl_format",
    "crossgl_type_expression",
    "is_abstract",
    "with_scalar",
]


class WgslImportError(TranslationDiagnostic):
    """A WGSL construct the importer cannot write as CrossGL."""

    def __init__(self, message, line=None):
        self.line = line
        where = f" (WGSL line {line})" if line else ""
        super().__init__(
            f"WGSL input: {message}{where}",
            target="cgl",
            code=f"{CODE_PREFIX}wgsl-import-unsupported",
        )


@dataclass(frozen=True)
class Scalar:
    name: str  # bool, i32, u32, f32, f16, abstract-int or abstract-float


BOOL, I32, U32, F32, F16 = (Scalar(n) for n in ("bool", "i32", "u32", "f32", "f16"))
ABSTRACT_INT, ABSTRACT_FLOAT = Scalar("abstract-int"), Scalar("abstract-float")


@dataclass(frozen=True)
class Vector:
    size: int
    elem: Scalar


@dataclass(frozen=True)
class Matrix:
    cols: int
    rows: int
    elem: Scalar

    @property
    def column(self) -> Vector:
        return Vector(self.rows, self.elem)


@dataclass(frozen=True)
class Array:
    """``count`` is the element count, ``None`` for a runtime-sized array; ``count_text`` the CrossGL spelling of
    an override-sized count."""

    elem: object
    count: int | None
    count_text: str | None = None


@dataclass(frozen=True)
class StructType:
    name: str


@dataclass(frozen=True)
class ResultStruct:
    """The built-in result structures of frexp, modf and atomicCompareExchangeWeak (sections 17.5, 17.8)."""

    kind: str
    fields: tuple

    def field(self, name):
        return dict(self.fields).get(name)


@dataclass(frozen=True)
class Atomic:
    elem: Scalar


@dataclass(frozen=True)
class Pointer:
    space: str
    elem: object
    access: str = "read_write"


@dataclass(frozen=True)
class Texture:
    """``kind`` is the WGSL type name; ``sampled`` the sampled type, ``texel_format`` and ``access`` those of a
    storage texture."""

    kind: str
    sampled: Scalar | None = None
    texel_format: str | None = None
    access: str | None = None

    @property
    def dimension(self) -> str:
        name = self.kind
        for prefix in (
            "texture_storage_",
            "texture_depth_multisampled_",
            "texture_multisampled_",
            "texture_depth_",
            "texture_",
        ):
            if name.startswith(prefix):
                return name[len(prefix) :]
        return name

    @property
    def is_storage(self) -> bool:
        return self.kind.startswith("texture_storage_")

    @property
    def is_depth(self) -> bool:
        return self.kind.startswith("texture_depth_")

    @property
    def is_multisampled(self) -> bool:
        return "multisampled" in self.kind


@dataclass(frozen=True)
class Sampler:
    comparison: bool = False


_SCALARS = {"bool": BOOL, "i32": I32, "u32": U32, "f32": F32, "f16": F16}
_SUFFIX_SCALARS = {"i": I32, "u": U32, "f": F32, "h": F16}
_VECTOR_ALIAS = re.compile(r"vec([234])([iufh])")
_MATRIX = re.compile(r"mat([234])x([234])")
_MATRIX_ALIAS = re.compile(r"mat([234])x([234])([fh])")
_SAMPLED_TEXTURES = frozenset(
    "texture_1d texture_2d texture_2d_array texture_3d texture_cube texture_cube_array texture_multisampled_2d".split()
)
_DEPTH_TEXTURES = frozenset(
    "texture_depth_2d texture_depth_2d_array texture_depth_cube texture_depth_cube_array"
    " texture_depth_multisampled_2d".split()
)
_STORAGE_TEXTURES = frozenset(
    "texture_storage_1d texture_storage_2d texture_storage_2d_array texture_storage_3d".split()
)


def is_abstract(type_) -> bool:
    scalar = element_scalar(type_)
    return scalar is not None and scalar.name.startswith("abstract")


def element_scalar(type_):
    """The scalar of a scalar, vector or matrix type, the element scalar of an array of them, else ``None``."""
    if isinstance(type_, Scalar):
        return type_
    if isinstance(type_, (Vector, Matrix, Atomic)):
        return type_.elem
    if isinstance(type_, Array):
        return element_scalar(type_.elem)
    return None


def with_scalar(type_, scalar):
    """``type_`` with ``scalar`` for its element scalar."""
    if isinstance(type_, Scalar):
        return scalar
    if isinstance(type_, Vector):
        return Vector(type_.size, scalar)
    if isinstance(type_, Matrix):
        return Matrix(type_.cols, type_.rows, scalar)
    if isinstance(type_, Array):
        return Array(with_scalar(type_.elem, scalar), type_.count, type_.count_text)
    return type_


def concretize(type_):
    """The concrete type an abstract type takes when nothing else decides it (section 6.1.5)."""
    scalar = element_scalar(type_)
    if scalar == ABSTRACT_INT:
        return with_scalar(type_, I32)
    if scalar == ABSTRACT_FLOAT:
        return with_scalar(type_, F32)
    return type_


def _scalar_argument(args, line, what):
    if not args or len(args) != 1:
        raise WgslImportError(f"{what} takes one type argument", line)
    return args[0]


class TypeResolver:
    """Resolves WGSL type expressions against a module's structures and aliases.

    ``eval_count`` folds an array element count to an integer, or returns ``None`` when the count is an override
    expression; ``count_text`` writes such a count as CrossGL.
    """

    def __init__(self, structs, aliases, eval_count, count_text):
        self.structs = structs
        self.aliases = aliases
        self.eval_count = eval_count
        self.count_text = count_text

    def resolve(self, node):
        if not isinstance(node, A.Ident):
            raise WgslImportError("expected a type", getattr(node, "line", None))
        name, args, line = node.name, node.template_args, node.line
        if name in _SCALARS and args is None:
            return _SCALARS[name]
        if name in self.aliases and args is None:
            return self.aliases[name]
        if name in self.structs and args is None:
            return StructType(name)
        alias = _VECTOR_ALIAS.fullmatch(name)
        if alias and args is None:
            return Vector(int(alias.group(1)), _SUFFIX_SCALARS[alias.group(2)])
        alias = _MATRIX_ALIAS.fullmatch(name)
        if alias and args is None:
            return Matrix(
                int(alias.group(1)),
                int(alias.group(2)),
                _SUFFIX_SCALARS[alias.group(3)],
            )
        if name in ("vec2", "vec3", "vec4"):
            return Vector(
                int(name[3]), self._scalar(_scalar_argument(args, line, name))
            )
        matrix = _MATRIX.fullmatch(name)
        if matrix:
            elem = self._scalar(_scalar_argument(args, line, name))
            return Matrix(int(matrix.group(1)), int(matrix.group(2)), elem)
        if name == "array":
            return self._array(args, line)
        if name == "atomic":
            return Atomic(self._scalar(_scalar_argument(args, line, name)))
        if name == "ptr":
            return self._pointer(args, line)
        if name in ("sampler", "sampler_comparison") and args is None:
            return Sampler(name == "sampler_comparison")
        if name in _SAMPLED_TEXTURES:
            return Texture(name, self._scalar(_scalar_argument(args, line, name)))
        if name in _DEPTH_TEXTURES and args is None:
            return Texture(name, F32)
        if name in _STORAGE_TEXTURES:
            if not args or len(args) != 2:
                raise WgslImportError(
                    f"{name} takes a texel format and an access mode", line
                )
            texel_format, access = (self._enumerant(arg, line) for arg in args)
            return Texture(
                name, _texel_scalar(texel_format, line), texel_format, access
            )
        raise WgslImportError(f"type '{name}' is not supported", line)

    def _scalar(self, node):
        type_ = self.resolve(node)
        if not isinstance(type_, Scalar):
            raise WgslImportError(
                "expected a scalar type argument", getattr(node, "line", None)
            )
        return type_

    @staticmethod
    def _enumerant(node, line):
        if not isinstance(node, A.Ident) or node.template_args is not None:
            raise WgslImportError("expected an enumerant", line)
        return node.name

    def _array(self, args, line):
        if not args or len(args) > 2:
            raise WgslImportError(
                "array takes an element type and an optional count", line
            )
        elem = self.resolve(args[0])
        if len(args) == 1:
            return Array(elem, None)
        count = self.eval_count(args[1])
        if count is None:
            return Array(elem, None, self.count_text(args[1]))
        if count <= 0:
            raise WgslImportError("an array count must be positive", line)
        return Array(elem, count)

    def _pointer(self, args, line):
        if not args or len(args) not in (2, 3):
            raise WgslImportError(
                "ptr takes an address space, a store type and an optional access mode",
                line,
            )
        space = self._enumerant(args[0], line)
        elem = self.resolve(args[1])
        access = (
            self._enumerant(args[2], line)
            if len(args) == 3
            else ("read" if space in ("uniform", "storage") else "read_write")
        )
        if space == "storage" and len(args) == 2:
            access = "read"
        return Pointer(space, elem, access)


# Texel formats (section 6.5.5) and the sampled type of their channels.
_TEXEL_FORMATS = {
    "rgba8unorm": F32,
    "rgba8snorm": F32,
    "rgba8uint": U32,
    "rgba8sint": I32,
    "rgba16uint": U32,
    "rgba16sint": I32,
    "rgba16float": F32,
    "r32uint": U32,
    "r32sint": I32,
    "r32float": F32,
    "rg32uint": U32,
    "rg32sint": I32,
    "rg32float": F32,
    "rgba32uint": U32,
    "rgba32sint": I32,
    "rgba32float": F32,
    "bgra8unorm": F32,
    "r8unorm": F32,
    "r8snorm": F32,
    "r8uint": U32,
    "r8sint": I32,
    "rg8unorm": F32,
    "rg8snorm": F32,
    "rg8uint": U32,
    "rg8sint": I32,
    "r16uint": U32,
    "r16sint": I32,
    "r16float": F32,
    "rg16uint": U32,
    "rg16sint": I32,
    "rg16float": F32,
    "rgb10a2uint": U32,
    "rgb10a2unorm": F32,
    "rg11b10ufloat": F32,
    "r16unorm": F32,
    "r16snorm": F32,
    "rg16unorm": F32,
    "rg16snorm": F32,
    "rgba16unorm": F32,
    "rgba16snorm": F32,
}
# CrossGL (GLSL) image format qualifiers of the WGSL texel formats.
_CROSSGL_FORMATS = {
    "rgba8unorm": "rgba8",
    "rgba8snorm": "rgba8_snorm",
    "rgba8uint": "rgba8ui",
    "rgba8sint": "rgba8i",
    "rgba16uint": "rgba16ui",
    "rgba16sint": "rgba16i",
    "rgba16float": "rgba16f",
    "r32uint": "r32ui",
    "r32sint": "r32i",
    "r32float": "r32f",
    "rg32uint": "rg32ui",
    "rg32sint": "rg32i",
    "rg32float": "rg32f",
    "rgba32uint": "rgba32ui",
    "rgba32sint": "rgba32i",
    "rgba32float": "rgba32f",
    "r8unorm": "r8",
    "r8snorm": "r8_snorm",
    "r8uint": "r8ui",
    "r8sint": "r8i",
    "rg8unorm": "rg8",
    "rg8snorm": "rg8_snorm",
    "rg8uint": "rg8ui",
    "rg8sint": "rg8i",
    "r16uint": "r16ui",
    "r16sint": "r16i",
    "r16float": "r16f",
    "rg16uint": "rg16ui",
    "rg16sint": "rg16i",
    "rg16float": "rg16f",
    "rgb10a2uint": "rgb10_a2ui",
    "rgb10a2unorm": "rgb10_a2",
    "rg11b10ufloat": "r11f_g11f_b10f",
    "r16unorm": "r16",
    "r16snorm": "r16_snorm",
    "rg16unorm": "rg16",
    "rg16snorm": "rg16_snorm",
    "rgba16unorm": "rgba16",
    "rgba16snorm": "rgba16_snorm",
}


def _texel_scalar(texel_format, line):
    if texel_format not in _TEXEL_FORMATS:
        raise WgslImportError(f"texel format '{texel_format}' is not supported", line)
    return _TEXEL_FORMATS[texel_format]


def crossgl_format(texel_format, line=None) -> str:
    if texel_format not in _CROSSGL_FORMATS:
        raise WgslImportError(
            f"texel format '{texel_format}' has no CrossGL image format", line
        )
    return _CROSSGL_FORMATS[texel_format]


_CROSSGL_SCALARS = {
    "bool": "bool",
    "i32": "int",
    "u32": "uint",
    "f32": "float",
    "f16": "half",
}
_VECTOR_PREFIX = {
    "bool": "bvec",
    "i32": "ivec",
    "u32": "uvec",
    "f32": "vec",
    "f16": "f16vec",
}
_TEXTURE_DIMENSIONS = {
    "1d": "1D",
    "2d": "2D",
    "2d_array": "2DArray",
    "3d": "3D",
    "cube": "Cube",
    "cube_array": "CubeArray",
}
_TEXTURE_PREFIX = {"f32": "", "i32": "i", "u32": "u"}


def crossgl_type(type_, line=None, struct_names=None) -> str:
    """The CrossGL spelling of ``type_``; an array is written as its element type (see :func:`crossgl_declarator`)."""
    names = struct_names or {}
    type_ = concretize(type_)
    if isinstance(type_, Scalar):
        return _CROSSGL_SCALARS[type_.name]
    if isinstance(type_, Vector):
        return f"{_VECTOR_PREFIX[type_.elem.name]}{type_.size}"
    if isinstance(type_, Matrix):
        if type_.elem == F32:
            return (
                f"mat{type_.cols}"
                if type_.cols == type_.rows
                else f"mat{type_.cols}x{type_.rows}"
            )
        if type_.elem == F16:
            return f"f16mat{type_.cols}x{type_.rows}"
    if isinstance(type_, Array):
        return crossgl_type(type_.elem, line, names)
    if isinstance(type_, StructType):
        return names.get(type_.name, type_.name)
    if isinstance(type_, Atomic):
        return _CROSSGL_SCALARS[type_.elem.name]
    if isinstance(type_, Sampler):
        return "comparison_sampler" if type_.comparison else "sampler"
    if isinstance(type_, Texture):
        return _crossgl_texture(type_, line)
    raise WgslImportError(f"type {type_} has no CrossGL spelling", line)


def _crossgl_texture(type_, line):
    dimension = type_.dimension
    if type_.is_storage:
        if dimension not in _TEXTURE_DIMENSIONS:
            raise WgslImportError(f"{type_.kind} is not supported", line)
        return f"{_TEXTURE_PREFIX[type_.sampled.name]}image{_TEXTURE_DIMENSIONS[dimension]}"
    if type_.is_multisampled:
        if dimension != "2d" or type_.is_depth:
            raise WgslImportError(f"{type_.kind} is not supported", line)
        return f"{_TEXTURE_PREFIX[type_.sampled.name]}texture2DMS"
    if dimension not in _TEXTURE_DIMENSIONS:
        raise WgslImportError(f"{type_.kind} is not supported", line)
    if type_.is_depth:
        return f"texture{_TEXTURE_DIMENSIONS[dimension]}Shadow"
    if type_.sampled.name not in _TEXTURE_PREFIX:
        raise WgslImportError(
            f"{type_.kind}<{type_.sampled.name}> is not supported", line
        )
    return (
        f"{_TEXTURE_PREFIX[type_.sampled.name]}texture{_TEXTURE_DIMENSIONS[dimension]}"
    )


def array_suffix(type_) -> str:
    """The ``[N][M]`` extents a declarator of ``type_`` takes: ``[]`` for a runtime-sized array."""
    suffix = ""
    while isinstance(type_, Array):
        if type_.count is not None:
            suffix += f"[{type_.count}]"
        elif type_.count_text is not None:
            suffix += f"[{type_.count_text}]"
        else:
            suffix += "[]"
        type_ = type_.elem
    return suffix


def crossgl_declarator(type_, name, line=None, struct_names=None) -> str:
    """``T name[N]``: a CrossGL declaration of ``name`` with ``type_``."""
    return f"{crossgl_type(type_, line, struct_names)} {name}{array_suffix(type_)}"


def crossgl_type_expression(type_, line=None, struct_names=None) -> str:
    """``T[N]`` for an array, as a constructor or return type spells it; ``T`` for any other type."""
    return f"{crossgl_type(type_, line, struct_names)}{array_suffix(type_)}"
