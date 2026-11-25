"""Textures, images and atomic counters for the GLSL compute shaders the harness runs on llvmpipe.

A GLSL shader declares an opaque uniform with ``layout(binding = n)`` (GLSL 4.60 specification, section 4.4.6): a
sampler takes texture unit ``n``, an image takes image unit ``n`` with the format its qualifier names, and an
``atomic_uint`` takes ``offset`` bytes into the atomic counter buffer bound at ``n``. An array of ``N`` samplers or
images takes the units ``n`` to ``n + N - 1``. CrossTL's GLSL writes the binding through its ``CROSSGL_BINDING(set,
vulkan, opengl)`` macro, whose OpenGL expansion is the third argument. :func:`declarations` reads every such
declaration into the record the job carries, and :func:`bind` creates and binds the resource of every record in the
current OpenGL context.

A texture's texels are a fixed function of the resource's ordinal among the shader's samplers or images (counting
array elements), its slice and its texel index, so a native probe and its translation read the same values when they
declare the same resources in the same order. The ordinal, not the unit, decides the texels: CrossTL renumbers units,
so the probe's ``binding = 8`` sampler is unit 0 in its translation. The shapes are those of
:mod:`.metal_textures`: one-dimensional and buffer textures are 64 texels wide, the others 4 texels on a side with 2
array layers, a cube has 6 faces per layer, and a multisampled texture has 4 samples, all zero. Every texture has one
level, nearest filtering and repeat addressing.

A texture or counter a shader writes is not read back: the probes that write one write a single texel or counter from
every invocation. Image atomics and counters return values that depend on the order the invocations ran in, so the
job of a shader that calls an atomic function is ``ordered``: the adapter starts llvmpipe without worker threads
(``LP_NUM_THREADS=0``), which runs the workgroups one after another in index order, in a worker process of its own.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import re
import struct

from .interface import InterfaceChanged

_WIDE = 64
_SIDE = 4
_LAYERS = 2
_SAMPLES = 4
_CUBE_FACES = 6

# GL enumerants (OpenGL 4.6 core profile, glcorearb.h).
GL_TEXTURE_1D = 0x0DE0
GL_TEXTURE_2D = 0x0DE1
GL_TEXTURE_3D = 0x806F
GL_TEXTURE_1D_ARRAY = 0x8C18
GL_TEXTURE_2D_ARRAY = 0x8C1A
GL_TEXTURE_CUBE_MAP = 0x8513
GL_TEXTURE_CUBE_MAP_POSITIVE_X = 0x8515
GL_TEXTURE_CUBE_MAP_ARRAY = 0x9009
GL_TEXTURE_BUFFER = 0x8C2A
GL_TEXTURE_2D_MULTISAMPLE = 0x9100
GL_TEXTURE_2D_MULTISAMPLE_ARRAY = 0x9102
GL_ATOMIC_COUNTER_BUFFER = 0x92C0
GL_TEXTURE0 = 0x84C0
GL_TEXTURE_MAG_FILTER = 0x2800
GL_TEXTURE_MIN_FILTER = 0x2801
GL_TEXTURE_WRAP_S = 0x2802
GL_TEXTURE_WRAP_T = 0x2803
GL_TEXTURE_WRAP_R = 0x8072
GL_TEXTURE_MAX_LEVEL = 0x813D
GL_NEAREST = 0x2600
GL_REPEAT = 0x2901
GL_RGBA = 0x1908
GL_RGBA_INTEGER = 0x8D99
GL_FLOAT = 0x1406
GL_INT = 0x1404
GL_UNSIGNED_INT = 0x1405
GL_STATIC_DRAW = 0x88E4
GL_READ_WRITE = 0x88BA
GL_NO_ERROR = 0

# Dimension: (texture target, storage extents, slices, layered when bound as an image). A cube's 6 faces are its
# slices, and a cube array's layer-faces.
_SHAPES = {
    "1D": (GL_TEXTURE_1D, (_WIDE,), 1, False),
    "1DArray": (GL_TEXTURE_1D_ARRAY, (_WIDE, _LAYERS), _LAYERS, True),
    "2D": (GL_TEXTURE_2D, (_SIDE, _SIDE), 1, False),
    "2DArray": (GL_TEXTURE_2D_ARRAY, (_SIDE, _SIDE, _LAYERS), _LAYERS, True),
    "3D": (GL_TEXTURE_3D, (_SIDE, _SIDE, _SIDE), 1, True),
    "Cube": (GL_TEXTURE_CUBE_MAP, (_SIDE, _SIDE), _CUBE_FACES, True),
    "CubeArray": (
        GL_TEXTURE_CUBE_MAP_ARRAY,
        (_SIDE, _SIDE, _CUBE_FACES * _LAYERS),
        _CUBE_FACES * _LAYERS,
        True,
    ),
    "Buffer": (GL_TEXTURE_BUFFER, (_WIDE,), 1, False),
    "2DMS": (GL_TEXTURE_2D_MULTISAMPLE, (_SIDE, _SIDE), 1, False),
    "2DMSArray": (
        GL_TEXTURE_2D_MULTISAMPLE_ARRAY,
        (_SIDE, _SIDE, _LAYERS),
        _LAYERS,
        True,
    ),
}
# Image format qualifier: (internal format, components, kind). A sampler, which names no format, gets the
# four-component 32-bit format of its element type.
_FORMATS = {
    "rgba32f": (0x8814, 4, "float"),
    "rgba16f": (0x881A, 4, "float"),
    "rg32f": (0x8230, 2, "float"),
    "rg16f": (0x822F, 2, "float"),
    "r32f": (0x822E, 1, "float"),
    "r16f": (0x822D, 1, "float"),
    "rgba8": (0x8058, 4, "unorm"),
    "rgba16": (0x805B, 4, "unorm"),
    "rg8": (0x822B, 2, "unorm"),
    "r8": (0x8229, 1, "unorm"),
    "r16": (0x822A, 1, "unorm"),
    "rgba8_snorm": (0x8F97, 4, "snorm"),
    "rgba32i": (0x8D82, 4, "int"),
    "rgba16i": (0x8D88, 4, "int"),
    "rgba8i": (0x8D8E, 4, "int"),
    "rg32i": (0x823B, 2, "int"),
    "r32i": (0x8235, 1, "int"),
    "rgba32ui": (0x8D70, 4, "uint"),
    "rgba16ui": (0x8D76, 4, "uint"),
    "rgba8ui": (0x8D7C, 4, "uint"),
    "rg32ui": (0x823C, 2, "uint"),
    "r32ui": (0x8236, 1, "uint"),
}
_SAMPLED = {"float": "rgba32f", "int": "rgba32i", "uint": "rgba32ui"}
# The bytes of one component of a buffer texture, which GL reads as stored: struct code per kind.
_BUFFER_CODES = {
    ("float", 0x8814): "f",
    ("float", 0x8230): "f",
    ("float", 0x822E): "f",
    ("int", 0x8D82): "i",
    ("int", 0x823B): "i",
    ("int", 0x8235): "i",
    ("uint", 0x8D70): "I",
    ("uint", 0x823C): "I",
    ("uint", 0x8236): "I",
    ("unorm", 0x8058): "B",
}
_QUALIFIERS = (
    r"(?:(?:readonly|writeonly|coherent|volatile|restrict|lowp|mediump|highp)\s+)*"
)
_DECLARATION = re.compile(
    r"(?:\blayout\s*\(((?:[^()]|\([^()]*\))*)\)\s*)?"
    rf"({_QUALIFIERS})\buniform\s+({_QUALIFIERS})"
    r"(\w+)\s+(\w+)\s*(?:\[\s*(\d*)\s*\])?\s*;"
)
_OPAQUE = re.compile(
    r"^([iu]?)(sampler|image)(1D|2D|3D|Cube|2DRect|Buffer|2DMS)(Array)?(Shadow)?$"
)
_BINDING = re.compile(
    r"\bbinding\s*=\s*(\d+)|\bCROSSGL_BINDING\s*\(\s*\d+\s*,\s*\d+\s*,\s*(\d+)\s*\)"
)
_ATOMIC_CALL = re.compile(r"\b(?:atomic|imageAtomic)\w*\s*\(")


def declarations(text: str) -> list[dict]:
    """The records of the opaque uniforms of the comment-free GLSL ``text``, in declaration order.

    A sampler or image record has ``kind``, ``type``, ``dimension``, ``element`` (``float``, ``int`` or ``uint``),
    ``format``, ``binding``, ``count`` and ``ordinal``, the number of samplers or images (counting array elements)
    declared before it; an ``atomic_uint`` record has ``kind``, ``binding``, ``offset`` and ``ordinal`` among the
    counters. Raises :class:`InterfaceChanged` for one the harness cannot create: a shadow, rectangle or separate
    texture or sampler, an array without a size, a declaration without an explicit binding, and an image format not in
    ``_FORMATS``.
    """
    records, ordinals, offsets = [], {"sampler": 0, "image": 0, "counter": 0}, {}
    for found in _DECLARATION.finditer(text):
        layout, type_name, name, size = (
            found.group(1) or "",
            found.group(4),
            found.group(5),
            found.group(6),
        )
        opaque = _OPAQUE.match(type_name)
        if type_name != "atomic_uint" and opaque is None:
            if re.match(r"^[iu]?(?:sampler|texture|image|subpassInput)\w*$", type_name):
                raise InterfaceChanged(
                    f"The GLSL harness creates no {type_name} ({name})"
                )
            continue
        binding = _BINDING.search(layout)
        if binding is None:
            raise InterfaceChanged(f"The GLSL uniform {name} has no explicit binding")
        unit = int(binding.group(1) or binding.group(2))
        if size == "":
            raise InterfaceChanged(
                f"The GLSL uniform {name} is an array without a size"
            )
        count = int(size) if size else 1
        if type_name == "atomic_uint":
            offset = re.search(r"\boffset\s*=\s*(\d+)", layout)
            # Without an offset, a counter follows the last one declared at its binding (GLSL 4.60 section 4.4.6.2).
            at = int(offset.group(1)) if offset else offsets.get(unit, 0)
            offsets[unit] = at + 4 * count
            records.append(
                {
                    "kind": "counter",
                    "binding": unit,
                    "offset": at,
                    "count": count,
                    "ordinal": ordinals["counter"],
                }
            )
            ordinals["counter"] += count
            continue
        prefix, kind, dimension, array, shadow = opaque.groups()
        dimension += array or ""
        if shadow or dimension not in _SHAPES:
            raise InterfaceChanged(f"The GLSL harness creates no {type_name} ({name})")
        element = {"": "float", "i": "int", "u": "uint"}[prefix]
        if kind == "image":
            names = "|".join(sorted(_FORMATS, key=len, reverse=True))
            qualifier = re.search(rf"\b({names})\b", layout)
            if qualifier is None:
                raise InterfaceChanged(
                    f"The GLSL image {name} has no format the harness creates"
                )
            image_format = qualifier.group(1)
        else:
            image_format = _SAMPLED[element]
        records.append(
            {
                "kind": kind,
                "type": type_name,
                "dimension": dimension,
                "element": element,
                "format": image_format,
                "binding": unit,
                "count": count,
                "ordinal": ordinals[kind],
            }
        )
        ordinals[kind] += count
    return records


def resources(records: list[dict], text: str) -> dict:
    """The job entries for the opaque uniform ``records`` of the GLSL ``text``: ``resources`` when it declares any,
    and ``ordered`` when it calls an atomic function. Raises :class:`InterfaceChanged` when two records share a unit.
    """
    for kind in ("sampler", "image"):
        units = [
            record["binding"] + offset
            for record in records
            if record["kind"] == kind
            for offset in range(record["count"])
        ]
        if len(units) != len(set(units)):
            raise InterfaceChanged(f"Two GLSL {kind}s share a unit: {units}")
    counters = [
        (record["binding"], record["offset"] + 4 * offset)
        for record in records
        if record["kind"] == "counter"
        for offset in range(record["count"])
    ]
    if len(counters) != len(set(counters)):
        raise InterfaceChanged(f"Two GLSL atomic counters share an offset: {counters}")
    entries = {"resources": records} if records else {}
    if _ATOMIC_CALL.search(text):
        entries["ordered"] = True
    return entries


def texels(record: dict, ordinal: int) -> bytes:
    """The contents of the texture ``record`` creates for the resource of ``ordinal``: every slice in order, each
    texel four components, as ``glTexSubImage*`` reads them with the external type of the record's format.
    """
    slices = _SHAPES[record["dimension"]][2]
    count = 4 * _texel_count(record["dimension"])
    per_slice = count // slices
    kind = _FORMATS[record["format"]][2]
    code = {"int": "i", "uint": "I"}.get(kind, "f")
    values = [
        _value(kind, (texel * 7 + index * 13 + ordinal * 29) % 61, ordinal)
        for index in range(slices)
        for texel in range(per_slice)
    ]
    return struct.pack(f"<{count}{code}", *values)


def _texel_count(dimension: str) -> int:
    _, extents, _, _ = _SHAPES[dimension]
    count = 1
    for extent in extents:
        count *= extent
    return count * (_CUBE_FACES if dimension == "Cube" else 1)


def _value(kind: str, word: int, ordinal: int):
    if kind == "float":
        return word * 0.125 + 0.0625
    if kind == "unorm":
        return word / 64.0
    if kind == "snorm":
        return (word - 30) / 32.0
    return word + ordinal


def buffer_texels(record: dict, ordinal: int) -> bytes:
    """The bytes of the buffer object behind the buffer texture ``record`` for the resource of ``ordinal``: GL reads
    them as the internal format stores them, with only the format's components."""
    internal, components, kind = _FORMATS[record["format"]]
    code = _BUFFER_CODES.get((kind, internal))
    if code is None:
        raise RuntimeError(
            f"The GLSL harness creates no {record['format']} buffer texture"
        )
    values = []
    for texel in range(_WIDE * components):
        word = (texel * 7 + ordinal * 29) % 61
        values.append(word * 4 if code == "B" else _value(kind, word, ordinal))
    return struct.pack(f"<{len(values)}{code}", *values)


def counter_values(record: dict) -> list[int]:
    """The starting values of the counters ``record`` declares."""
    return [
        100 + 29 * (record["ordinal"] + offset) for offset in range(record["count"])
    ]


# The result and argument types of every GL entry point :func:`bind` calls.
_SIGNATURES = {
    "glGetError": (ctypes.c_uint,),
    "glGenTextures": (None, ctypes.c_int, ctypes.POINTER(ctypes.c_uint)),
    "glDeleteTextures": (None, ctypes.c_int, ctypes.POINTER(ctypes.c_uint)),
    "glGenBuffers": (None, ctypes.c_int, ctypes.POINTER(ctypes.c_uint)),
    "glDeleteBuffers": (None, ctypes.c_int, ctypes.POINTER(ctypes.c_uint)),
    "glBindBuffer": (None, ctypes.c_uint, ctypes.c_uint),
    "glBufferData": (
        None,
        ctypes.c_uint,
        ctypes.c_ssize_t,
        ctypes.c_char_p,
        ctypes.c_uint,
    ),
    "glBindBufferBase": (None, ctypes.c_uint, ctypes.c_uint, ctypes.c_uint),
    "glActiveTexture": (None, ctypes.c_uint),
    "glBindTexture": (None, ctypes.c_uint, ctypes.c_uint),
    "glTexParameteri": (None, ctypes.c_uint, ctypes.c_uint, ctypes.c_int),
    "glTexStorage1D": (
        None,
        ctypes.c_uint,
        ctypes.c_int,
        ctypes.c_uint,
        ctypes.c_int,
    ),
    "glTexStorage2D": (
        None,
        ctypes.c_uint,
        ctypes.c_int,
        ctypes.c_uint,
        ctypes.c_int,
        ctypes.c_int,
    ),
    "glTexStorage3D": (
        None,
        ctypes.c_uint,
        ctypes.c_int,
        ctypes.c_uint,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
    ),
    "glTexStorage2DMultisample": (
        None,
        ctypes.c_uint,
        ctypes.c_int,
        ctypes.c_uint,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_ubyte,
    ),
    "glTexStorage3DMultisample": (
        None,
        ctypes.c_uint,
        ctypes.c_int,
        ctypes.c_uint,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_ubyte,
    ),
    "glTexSubImage1D": (
        None,
        ctypes.c_uint,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_uint,
        ctypes.c_uint,
        ctypes.c_char_p,
    ),
    "glTexSubImage2D": (
        None,
        ctypes.c_uint,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_uint,
        ctypes.c_uint,
        ctypes.c_char_p,
    ),
    "glTexSubImage3D": (
        None,
        ctypes.c_uint,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_uint,
        ctypes.c_uint,
        ctypes.c_char_p,
    ),
    "glClearTexImage": (
        None,
        ctypes.c_uint,
        ctypes.c_int,
        ctypes.c_uint,
        ctypes.c_uint,
        ctypes.c_void_p,
    ),
    "glTexBuffer": (None, ctypes.c_uint, ctypes.c_uint, ctypes.c_uint),
    "glBindImageTexture": (
        None,
        ctypes.c_uint,
        ctypes.c_uint,
        ctypes.c_int,
        ctypes.c_ubyte,
        ctypes.c_int,
        ctypes.c_uint,
        ctypes.c_uint,
    ),
}


class _GL:
    """The OpenGL entry points :func:`bind` calls, from ``eglGetProcAddress`` of the context moderngl made current."""

    def __init__(self):
        egl = ctypes.CDLL(ctypes.util.find_library("EGL") or "libEGL.so.1")
        egl.eglGetProcAddress.restype = ctypes.c_void_p
        egl.eglGetProcAddress.argtypes = [ctypes.c_char_p]
        for name, (result, *arguments) in _SIGNATURES.items():
            address = egl.eglGetProcAddress(name.encode())
            if not address:
                raise RuntimeError(f"The OpenGL driver has no {name}")
            setattr(self, name, ctypes.CFUNCTYPE(result, *arguments)(address))

    def check(self, what: str):
        error = self.glGetError()
        if error != GL_NO_ERROR:
            raise RuntimeError(f"OpenGL error 0x{error:04X} while {what}")


def _texture(gl: _GL, record: dict, ordinal: int, made: dict) -> int:
    target, extents, _, _ = _SHAPES[record["dimension"]]
    width, height, depth = (*extents, 1, 1)[:3]
    internal = _FORMATS[record["format"]][0]
    kind = _FORMATS[record["format"]][2]
    external = GL_RGBA_INTEGER if kind in ("int", "uint") else GL_RGBA
    component = {"int": GL_INT, "uint": GL_UNSIGNED_INT}.get(kind, GL_FLOAT)
    name = ctypes.c_uint()
    gl.glGenTextures(1, ctypes.byref(name))
    made["textures"].append(name.value)
    gl.glBindTexture(target, name.value)
    dimension = record["dimension"]
    if dimension == "Buffer":
        data = buffer_texels(record, ordinal)
        backing = ctypes.c_uint()
        gl.glGenBuffers(1, ctypes.byref(backing))
        made["buffers"].append(backing.value)
        gl.glBindBuffer(GL_TEXTURE_BUFFER, backing.value)
        gl.glBufferData(GL_TEXTURE_BUFFER, len(data), data, GL_STATIC_DRAW)
        gl.glTexBuffer(GL_TEXTURE_BUFFER, internal, backing.value)
        gl.check(f"creating a {record['type']}")
        return name.value
    if dimension in ("2DMS", "2DMSArray"):
        if dimension == "2DMS":
            gl.glTexStorage2DMultisample(target, _SAMPLES, internal, width, height, 1)
        else:
            gl.glTexStorage3DMultisample(
                target, _SAMPLES, internal, width, height, depth, 1
            )
        gl.glClearTexImage(name.value, 0, external, component, None)
        gl.check(f"creating a {record['type']}")
        return name.value
    for parameter, value in (
        (GL_TEXTURE_MIN_FILTER, GL_NEAREST),
        (GL_TEXTURE_MAG_FILTER, GL_NEAREST),
        (GL_TEXTURE_WRAP_S, GL_REPEAT),
        (GL_TEXTURE_WRAP_T, GL_REPEAT),
        (GL_TEXTURE_WRAP_R, GL_REPEAT),
        (GL_TEXTURE_MAX_LEVEL, 0),
    ):
        gl.glTexParameteri(target, parameter, value)
    data = texels(record, ordinal)
    if dimension == "1D":
        gl.glTexStorage1D(target, 1, internal, width)
        gl.glTexSubImage1D(target, 0, 0, width, external, component, data)
    elif dimension in ("1DArray", "2D"):
        gl.glTexStorage2D(target, 1, internal, width, height)
        gl.glTexSubImage2D(target, 0, 0, 0, width, height, external, component, data)
    elif dimension == "Cube":
        gl.glTexStorage2D(target, 1, internal, width, height)
        face = len(data) // _CUBE_FACES
        for index in range(_CUBE_FACES):
            gl.glTexSubImage2D(
                GL_TEXTURE_CUBE_MAP_POSITIVE_X + index,
                0,
                0,
                0,
                width,
                height,
                external,
                component,
                data[index * face : (index + 1) * face],
            )
    else:
        gl.glTexStorage3D(target, 1, internal, width, height, depth)
        gl.glTexSubImage3D(
            target, 0, 0, 0, 0, width, height, depth, external, component, data
        )
    gl.check(f"creating a {record['type']}")
    return name.value


def bind(job: dict) -> dict:
    """Create the resource of every record of ``job`` and bind it at its unit in the current OpenGL context.

    Returns what :func:`release` deletes once the dispatch has completed.
    """
    made = {"textures": [], "buffers": [], "gl": None}
    records = job.get("resources", ())
    if not records:
        return made
    gl = made["gl"] = _GL()
    counters, created = {}, []
    # Creating a texture binds it to the active texture unit, so every texture is created before any is bound.
    for record in records:
        if record["kind"] == "counter":
            words = counters.setdefault(record["binding"], {})
            for offset, value in enumerate(counter_values(record)):
                words[record["offset"] + 4 * offset] = value
            continue
        for element in range(record["count"]):
            name = _texture(gl, record, record["ordinal"] + element, made)
            created.append((record, record["binding"] + element, name))
    for record, unit, name in created:
        if record["kind"] == "sampler":
            gl.glActiveTexture(GL_TEXTURE0 + unit)
            gl.glBindTexture(_SHAPES[record["dimension"]][0], name)
        else:
            gl.glBindImageTexture(
                unit,
                name,
                0,
                1 if _SHAPES[record["dimension"]][3] else 0,
                0,
                GL_READ_WRITE,
                _FORMATS[record["format"]][0],
            )
        gl.check(f"binding a {record['type']} at unit {unit}")
    gl.glActiveTexture(GL_TEXTURE0)
    for binding, words in sorted(counters.items()):
        size = max(words) + 4
        data = bytearray(size)
        for offset, value in words.items():
            struct.pack_into("<I", data, offset, value)
        name = ctypes.c_uint()
        gl.glGenBuffers(1, ctypes.byref(name))
        made["buffers"].append(name.value)
        gl.glBindBuffer(GL_ATOMIC_COUNTER_BUFFER, name.value)
        gl.glBufferData(GL_ATOMIC_COUNTER_BUFFER, size, bytes(data), GL_STATIC_DRAW)
        gl.glBindBufferBase(GL_ATOMIC_COUNTER_BUFFER, binding, name.value)
        gl.check(f"binding an atomic counter buffer at {binding}")
    return made


def release(made: dict):
    """Delete the textures and buffers :func:`bind` created."""
    gl = made.get("gl")
    if gl is None:
        return
    for names, delete in (
        (made["textures"], gl.glDeleteTextures),
        (made["buffers"], gl.glDeleteBuffers),
    ):
        if names:
            array = (ctypes.c_uint * len(names))(*names)
            delete(len(names), array)
