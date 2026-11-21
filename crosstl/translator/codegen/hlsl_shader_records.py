"""Shader record buffers in HLSL output: ``[[vk::shader_record_ext]]`` for SPIR-V, local root constants for Direct3D 12.

A ray tracing stage reads the data of its shader binding table record through a CrossGL shader record block
(:mod:`..semantics.shader_records`), ``Record record @glsl_buffer_block(shaderRecordEXT, std430);``. HLSL declares it
once for each of DXC's targets, selected by the ``__spirv__`` macro DXC defines for SPIR-V::

    #ifdef __spirv__
    [[vk::shader_record_ext]] ConstantBuffer<Record> record;
    #else
    ConstantBuffer<Record> record : register(b0, space1);
    LocalRootSignature __crossgl_record_signature_record = { "RootConstants(num32BitConstants=4, b0, space=1)" };
    SubobjectToExportsAssociation __crossgl_record_exports_record = { "__crossgl_record_signature_record",
        "MissMain" };
    #endif

* For SPIR-V, DXC places a ``[[vk::shader_record_ext]]`` buffer in the ``ShaderRecordBufferKHR`` storage class
  (DXC SPIR-V reference, "Vulkan specific attributes"). It lays the buffer out with its relaxed std430 rule: a vector
  of 32-bit components is aligned to its component size unless it would then straddle a 16-byte boundary
  (:mod:`crosstl.backend.DirectX.shader_record_import` reads the same rule).
* Direct3D 12 passes a shader its record as local root arguments, which a local root signature maps to registers
  (DirectX Raytracing functional spec, "Local root signatures vs global root signatures"). DXIL has no attribute for
  this, so the program declares the local root signature as a DXIL subobject: root constants that cover the record's
  bytes at the buffer's register, associated with every entry point that reads the record. The runtime takes both
  from the library when the host adds it to a state object. Root constants are read as a constant buffer, so the
  record's members are at the offsets HLSL constant-buffer packing gives them. The arguments follow the 32-byte
  shader identifier in the record, as Vulkan's record data follows the shader group handle, so the host writes the
  same bytes to both.

The register space is one above every space the program's resources name, so the local root signature cannot overlap
the ranges of the program's global one; each record takes the next ``b`` register in it.

The one declaration is exact only where three layouts agree: the record's own (std430, std140 or scalar), DXC's
relaxed std430 and HLSL constant-buffer packing. :class:`HlslShaderRecordError` refuses a record on which they differ
(a ``float`` array, whose constant-buffer elements are 16 bytes apart, or a ``vec3`` after a ``float`` under std430),
a record with a matrix member, a record a stage without a shader binding table reads, an entry point that reads two
records, and a record larger than the 4064 bytes of local root arguments a record holds
(``D3D12_RAYTRACING_MAX_SHADER_RECORD_STRIDE`` minus ``D3D12_SHADER_IDENTIFIER_SIZE_IN_BYTES``).
"""

from __future__ import annotations

import re

from ..ast import IdentifierNode
from ..diagnostics import CODE_PREFIX, TranslationDiagnostic
from ..explicit_specializations import functions_by_called_name
from ..semantics.shader_records import (
    is_shader_record_buffer,
    shader_record_memory_layout,
)
from .glsl_block_layout import (
    block_alignment,
    block_array_stride,
    block_size,
    block_struct_placements,
    crossgl_struct_layout_type,
)
from .glsl_buffer_layout import glsl_buffer_block_node_type
from .hlsl_buffer_layout import (
    HlslLayoutError,
    align_to,
    cbuffer_array_stride,
    cbuffer_held_struct_placements,
    cbuffer_struct_placements,
)
from .stage_utils import collect_stage_entry_records

__all__ = [
    "HlslShaderRecordError",
    "HlslShaderRecords",
    "is_shader_record_buffer",
]

# Local root arguments a record can hold: D3D12_RAYTRACING_MAX_SHADER_RECORD_STRIDE (4096) less the
# D3D12_SHADER_IDENTIFIER_SIZE_IN_BYTES (32) identifier that starts it.
MAX_RECORD_BYTES = 4096 - 32
WORD_BYTES = 4
VECTOR_BOUNDARY = 16
STD430 = "std430"
_RAY_STAGES = frozenset(
    {
        "ray_generation",
        "ray_intersection",
        "ray_closest_hit",
        "ray_any_hit",
        "ray_miss",
        "ray_callable",
        "intersection",
        "closesthit",
        "anyhit",
        "miss",
        "callable",
    }
)


class HlslShaderRecordError(TranslationDiagnostic):
    """A shader record block HLSL cannot declare with the same bytes for Direct3D 12 and SPIR-V."""

    code = f"{CODE_PREFIX}directx-shader-record-unsupported"

    def __init__(self, name, reason, *, source_location=None):
        self.record = name
        self.source_location = source_location
        self.missing_capabilities = ("directx.shader-record-buffer",)
        super().__init__(
            f"shader record buffer '{name}' (GL_EXT_ray_tracing shaderRecordEXT) cannot be declared in HLSL: "
            f"{reason}",
            target="directx",
        )


# Layouts ----------------------------------------------------------------------------------------------------------


def _relaxed_placements(layout_type):
    """``(name, layout type, offset)`` of a struct's members under DXC's relaxed std430 rule.

    Only a vector of 32-bit or narrower components is relaxed: it is aligned to its component size and moved to the
    next 16-byte boundary when it would straddle one. Every other member keeps its std430 alignment.
    """
    placements = []
    cursor = 0
    for name, member_type in layout_type.members:
        size = _relaxed_size(member_type)
        if member_type.kind == "vector" and member_type.scalar_size <= WORD_BYTES:
            start = align_to(cursor, member_type.scalar_size)
            if start // VECTOR_BOUNDARY != (start + size - 1) // VECTOR_BOUNDARY:
                start = align_to(start, VECTOR_BOUNDARY)
        else:
            start = align_to(cursor, block_alignment(member_type, STD430))
        placements.append((name, member_type, start))
        cursor = start + size
    return placements


def _relaxed_size(layout_type):
    if layout_type.kind == "struct":
        end = max(
            (
                start + _relaxed_size(member)
                for _name, member, start in _relaxed_placements(layout_type)
            ),
            default=0,
        )
        return align_to(end, block_alignment(layout_type, STD430))
    if layout_type.kind == "array":
        return _relaxed_stride(layout_type) * layout_type.count
    return block_size(layout_type, STD430)


def _relaxed_stride(layout_type):
    return align_to(
        _relaxed_size(layout_type.element),
        block_alignment(layout_type.element, STD430),
    )


def _placements(layout_type, rules, root):
    """``(name, layout type, offset)`` of a struct's members under ``rules``."""
    if rules == "cbuffer":
        placements, _size = (
            cbuffer_struct_placements(layout_type)
            if root
            else cbuffer_held_struct_placements(layout_type)
        )
    elif rules == "relaxed":
        return _relaxed_placements(layout_type)
    else:
        placements, _size = block_struct_placements(layout_type, rules, root=root)
    return [
        (placement.name, placement.layout_type, placement.offset)
        for placement in placements
    ]


def _stride(layout_type, rules):
    if rules == "cbuffer":
        return cbuffer_array_stride(layout_type)
    if rules == "relaxed":
        return _relaxed_stride(layout_type)
    return block_array_stride(layout_type, rules)


def _leaves(layout_type, rules, base=0, path="", root=True):
    """Every member offset and array stride of a record under ``rules``, in member order."""
    if layout_type.kind == "array":
        return ((path, "stride", _stride(layout_type, rules)),) + _leaves(
            layout_type.element, rules, base, f"{path}[]", root=False
        )
    if layout_type.kind != "struct":
        return ()
    found = ()
    for name, member_type, offset in _placements(layout_type, rules, root):
        member_path = f"{path}.{name}" if path else name
        found += ((member_path, "offset", base + offset),)
        found += _leaves(member_type, rules, base + offset, member_path, root=False)
    return found


def _matrix_path(layout_type, path=""):
    if layout_type.kind == "matrix":
        return path or layout_type.struct_name
    if layout_type.kind == "array":
        return _matrix_path(layout_type.element, f"{path}[]")
    for name, member_type in (
        layout_type.members if layout_type.kind == "struct" else ()
    ):
        found = _matrix_path(member_type, f"{path}.{name}" if path else name)
        if found:
            return found
    return None


def _first_difference(declared, other):
    for expected, actual in zip(declared, other):
        if expected != actual:
            return expected, actual
    return None


# Program --------------------------------------------------------------------------------------------------------


class HlslShaderRecords:
    """The shader record blocks of one program and the declarations HLSL writes for them."""

    def __init__(self, codegen, ast, target_stage):
        self.codegen = codegen
        self.ast = ast
        self.target_stage = target_stage
        self._declarations = None

    @staticmethod
    def collect_struct_names(codegen, declarations):
        """Names of the structs record blocks hold, which the generator declares as ordinary structs."""
        names = set()
        for node in declarations:
            if is_shader_record_buffer(node):
                names.add(
                    str(codegen.resource_base_type(glsl_buffer_block_node_type(node)))
                )
        return names

    def declaration(self, node):
        """The HLSL declaration of the record block ``node``, with its local root signature."""
        if self._declarations is None:
            self._declarations = self._plan()
        return self._declarations.get(id(node), "")

    # Planning ---------------------------------------------------------------------------------------------------

    def _records(self):
        found = []
        seen = set()
        declarations = list(getattr(self.ast, "global_variables", None) or [])
        for _stage_type, stage in getattr(self.ast, "stages", {}).items():
            declarations += list(getattr(stage, "local_variables", None) or [])
        for node in declarations:
            if is_shader_record_buffer(node) and id(node) not in seen:
                seen.add(id(node))
                found.append(node)
        return found

    def _space(self):
        spaces = [0]
        declarations = list(getattr(self.ast, "global_variables", None) or [])
        declarations += list(getattr(self.ast, "cbuffers", None) or [])
        for _stage_type, stage in getattr(self.ast, "stages", {}).items():
            declarations += list(getattr(stage, "local_variables", None) or [])
            declarations += list(getattr(stage, "local_cbuffers", None) or [])
        for node in declarations:
            # The generator spells an explicit space as its register text, ``space3``.
            match = re.fullmatch(
                r"(?:space)?(\d+)",
                str(self.codegen.explicit_resource_register_space(node) or "").strip(),
            )
            if match:
                spaces.append(int(match.group(1)))
        return max(spaces) + 1

    def _plan(self):
        records = self._records()
        if not records:
            return {}
        readers = self._readers({str(node.name) for node in records})
        space = self._space()
        declarations = {}
        for register, node in enumerate(records):
            declarations[id(node)] = self._declare(
                node, register, space, readers.get(str(node.name), [])
            )
        return declarations

    def _refuse(self, node, reason):
        return HlslShaderRecordError(
            str(node.name),
            reason,
            source_location=getattr(node, "source_location", None),
        )

    def _readers(self, record_names):
        """Entry names, in program order, of the stages whose entry or callees read each record."""
        functions = list(getattr(self.ast, "functions", None) or [])
        for _stage_type, stage in getattr(self.ast, "stages", {}).items():
            functions.extend(getattr(stage, "local_functions", None) or [])
        functions_by_name = functions_by_called_name(functions)
        names = self.codegen.stage_entry_names(self.ast, self.target_stage)
        readers = {}
        entries = collect_stage_entry_records(
            self.ast, self.target_stage, self.codegen.stage_entry_types()
        )
        for entry_id, stage_name, entry in entries:
            reached = [entry]
            for name in sorted(
                self.codegen.entry_reachable_function_names(self.ast, entry)
            ):
                reached.extend(functions_by_name.get(name, ()))
            read = sorted(
                {
                    node.name
                    for function in reached
                    for node in self.codegen.walk_ast(getattr(function, "body", None))
                    if isinstance(node, IdentifierNode) and node.name in record_names
                }
            )
            if not read:
                continue
            entry_name = names.get(entry_id) or entry.name
            if stage_name not in _RAY_STAGES:
                raise HlslShaderRecordError(
                    read[0],
                    f"the {stage_name} entry point '{entry_name}' reads it, and only ray tracing stages have a "
                    "shader binding table record",
                )
            if len(read) > 1:
                raise HlslShaderRecordError(
                    read[0],
                    f"entry point '{entry_name}' reads the records {', '.join(read)}, and a stage has one "
                    "shader binding table record",
                )
            readers.setdefault(read[0], []).append(entry_name)
        return readers

    def _layout(self, node):
        struct_name = str(
            self.codegen.resource_base_type(glsl_buffer_block_node_type(node))
        )
        struct = self.codegen.structs_by_name.get(struct_name)
        if struct is None:
            raise self._refuse(node, f"its type '{struct_name}' is not a structure")
        if getattr(node, "var_type", None) is not None and getattr(
            node.var_type, "size", None
        ):
            raise self._refuse(
                node, "a stage has one shader record, not an array of them"
            )

        def members(name):
            held = self.codegen.structs_by_name.get(name)
            if held is None:
                return None
            return [
                self.codegen.hlsl_block_member(member)
                for member in getattr(held, "members", None) or []
                if getattr(member, "name", None)
            ]

        try:
            layout_type = crossgl_struct_layout_type(
                struct_name,
                members(struct_name),
                members,
                (struct_name,),
                self.codegen.hlsl_constant_int_value,
            )
        except HlslLayoutError as exc:
            raise self._refuse(node, str(exc)) from exc
        return struct_name, layout_type

    def _check_layouts(self, node, layout, layout_type):
        matrix = _matrix_path(layout_type)
        if matrix:
            raise self._refuse(
                node,
                f"member '{matrix}' is a matrix, whose orientation CrossTL does not carry into a shader record",
            )
        try:
            declared = _leaves(layout_type, layout)
            _placements_, size = block_struct_placements(layout_type, layout, root=True)
            targets = (
                (
                    "DXC's relaxed std430 rule for SPIR-V shader records",
                    _leaves(layout_type, "relaxed"),
                ),
                (
                    "HLSL constant-buffer packing of Direct3D 12 root constants",
                    _leaves(layout_type, "cbuffer"),
                ),
            )
        except HlslLayoutError as exc:
            raise self._refuse(node, str(exc)) from exc
        for rules, leaves in targets:
            difference = _first_difference(declared, leaves)
            if difference is not None:
                (path, kind, expected), (_path, _kind, actual) = difference
                raise self._refuse(
                    node,
                    f"the {kind} of '{path}' is {expected} bytes under {layout} and {actual} under {rules}",
                )
        if size > MAX_RECORD_BYTES:
            raise self._refuse(
                node,
                f"it takes {size} bytes, and a Direct3D 12 shader record holds at most {MAX_RECORD_BYTES} bytes "
                "of local root arguments after its shader identifier",
            )
        return size

    def _declare(self, node, register, space, readers):
        layout = shader_record_memory_layout(node)
        struct_name, layout_type = self._layout(node)
        size = self._check_layouts(node, layout, layout_type)
        name = str(node.name)
        buffer = f"ConstantBuffer<{struct_name}> {name}"
        code = (
            f"#ifdef __spirv__\n[[vk::shader_record_ext]] {buffer};\n#else\n"
            f"{buffer} : register(b{register}, space{space});\n"
        )
        if readers:
            signature = f"__crossgl_record_signature_{name}"
            words = max(1, -(-size // WORD_BYTES))
            code += (
                f'LocalRootSignature {signature} = {{ "RootConstants(num32BitConstants={words}, '
                f'b{register}, space={space})" }};\n'
                f"SubobjectToExportsAssociation __crossgl_record_exports_{name} = "
                f'{{ "{signature}", "{";".join(readers)}" }};\n'
            )
        return code + "#endif\n"
