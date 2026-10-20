"""OpenGL storage that reproduces HLSL structured-buffer element layouts.

HLSL structured buffers align every member to its component size, while GLSL
std430 aligns two-component vectors to eight bytes and three- and
four-component vectors to sixteen: ``struct { int2 P; bool B; }`` occupies 12
bytes in HLSL and 16 in std430. HLSL also stores ``bool`` as a 32-bit ``0`` or
``1``. When the element struct of a buffer marked ``@ buffer_layout
(hlsl_structured)`` has a different std430 layout or holds ``bool`` values, the
buffer stores a scalarized copy of the struct instead: vectors become arrays
of scalars and ``bool`` becomes ``uint``, which std430 places exactly like HLSL.
Reads and writes of the buffer convert between that storage and the logical
types; a scalar member keeps direct access, so atomics and compound
assignments on it still work in place.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from ..ast import ArrayAccessNode, IdentifierNode, MemberAccessNode, VariableNode
from .hlsl_buffer_layout import HLSL_STRUCTURED_LAYOUT, buffer_layout_from_attributes

SCALARS = {"int": "int", "uint": "uint", "float": "float", "bool": "bool"}
VECTOR_SCALARS = {"": "float", "i": "int", "u": "uint", "b": "bool"}
SWIZZLE_LANES = {"x": 0, "y": 1, "z": 2, "w": 3, "r": 0, "g": 1, "b": 2, "a": 3}


class OpenGLStructuredLayoutError(ValueError):
    """An HLSL structured-buffer access OpenGL storage cannot reproduce."""

    def __init__(self, message, *, buffer=None, reason=None):
        super().__init__(message)
        self.buffer = buffer
        self.reason = reason


@dataclass
class StorageMember:
    """One logical member and its scalarized storage."""

    name: str
    kind: str  # "scalar", "vector" or "struct"
    scalar: str = ""
    width: int = 1
    logical_type: str = ""
    struct: Optional["StoragePlan"] = None

    def declaration(self):
        storage_scalar = "uint" if self.scalar == "bool" else self.scalar
        if self.kind == "scalar":
            return f"{storage_scalar} {self.name};"
        if self.kind == "vector":
            return f"{storage_scalar} {self.name}[{self.width}];"
        return f"{self.struct.storage_name} {self.name};"


@dataclass
class StoragePlan:
    """Scalarized storage for one logical struct."""

    logical_name: str
    storage_name: str
    members: List[StorageMember] = field(default_factory=list)

    def member(self, name):
        for member in self.members:
            if member.name == name:
                return member
        return None


@dataclass
class StructuredBuffer:
    source_name: str
    glsl_name: str
    plan: StoragePlan


def std430_size_align(kind, width):
    if kind == "scalar":
        return 4, 4
    alignment = 8 if width == 2 else 16
    return 4 * width, alignment


def align_to(value, alignment):
    return ((value + alignment - 1) // alignment) * alignment


class GLSLHlslStructuredBuffers:
    """GLSL code generator support for HLSL structured-buffer layouts."""

    def reset_glsl_hlsl_structured_buffers(self):
        self.glsl_hlsl_structured_buffers: Dict[str, StructuredBuffer] = {}
        self.glsl_hlsl_storage_plans: Dict[str, StoragePlan] = {}
        self.glsl_hlsl_storage_declarations: Dict[str, str] = {}
        self.glsl_hlsl_structured_helpers: Dict[str, str] = {}

    # Planning --------------------------------------------------------------

    def glsl_hlsl_structured_storage_type(self, node, glsl_name, element_type):
        """Storage element type for an HLSL structured buffer, or ``None``.

        Returns ``(storage_type, declarations)`` when the logical element needs
        scalarized storage.
        """
        if not hasattr(self, "glsl_hlsl_structured_buffers"):
            self.reset_glsl_hlsl_structured_buffers()
        attributes = getattr(node, "attributes", None) if node is not None else None
        if buffer_layout_from_attributes(attributes) != HLSL_STRUCTURED_LAYOUT:
            return None
        struct = self.glsl_struct_node(element_type)
        if struct is None:
            return None
        plan = self.glsl_hlsl_storage_plan(element_type)
        if plan is None or not self.glsl_hlsl_storage_required(plan):
            return None
        source_name = getattr(node, "name", None) or glsl_name
        self.glsl_hlsl_structured_buffers[source_name] = StructuredBuffer(
            source_name, glsl_name, plan
        )
        declarations = ""
        for name in self.glsl_hlsl_plan_order(plan):
            if name not in self.glsl_hlsl_storage_declarations:
                text = self.glsl_hlsl_storage_struct_text(
                    self.glsl_hlsl_storage_plans[name]
                )
                self.glsl_hlsl_storage_declarations[name] = text
                declarations += text
        return plan.storage_name, declarations

    def glsl_hlsl_storage_plan(self, type_name, stack=()):
        """Scalarized storage for a struct of 32-bit scalars and vectors."""
        struct = self.glsl_struct_node(type_name)
        if struct is None:
            return None
        logical_name = self.map_type(type_name)
        if logical_name in self.glsl_hlsl_storage_plans:
            return self.glsl_hlsl_storage_plans[logical_name]
        if logical_name in stack:
            return None
        plan = StoragePlan(logical_name, f"crossgl_hlsl_{logical_name}")
        member_types = self.glsl_struct_member_types(type_name)
        for member in getattr(struct, "members", []) or []:
            name = getattr(member, "name", None)
            source_type = self.type_name_string(member_types.get(name))
            if not name or not source_type or "[" in source_type:
                return None
            mapped = self.map_type(source_type)
            if mapped in SCALARS:
                plan.members.append(
                    StorageMember(name, "scalar", mapped, 1, source_type)
                )
                continue
            vector = self.glsl_hlsl_vector_shape(mapped)
            if vector is not None:
                scalar, width = vector
                plan.members.append(
                    StorageMember(name, "vector", scalar, width, source_type)
                )
                continue
            nested = self.glsl_hlsl_storage_plan(source_type, stack + (logical_name,))
            if nested is None:
                return None
            plan.members.append(
                StorageMember(name, "struct", logical_type=source_type, struct=nested)
            )
        self.glsl_hlsl_storage_plans[logical_name] = plan
        return plan

    @staticmethod
    def glsl_hlsl_vector_shape(mapped):
        if not isinstance(mapped, str) or len(mapped) < 4:
            return None
        prefix, rest = mapped[:-4], mapped[-4:]
        if rest[:3] != "vec" or rest[3] not in "234" or prefix not in VECTOR_SCALARS:
            return None
        return VECTOR_SCALARS[prefix], int(rest[3])

    def glsl_hlsl_plan_order(self, plan, seen=None):
        """Storage structs in dependency order."""
        seen = [] if seen is None else seen
        for member in plan.members:
            if member.kind == "struct":
                self.glsl_hlsl_plan_order(member.struct, seen)
        if plan.logical_name not in seen:
            seen.append(plan.logical_name)
        return seen

    def glsl_hlsl_storage_required(self, plan):
        """Whether std430 or ``bool`` storage differs from the HLSL element."""
        if self.glsl_hlsl_plan_has_bool(plan):
            return True
        std430_offsets, std430_size, _alignment = self.glsl_hlsl_std430_layout(plan)
        return (std430_offsets, std430_size) != self.glsl_hlsl_structured_layout(plan)

    def glsl_hlsl_plan_has_bool(self, plan):
        for member in plan.members:
            if member.kind == "struct" and self.glsl_hlsl_plan_has_bool(member.struct):
                return True
            if member.kind != "struct" and member.scalar == "bool":
                return True
        return False

    def glsl_hlsl_std430_layout(self, plan):
        """``(offsets, size, alignment)`` under std430."""
        offsets: List[Tuple[str, int, tuple]] = []
        cursor, max_alignment = 0, 4
        for member in plan.members:
            if member.kind == "struct":
                nested_offsets, size, alignment = self.glsl_hlsl_std430_layout(
                    member.struct
                )
            else:
                nested_offsets = ()
                size, alignment = std430_size_align(member.kind, member.width)
            cursor = align_to(cursor, alignment)
            offsets.append((member.name, cursor, tuple(nested_offsets)))
            cursor += size
            max_alignment = max(max_alignment, alignment)
        return tuple(offsets), align_to(cursor, max_alignment), max_alignment

    def glsl_hlsl_structured_layout(self, plan):
        """``(offsets, size)`` under HLSL structured-buffer rules.

        Every member of a struct of 32-bit values is four-byte aligned, so
        members follow each other without padding.
        """
        offsets: List[Tuple[str, int, tuple]] = []
        cursor = 0
        for member in plan.members:
            if member.kind == "struct":
                nested_offsets, size = self.glsl_hlsl_structured_layout(member.struct)
            else:
                nested_offsets, size = (), 4 * member.width
            offsets.append((member.name, cursor, tuple(nested_offsets)))
            cursor += size
        return tuple(offsets), cursor

    def glsl_hlsl_storage_struct_text(self, plan):
        lines = "".join(f"    {member.declaration()}\n" for member in plan.members)
        return f"struct {plan.storage_name} {{\n{lines}}};\n"

    # Access ------------------------------------------------------------------

    def glsl_hlsl_structured_path(self, expr):
        """``(buffer, element access, steps)`` when ``expr`` reads storage."""
        buffers = getattr(self, "glsl_hlsl_structured_buffers", None)
        if not buffers:
            return None
        steps = []
        node = expr
        while True:
            if isinstance(node, MemberAccessNode):
                steps.append(("member", str(node.member)))
                node = node.object
                continue
            if isinstance(node, ArrayAccessNode):
                root = node.array
                if isinstance(root, (VariableNode, IdentifierNode)):
                    name = getattr(root, "name", None)
                    buffer = buffers.get(name)
                    if buffer is not None and name not in self.local_variable_types:
                        steps.reverse()
                        return buffer, node, steps
                steps.append(("index", node.index))
                node = node.array
                continue
            return None

    def glsl_hlsl_resolve_path(self, buffer, steps):
        """Walk logical members; returns ``(member, lanes, trailing)``.

        ``member`` is ``None`` for the whole element; ``lanes`` selects vector
        components (a list of ints or one index expression).
        """
        plan, member, chain = buffer.plan, None, []
        lanes = None
        for kind, value in steps:
            if lanes is not None:
                self.glsl_hlsl_structured_error(buffer, "nested-component-access")
            if member is not None and member.kind == "vector":
                if kind == "member" and all(ch in SWIZZLE_LANES for ch in value):
                    lanes = [SWIZZLE_LANES[ch] for ch in value]
                elif kind == "index":
                    lanes = value
                else:
                    self.glsl_hlsl_structured_error(buffer, "vector-member-access")
                continue
            if kind != "member":
                self.glsl_hlsl_structured_error(buffer, "indexed-struct-member")
            if member is not None:
                if member.kind != "struct":
                    self.glsl_hlsl_structured_error(buffer, "scalar-member-access")
                plan = member.struct
            member = plan.member(value)
            if member is None:
                self.glsl_hlsl_structured_error(buffer, f"unknown-member-{value}")
            chain.append(member)
        return chain, lanes

    def glsl_hlsl_structured_error(self, buffer, reason):
        raise OpenGLStructuredLayoutError(
            "OpenGL cannot reproduce this access to the HLSL structured buffer "
            f"'{buffer.source_name}': {reason}",
            buffer=buffer.source_name,
            reason=reason,
        )

    def glsl_hlsl_storage_reference(self, buffer, element, chain):
        index = self.glsl_resource_array_index_expression(element)
        path = "".join(f".{member.name}" for member in chain)
        return f"{buffer.glsl_name}[{index}]{path}"

    def glsl_hlsl_value_from_storage(self, member_or_plan, reference):
        """Logical value of a storage member or struct at ``reference``."""
        if isinstance(member_or_plan, StoragePlan):
            values = [
                self.glsl_hlsl_value_from_storage(member, f"{reference}.{member.name}")
                for member in member_or_plan.members
            ]
            return f"{member_or_plan.logical_name}({', '.join(values)})"
        member = member_or_plan
        if member.kind == "struct":
            return self.glsl_hlsl_value_from_storage(member.struct, reference)
        if member.kind == "scalar":
            return f"({reference} != 0u)" if member.scalar == "bool" else reference
        lanes = [f"{reference}[{lane}]" for lane in range(member.width)]
        if member.scalar == "bool":
            lanes = [f"{lane} != 0u" for lane in lanes]
        vector = self.map_type(member.logical_type)
        return f"{vector}({', '.join(lanes)})"

    def glsl_hlsl_store_statements(self, member_or_plan, reference, value):
        if isinstance(member_or_plan, StoragePlan):
            statements = []
            for member in member_or_plan.members:
                statements.extend(
                    self.glsl_hlsl_store_statements(
                        member, f"{reference}.{member.name}", f"{value}.{member.name}"
                    )
                )
            return statements
        member = member_or_plan
        if member.kind == "struct":
            return self.glsl_hlsl_store_statements(member.struct, reference, value)
        if member.kind == "scalar":
            stored = f"uint({value})" if member.scalar == "bool" else value
            return [f"{reference} = {stored};"]
        statements = []
        for lane in range(member.width):
            component = f"{value}[{lane}]"
            stored = f"uint({component})" if member.scalar == "bool" else component
            statements.append(f"{reference}[{lane}] = {stored};")
        return statements

    def glsl_hlsl_helper_suffix(self, chain):
        return "".join(f"_{member.name}" for member in chain)

    def generate_glsl_hlsl_structured_read(self, expr):
        located = self.glsl_hlsl_structured_path(expr)
        if located is None:
            return None
        buffer, element, steps = located
        chain, lanes = self.glsl_hlsl_resolve_path(buffer, steps)
        target = chain[-1] if chain else buffer.plan
        reference = self.glsl_hlsl_storage_reference(buffer, element, chain)
        if lanes is not None:
            if isinstance(lanes, list) and len(lanes) == 1:
                value = f"{reference}[{lanes[0]}]"
            elif isinstance(lanes, list):
                loaded = self.glsl_hlsl_load_helper(buffer, element, chain, target)
                return f"{loaded}.{''.join('xyzw'[lane] for lane in lanes)}"
            else:
                value = f"{reference}[{self.generate_expression(lanes)}]"
            return f"({value} != 0u)" if target.scalar == "bool" else value
        if isinstance(target, StorageMember) and target.kind == "scalar":
            return self.glsl_hlsl_value_from_storage(target, reference)
        return self.glsl_hlsl_load_helper(buffer, element, chain, target)

    def glsl_hlsl_load_helper(self, buffer, element, chain, target):
        suffix = self.glsl_hlsl_helper_suffix(chain)
        name = f"crossgl_hlsl_load_{buffer.glsl_name}{suffix}"
        if isinstance(target, StoragePlan):
            result_type = target.logical_name
        elif target.kind == "struct":
            result_type = target.struct.logical_name
        else:
            result_type = self.map_type(target.logical_type)
        path = "".join(f".{member.name}" for member in chain)
        value = self.glsl_hlsl_value_from_storage(
            target, f"{buffer.glsl_name}[index]{path}"
        )
        self.glsl_hlsl_structured_helpers.setdefault(
            name,
            f"{result_type} {name}(uint index) {{\n    return {value};\n}}\n",
        )
        index = self.glsl_resource_array_index_expression(element)
        return f"{name}(uint({index}))"

    def generate_glsl_hlsl_structured_assignment(
        self, target, value, op, *, statement_context
    ):
        located = self.glsl_hlsl_structured_path(target)
        if located is None:
            return None
        buffer, element, steps = located
        chain, lanes = self.glsl_hlsl_resolve_path(buffer, steps)
        member = chain[-1] if chain else buffer.plan
        reference = self.glsl_hlsl_storage_reference(buffer, element, chain)
        logical_type = self.glsl_source_expression_type(target)
        rendered = self.generate_expression_with_expected(value, logical_type)
        is_bool = isinstance(member, StorageMember) and member.scalar == "bool"
        if lanes is not None:
            if isinstance(lanes, list) and len(lanes) != 1:
                self.glsl_hlsl_structured_error(buffer, "swizzle-store")
            lane = (
                lanes[0] if isinstance(lanes, list) else self.generate_expression(lanes)
            )
            reference = f"{reference}[{lane}]"
            if is_bool:
                if op != "=":
                    self.glsl_hlsl_structured_error(buffer, "boolean-compound-store")
                return f"{reference} = uint({rendered})"
            return f"{reference} {op} {rendered}"
        if isinstance(member, StorageMember) and member.kind == "scalar":
            if is_bool:
                if op != "=":
                    self.glsl_hlsl_structured_error(buffer, "boolean-compound-store")
                return f"{reference} = uint({rendered})"
            return f"{reference} {op} {rendered}"
        if op != "=":
            self.glsl_hlsl_structured_error(buffer, "aggregate-compound-store")
        if not statement_context:
            self.glsl_hlsl_structured_error(buffer, "aggregate-store-value-used")
        if isinstance(member, StoragePlan):
            value_type = member.logical_name
        elif member.kind == "struct":
            value_type = member.struct.logical_name
        else:
            value_type = self.map_type(member.logical_type)
        suffix = self.glsl_hlsl_helper_suffix(chain)
        name = f"crossgl_hlsl_store_{buffer.glsl_name}{suffix}"
        path = "".join(f".{item.name}" for item in chain)
        statements = self.glsl_hlsl_store_statements(
            member, f"{buffer.glsl_name}[index]{path}", "value"
        )
        body = "".join(f"    {statement}\n" for statement in statements)
        self.glsl_hlsl_structured_helpers.setdefault(
            name,
            f"void {name}(uint index, {value_type} value) {{\n{body}}}\n",
        )
        index = self.glsl_resource_array_index_expression(element)
        return f"{name}(uint({index}), {rendered})"

    def generate_glsl_hlsl_structured_helpers(self):
        helpers = getattr(self, "glsl_hlsl_structured_helpers", {}) or {}
        return "".join(text + "\n" for text in helpers.values())
