"""Represent source union value views with one shared GLSL word allocation."""

from ..ast import ArrayAccessNode, MemberAccessNode, UnaryOpNode
from .array_utils import split_array_type_suffix
from .union_layout import UnionLayoutCollector


class OpenGLUnionLayoutError(ValueError):
    project_diagnostic_code = "project.translate.opengl-union-layout-unsupported"
    missing_capabilities = ("opengl.union-storage-aliasing",)

    def __init__(self, message, **details):
        super().__init__(message)
        for name, value in details.items():
            setattr(self, name, value)


class UnionStorage:
    def __init__(self, generator, structs=()):
        self.generator = generator
        self.layouts = UnionLayoutCollector(
            generator, self.error, generator.glsl_static_struct_member
        ).collect(structs)
        self.helpers = {}

    def error(self, node, *, reason, detail, member=None, **details):
        name = getattr(node, "name", None)
        return OpenGLUnionLayoutError(
            f"OpenGL cannot preserve union '{name}': {detail}",
            union_name=name,
            member_name=getattr(member, "name", None),
            reason=reason,
            source_location=getattr(member, "source_location", None)
            or getattr(node, "source_location", None),
            **details,
        )

    def reject(self, info, reason, detail):
        layout = info["layout"]
        member = info["member"]
        raise self.error(
            layout["node"],
            reason=reason,
            detail=detail,
            member=member["node"],
            member_type=member["type"],
            size=layout["size"],
            alignment=layout["alignment"],
        )

    def layout(self, kind):
        if not self.layouts:
            return None
        g = self.generator
        base, _ = split_array_type_suffix(str(g.type_name_string(kind) or ""))
        while base.startswith(("const ", "volatile ")):
            base = base.split(" ", 1)[1].strip()
        return self.layouts.get(
            g.glsl_source_type_identifier_name(base.rstrip("&*").strip())
        )

    def declaration(self, node):
        layout = self.layouts.get(node.name)
        if layout is None:
            return None
        name = self.generator.glsl_type_identifier_name(node.name)
        return (
            f"struct {name} {{\n"
            f"    {layout['storage_type']} {layout['storage_field']};\n}};\n"
        )

    def member(self, expr):
        if not isinstance(expr, MemberAccessNode):
            return None
        layout = self.layout(self.generator.glsl_source_expression_type(expr.object))
        member = layout["members"].get(expr.member) if layout else None
        if member is None:
            return None
        return {
            "layout": layout,
            "member": member,
            "object": expr.object,
            "index": None,
        }

    def construct(self, expr, kind, arguments):
        g = self.generator
        _, suffix = split_array_type_suffix(str(g.type_name_string(kind) or ""))
        layout = self.layout(kind)
        if layout is None or suffix:
            return None
        constructor = g.map_type(kind)
        if not arguments:
            return f"{constructor}({layout['storage_type']}(0u))"
        if (
            len(arguments) == 1
            and self.layout(g.glsl_source_expression_type(arguments[0])) is layout
        ):
            return g.generate_expression(arguments[0])
        member = next(iter(layout["members"].values()))
        info = {"layout": layout, "member": member, "index": None}
        if len(arguments) != 1 or member["kind"].endswith("_array"):
            self.reject(
                info,
                "union-initializer-shape-unsupported",
                "a union initializer must initialize its first value member",
            )
        value = g.generate_expression_with_expected(arguments[0], member["type"])
        return f"{constructor}({self.encode(info, value)})"

    def access(self, expr):
        info = self.member(expr)
        if info is not None:
            return info
        if isinstance(expr, ArrayAccessNode):
            info = self.member(expr.array)
            if info is not None and info["member"]["kind"].endswith("_array"):
                return {**info, "index": expr.index}
        return None

    def ancestor(self, expr):
        while isinstance(expr, (MemberAccessNode, ArrayAccessNode)):
            info = self.access(expr)
            if info is not None:
                return info
            expr = expr.object if isinstance(expr, MemberAccessNode) else expr.array
        return None

    def storage(self, info):
        g = self.generator
        value = g.generate_expression_with_expected(info["object"], None)
        value = f"{value}.{info['layout']['storage_field']}"
        if info["index"] is not None:
            value += f"[{g.generate_expression_with_expected(info['index'], 'uint')}]"
        return value

    def value_kind(self, info):
        kind = info["member"]["kind"]
        if kind.endswith("_array"):
            if info["index"] is None:
                self.reject(
                    info,
                    "whole-array-view-unsupported",
                    "a union array view cannot escape its storage",
                )
            return kind[:-6]
        return kind

    def word_type(self, info, *, signed=False):
        width = 1 if info["index"] is not None else info["layout"]["word_count"]
        return (
            ("int" if signed else "uint")
            if width == 1
            else f"{'i' if signed else 'u'}vec{width}"
        )

    def encode(self, info, value):
        kind = self.value_kind(info)
        if kind == "float32":
            return f"floatBitsToUint({value})"
        if kind == "int32":
            return f"{self.word_type(info)}({value})"
        if kind in {"u8x4", "bool8x4"}:
            return f"{self.helper('pack', kind)}({value})"
        return value

    def read(self, expr):
        if not self.layouts:
            return None
        if isinstance(expr, UnaryOpNode):
            info = self.ancestor(expr.operand)
            if info is not None:
                op = self.generator.map_operator(expr.op)
                if op == "&" or (
                    op in {"++", "--"} and self.value_kind(info) != "uint32"
                ):
                    self.reject(
                        info,
                        "interpreted-address-or-update-unsupported",
                        "this view is not a canonical word lvalue",
                    )
            return None
        info = self.access(expr)
        if info is None:
            return None
        kind = self.value_kind(info)
        value = self.storage(info)
        if kind == "int32":
            return f"{self.word_type(info, signed=True)}({value})"
        if kind == "float32":
            return f"uintBitsToFloat({value})"
        if kind in {"u8x4", "bool8x4"}:
            return f"{self.helper('unpack', kind)}({value})"
        return value

    def lane_target(self, expr):
        if isinstance(expr, ArrayAccessNode):
            value, lane = expr.array, expr.index
        elif (
            isinstance(expr, MemberAccessNode)
            and expr.member in "xyzwrgba"
            and len(expr.member) == 1
        ):
            value = expr.object
            lane = f"{'xyzwrgba'.index(expr.member) % 4}u"
        else:
            return None
        info = self.access(value)
        if (
            info is not None
            and (
                info["index"] is not None
                or not info["member"]["kind"].endswith("_array")
            )
            and self.value_kind(info) in {"u8x4", "bool8x4"}
        ):
            return info, lane
        return None

    def set_lane(self, info, lane, value):
        g = self.generator
        kind = self.value_kind(info)
        selected = g.generate_expression_with_expected(
            value, "bool" if kind == "bool8x4" else "uint"
        )
        selector = g.generate_expression_with_expected(lane, "uint")
        return (
            f"{self.helper('set', kind)}({self.storage(info)}, {selector}, {selected})"
        )

    def assignment(self, target, value, operator, statement_context):
        if not self.layouts:
            return None
        lane = self.lane_target(target)
        if lane is not None:
            info, selector = lane
            if operator != "=":
                self.reject(
                    info,
                    "byte-lane-compound-write-unsupported",
                    "compound byte writes need a read-modify-write contract",
                )
            return self.set_lane(info, selector, value)
        info = self.access(target)
        if info is not None:
            kind = self.value_kind(info)
            if kind != "uint32" and (operator != "=" or not statement_context):
                self.reject(
                    info,
                    "interpreted-assignment-value-unsupported",
                    "this reinterpreted write needs a discarded simple assignment",
                )
            expected = info["member"]["type"]
            if info["index"] is not None:
                expected, _ = split_array_type_suffix(expected)
            rendered = self.generator.generate_expression_with_expected(value, expected)
            return f"{self.storage(info)} {operator} {self.encode(info, rendered)}"
        info = self.ancestor(target)
        if info is not None and self.value_kind(info) != "uint32":
            self.reject(
                info,
                "interpreted-subcomponent-write-unsupported",
                "this subcomponent is not a canonical word lvalue",
            )
        return None

    def call(self, name, args):
        kinds = {
            "CrossGLMetalVectorIndex_u8vec4_set": "u8x4",
            "CrossGLMetalVectorIndex_bvec4_set": "bool8x4",
        }
        if name not in kinds or len(args) != 3:
            return None
        info = self.access(args[0])
        if info is None or self.value_kind(info) != kinds[name]:
            return None
        return self.set_lane(info, args[1], args[2])

    def validate_argument(self, expr, parameter_type, qualifiers):
        if not set(qualifiers or ()) & {"in", "out", "inout"} and not str(
            parameter_type
        ).rstrip().endswith("&"):
            return
        info = self.ancestor(expr)
        if info is not None:
            self.reject(
                info,
                "member-reference-unsupported",
                "reference helper arguments must preserve overlap with other union views",
            )
        layout = self.layout(self.generator.glsl_source_expression_type(expr))
        if layout is not None:
            raise self.error(
                layout["node"],
                reason="union-reference-unsupported",
                detail="GLSL copy-in/copy-out cannot preserve overlapping union references",
            )

    def validate_buffer(self, kind, seen=None):
        if not self.layouts:
            return
        base, _ = split_array_type_suffix(str(kind))
        base = self.generator.glsl_source_type_identifier_name(base)
        seen = set() if seen is None else seen
        if base in seen:
            return
        seen.add(base)
        layout = self.layouts.get(base)
        if layout is not None:
            raise self.error(
                layout["node"],
                reason="buffer-union-layout-unsupported",
                detail="buffer reflection does not describe overlapping member storage",
            )
        for field in self.generator.struct_member_types.get(base, {}).values():
            self.validate_buffer(field, seen)

    def helper(self, operation, kind):
        key = operation, kind
        if key not in self.helpers:
            self.helpers[key] = self.generator.glsl_generated_module_identifier(
                ("union-storage", *key), f"crossgl_union_{operation}_{kind}"
            )
        return self.helpers[key]

    def helper_definitions(self):
        code = []
        for (operation, kind), name in self.helpers.items():
            boolean = kind == "bool8x4"
            vector, scalar = ("bvec4", "bool") if boolean else ("uvec4", "uint")
            if operation == "unpack":
                words = [f"((word >> {shift}u) & 255u)" for shift in (0, 8, 16, 24)]
                if boolean:
                    words = [f"{word} != 0u" for word in words]
                code.append(
                    f"{vector} {name}(uint word) {{ return {vector}({', '.join(words)}); }}\n"
                )
            elif operation == "pack":
                words = [
                    f"(({'value.' + lane + ' ? 1u : 0u' if boolean else 'value.' + lane + ' & 255u'}) << {8 * index}u)"
                    for index, lane in enumerate("xyzw")
                ]
                code.append(
                    f"uint {name}({vector} value) {{ return {' | '.join(words)}; }}\n"
                )
            else:
                encoded = "(selected ? 1u : 0u)" if boolean else "(selected & 255u)"
                result = "selected" if boolean else "(selected & 255u)"
                code.append(
                    f"{scalar} {name}(inout uint word, uint lane, {scalar} selected) {{\n"
                    "    uint shift = (lane & 3u) * 8u;\n"
                    f"    word = (word & ~(255u << shift)) | ({encoded} << shift);\n"
                    f"    return {result};\n}}\n"
                )
        return "\n".join(code)
