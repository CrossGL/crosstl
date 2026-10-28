"""Packed sub-word lane views of canonical union storage.

A union such as ``union { uchar val[4]; uint bits; }`` keeps one canonical
32-bit word per four bytes. Lane ``i`` of a sub-word array view lives in bits
``(i % lanes_per_word) * width`` of word ``i / lanes_per_word`` (little-endian,
as the union contract requires). Reads extract and sign- or bit-convert the
lane; writes replace only that lane and return the stored lane value, so an
assignment keeps its value in expression context.

The helpers take the whole storage and the lane index once, so an index with
side effects is evaluated exactly once.
"""

from .array_utils import split_array_type_suffix

INTEGER_ELEMENTS = {"uint8", "int8", "uint16", "int16"}
SIGNED_ELEMENTS = {"int8", "int16"}


class LaneHelpers:
    """Collect and emit lane accessors for one target dialect."""

    def __init__(self, dialect, value_type, identifier):
        if dialect not in {"hlsl", "glsl"}:
            raise ValueError(f"unknown lane dialect {dialect!r}")
        self.dialect = dialect
        self.value_type = value_type
        self.identifier = identifier
        self.used = {}

    @staticmethod
    def kind(member):
        return member["element"], int(member["lane_bits"])

    def storage_type(self, words):
        if words == 1:
            return "uint"
        return f"uint{words}" if self.dialect == "hlsl" else f"uvec{words}"

    def name(self, operation, member, words, reject):
        element, bits = self.kind(member)
        key = (operation, element, bits, int(words))
        if key not in self.used:
            mapped = self.value_type(element)
            self.check_value_type(element, mapped, reject)
            self.used[key] = (
                self.identifier(operation, f"{element}_lane_w{words}"),
                mapped,
            )
        return self.used[key][0]

    def check_value_type(self, element, mapped, reject):
        if self.dialect == "hlsl":
            allowed = {
                "uint8": {"uint"},
                "int8": {"int"},
                "uint16": {"uint", "uint16_t"},
                "int16": {"int", "int16_t"},
                "float16": {"float", "half", "float16_t"},
            }[element]
        else:
            allowed = {
                "uint8": {"uint"},
                "int8": {"int"},
                "uint16": {"uint"},
                "int16": {"int"},
                "float16": {"float"},
            }[element]
        if mapped not in allowed:
            reject(
                "lane-value-type-unsupported",
                f"{element} lanes have no exact encoding for target type '{mapped}'",
            )

    def encode(self, element, bits, mapped):
        mask = "0xffu" if bits == 8 else "0xffffu"
        if self.dialect == "glsl":
            mask = "255u" if bits == 8 else "65535u"
        if element in INTEGER_ELEMENTS:
            return f"(uint(selected) & {mask})"
        if self.dialect == "glsl":
            return f"(packHalf2x16(vec2(selected, 0.0)) & {mask})"
        if mapped == "float":
            return f"(f32tof16(selected) & {mask})"
        return "uint(asuint16(selected))"

    def decode(self, element, bits, mapped, value):
        if element in SIGNED_ELEMENTS:
            if self.dialect == "glsl":
                return f"bitfieldExtract(int({value}), 0, {bits})"
            rest = 32 - bits
            signed = f"(asint({value} << {rest}u) >> {rest})"
            return signed if mapped == "int" else f"{mapped}({signed})"
        if element in INTEGER_ELEMENTS:
            return value if mapped == "uint" else f"{mapped}({value})"
        if self.dialect == "glsl":
            return f"unpackHalf2x16({value}).x"
        if mapped == "float":
            return f"f16tof32({value})"
        return f"asfloat16(uint16_t({value}))"

    def definitions(self):
        code = []
        for key in sorted(self.used):
            operation, element, bits, words = key
            name, mapped = self.used[key]
            storage_type = self.storage_type(words)
            lanes = 32 // bits
            mask = (
                ("0xffu" if bits == 8 else "0xffffu")
                if self.dialect == "hlsl"
                else ("255u" if bits == 8 else "65535u")
            )
            word = "storage" if words == 1 else f"storage[index / {lanes}u]"
            shift = f"    uint shift = (index % {lanes}u) * {bits}u;\n"
            if operation == "get":
                value = self.decode(element, bits, mapped, "bits")
                code.append(
                    f"{mapped} {name}({storage_type} storage, uint index) {{\n"
                    f"{shift}"
                    f"    uint bits = ({word} >> shift) & {mask};\n"
                    f"    return {value};\n}}\n"
                )
            else:
                value = self.decode(element, bits, mapped, "bits")
                code.append(
                    f"{mapped} {name}(inout {storage_type} storage, uint index, "
                    f"{mapped} selected) {{\n"
                    f"{shift}"
                    f"    uint bits = {self.encode(element, bits, mapped)};\n"
                    f"    {word} = ({word} & ~({mask} << shift)) | (bits << shift);\n"
                    f"    return {value};\n}}\n"
                )
        return "\n".join(code)


def hlsl_lane_helpers(generator):
    return LaneHelpers(
        "hlsl",
        generator.map_type,
        lambda operation, kind: f"__crossgl_union_{operation}_{kind}",
    )


def hlsl_lane_reject(generator, info):
    def reject(reason, detail):
        raise generator.hlsl_union_usage_error(info, None, reason, detail)

    return reject


def hlsl_lane_helper_definitions(generator):
    """Emit the get/set helpers the generated functions referenced."""
    lanes = getattr(generator, "hlsl_union_lanes", None)
    definitions = lanes.definitions() if lanes is not None else ""
    return f"\n{definitions}\n" if definitions else ""


def hlsl_lane_call(generator, info, operation, value=None):
    """Render a packed lane read or a lane write that yields the stored lane."""
    layout = info["layout"]
    lanes = getattr(generator, "hlsl_union_lanes", None)
    if lanes is None:
        lanes = generator.hlsl_union_lanes = hlsl_lane_helpers(generator)
    name = lanes.name(
        operation,
        info["member"],
        layout["word_count"],
        hlsl_lane_reject(generator, info),
    )
    storage = generator.hlsl_union_storage_expression(info)
    index = generator.generate_expression_with_expected(info["index"], "uint")
    if operation == "get":
        return f"{name}({storage}, {index})"
    element, _suffix = split_array_type_suffix(info["member"]["type"])
    rendered = generator.generate_expression_with_expected(value, element)
    return f"{name}({storage}, {index}, {rendered})"
