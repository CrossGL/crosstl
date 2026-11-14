"""Direct3D float-to-UNORM conversion for HLSL storage texture stores.

Direct3D converts a float written to a UNORM texel by clamping it to
``[0, 1]``, multiplying it by ``2^n - 1`` in single precision and rounding the
product to the nearest integer, halfway cases to even; NaN becomes 0. Some
targets round the exact product instead. Apple GPUs store ``0.3f`` in an
8-bit channel as 77, because ``0.3f * 255`` is 76.500003, where Direct3D
rounds the single-precision product 76.5 to 76.

With ``unorm_store_rounding="single_precision"`` the HLSL frontend quantizes
every value stored to a storage texture declared with a ``unorm`` element
type the way Direct3D converts it. The value stored is then an exactly
representable level, which every target converts to the same integer.

The bit depth of a channel belongs to the bound view, not to the shader. Each
quantized texture therefore gets a constant buffer named
``crosstl_unorm_store_<register>`` (for example ``crosstl_unorm_store_u0`` or
``crosstl_unorm_store_u1_space2``) with one ``float4`` of per-channel scales,
which the host binds with the view: ``2^n - 1`` for an n-bit UNORM channel,
or 0 to store that channel unchanged. The buffers take the ``b`` registers of
space 0 after the highest one the source uses, and target resource binding
records report them by name.
"""

import re

from .DirectxAst import VariableNode

UNORM_STORE_ROUNDINGS = ("target", "single_precision")
SCALE_BUFFER_PREFIX = "crosstl_unorm_store_"
SCALE_MEMBER_PREFIX = "crosstl_unorm_store_scale_"
HELPER_NAME = "crosstl_hlsl_unorm_store"
CHANNEL_HELPER_NAME = "crosstl_hlsl_unorm_store_channel"
UNORM_ELEMENT = re.compile(r"\s*unorm\b")
REGISTER = re.compile(r"\s*(u)(\d+)\s*(?:,\s*space(\d+)\s*)?", re.IGNORECASE)
CONSTANT_REGISTER = re.compile(r"\s*b(\d+)\s*(?:,\s*space(\d+)\s*)?", re.IGNORECASE)


class HLSLUnormStoreRoundingError(ValueError):
    """A UNORM storage texture store cannot be quantized as Direct3D does."""

    project_diagnostic_code = "project.translate.hlsl-unorm-store-rounding-unsupported"
    missing_capabilities = ("hlsl.unorm-store-rounding",)

    def __init__(self, reason):
        self.reason = reason
        super().__init__(
            f"Cannot apply single-precision UNORM store rounding: {reason}"
        )


def scale_buffer_suffix(register):
    """``u0`` for ``register(u0)``, ``u1_space2`` for ``register(u1, space2)``."""
    match = REGISTER.fullmatch(str(register or ""))
    if match is None:
        return None
    suffix = f"u{int(match.group(2))}"
    space = match.group(3)
    if space is not None and int(space) != 0:
        suffix += f"_space{int(space)}"
    return suffix


class HLSLUnormStoreRoundingMixin:
    """Quantize UNORM storage texture stores for ``unorm_store_rounding``."""

    UNORM_STORE_ROUNDINGS = UNORM_STORE_ROUNDINGS

    def unorm_store_rounding_enabled(self):
        return getattr(self, "unorm_store_rounding", "target") == "single_precision"

    def is_unorm_storage_texture_type(self, type_name):
        if not type_name or not self.is_rw_texture_type(type_name):
            return False
        text = str(type_name).strip()
        if "<" not in text or not text.endswith(">"):
            return False
        return UNORM_ELEMENT.match(text.split("<", 1)[1]) is not None

    def collect_unorm_store_textures(self, ast):
        """Record the scale buffer of every global UNORM storage texture."""
        self.unorm_store_textures = {}
        self.unorm_store_used = {}
        self.unorm_store_first_register = 0
        if not self.unorm_store_rounding_enabled():
            return
        for node in [
            *(getattr(ast, "cbuffers", []) or []),
            *(getattr(ast, "global_variables", []) or []),
        ]:
            match = CONSTANT_REGISTER.fullmatch(str(getattr(node, "register", "")))
            if match is not None and int(match.group(2) or 0) == 0:
                self.unorm_store_first_register = max(
                    self.unorm_store_first_register, int(match.group(1)) + 1
                )
        reserved = getattr(self, "integer_helper_reserved_names", set())
        for name in (HELPER_NAME, CHANNEL_HELPER_NAME):
            if name in reserved:
                raise HLSLUnormStoreRoundingError(f"the source already declares {name}")
        for node in getattr(ast, "global_variables", []) or []:
            vtype = getattr(node, "vtype", None)
            if not self.is_unorm_storage_texture_type(vtype):
                continue
            if getattr(node, "array_sizes", None):
                continue
            suffix = scale_buffer_suffix(getattr(node, "register", None))
            if suffix is None:
                continue
            buffer_name = SCALE_BUFFER_PREFIX + suffix
            member_name = SCALE_MEMBER_PREFIX + suffix
            for name in (buffer_name, member_name):
                if name in reserved:
                    raise HLSLUnormStoreRoundingError(
                        f"the source already declares {name}"
                    )
            self.unorm_store_textures[node.name] = (vtype, buffer_name, member_name)

    def unorm_store_texture(self, image_expr):
        """The scale buffer entry of the global texture a store writes."""
        name = None
        if isinstance(image_expr, str):
            name = image_expr
        elif isinstance(image_expr, VariableNode) and isinstance(image_expr.name, str):
            name = image_expr.name
        entry = self.unorm_store_textures.get(name) if name else None
        if entry is None:
            label = name or type(image_expr).__name__
            raise HLSLUnormStoreRoundingError(
                f"{label} is not a global UNORM storage texture with an explicit "
                "u register, so the host cannot bind its channel scales"
            )
        # A parameter or local with the texture's name hides the global.
        if self.current_variable_types.get(name) is not entry[0]:
            raise HLSLUnormStoreRoundingError(
                f"{name} refers to a parameter or local, not the global texture"
            )
        return name, entry

    def unorm_store_value(self, image_expr, resource_type, rendered_value):
        """Wrap a widened ``vec4`` store value in the quantization helper."""
        if not self.unorm_store_rounding_enabled():
            return rendered_value
        if not self.is_unorm_storage_texture_type(resource_type):
            return rendered_value
        name, (_vtype, buffer_name, member_name) = self.unorm_store_texture(image_expr)
        self.unorm_store_used.setdefault(name, (buffer_name, member_name))
        return f"{HELPER_NAME}({rendered_value}, {member_name})"

    def generate_unorm_store_declarations(self):
        """Scale buffers and helpers for the textures stores quantized."""
        used = getattr(self, "unorm_store_used", None)
        if not used:
            return ""
        code = ""
        for index, (buffer_name, member_name) in enumerate(used.values()):
            register = self.unorm_store_first_register + index
            code += (
                f"    @ register(b{register})\n"
                "    @ buffer_layout(hlsl_cbuffer)\n"
                f"    cbuffer {buffer_name} {{\n"
                f"        vec4 {member_name};\n"
                "    }\n"
            )
        # Direct3D: NaN stores 0; otherwise saturate, scale in single
        # precision and round halfway cases to even. A zero scale keeps the
        # value for the target to convert.
        code += (
            "    @ precise\n"
            f"    float {CHANNEL_HELPER_NAME}(float value, float scale) {{\n"
            "        if (!(scale > 0.0)) {\n"
            "            return value;\n"
            "        }\n"
            "        if (isnan(value)) {\n"
            "            return 0.0;\n"
            "        }\n"
            "        float scaled = clamp(value, 0.0, 1.0) * scale;\n"
            "        return roundEven(scaled) / scale;\n"
            "    }\n\n"
            "    @ precise\n"
            f"    vec4 {HELPER_NAME}(vec4 value, vec4 scale) {{\n"
            "        return vec4("
            + ", ".join(f"{CHANNEL_HELPER_NAME}(value.{c}, scale.{c})" for c in "xyzw")
            + ");\n"
            "    }\n\n"
        )
        return code
