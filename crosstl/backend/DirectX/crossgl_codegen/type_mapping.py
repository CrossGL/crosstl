"""Type and semantic mapping in the HLSL to CrossGL converter: minimum-precision, composite, matrix, template vector and matrix, rasterizer-ordered, feedback and RW texture types, and semantics."""

import ast as python_ast
import re

from crosstl.translator.codegen.stage_distance_layout import distance_semantic_spelling
from crosstl.translator.ray_query_vocabulary import canonical_ray_query_type

from ..hlsl_matrix_import import crossgl_matrix_type_name
from ..import_tables import CONSERVATIVE_DEPTH_LAYOUTS


class TypeMappingMixin:
    """Type mapping members of ``HLSLToCrossGLConverter``."""

    @staticmethod
    def minimum_precision_type_map():
        type_map = {}
        vector_aliases = {
            "min16float": "f16vec",
            "min10float": "f16vec",
            "float16_t": "f16vec",
            "min16int": "i16vec",
            "min12int": "i16vec",
            "int16_t": "i16vec",
            "min16uint": "u16vec",
            "uint16_t": "u16vec",
        }
        for hlsl_prefix, crossgl_prefix in vector_aliases.items():
            for width in range(2, 5):
                type_map[f"{hlsl_prefix}{width}"] = f"{crossgl_prefix}{width}"

        for hlsl_prefix in ("min16float", "min10float", "float16_t"):
            for rows in range(2, 5):
                for cols in range(2, 5):
                    type_map[f"{hlsl_prefix}{rows}x{cols}"] = crossgl_matrix_type_name(
                        hlsl_prefix, rows, cols
                    )

        return type_map

    def map_type(self, hlsl_type):
        """Map an HLSL type name to the closest CrossGL type name.

        HLSL ``floatRxC`` is CrossGL ``matCxR``, the same matrix (ADR 0004),
        and the integer ``intRxC`` keeps its spelling. With row-major packing
        CrossGL holds the transpose of every HLSL matrix (HLSL rows become
        columns, see :meth:`generate_mul_expression`), so a non-square HLSL
        ``floatRxC`` becomes ``matRxC`` and ``intRxC`` becomes ``intCxR``.
        """
        ray_query = canonical_ray_query_type(hlsl_type)
        if ray_query is not None:
            return ray_query
        mapped = self.map_type_preserving_matrix_shape(hlsl_type)
        if self.matrix_packing != "row_major" or not isinstance(mapped, str):
            return mapped
        transposed = re.fullmatch(
            r"((?:f16|d)?mat|int|uint|bool)([2-4])x([2-4])", mapped
        )
        if transposed is None or transposed.group(2) == transposed.group(3):
            return mapped
        prefix = transposed.group(1)
        return f"{prefix}{transposed.group(3)}x{transposed.group(2)}"

    def canonical_composite_type(self, type_name):
        """Normalize HLSL C-style signedness spellings before CrossGL mapping."""
        text = str(type_name).strip()
        fixed_width_aliases = {
            "float32_t": "float",
            "float64_t": "double",
            "int32_t": "int",
            "uint32_t": "uint",
            # HLSL declares the packed 8-bit dot-product types as uint aliases.
            "int8_t4_packed": "uint",
            "uint8_t4_packed": "uint",
        }
        if text in fixed_width_aliases:
            return fixed_width_aliases[text]

        vector_alias_match = re.fullmatch(
            r"(float32_t|float64_t|int32_t|uint32_t)([1-4])", text
        )
        if vector_alias_match:
            scalar, width = vector_alias_match.groups()
            scalar_prefix = {
                "float32_t": "float",
                "float64_t": "double",
                "int32_t": "int",
                "uint32_t": "uint",
            }[scalar]
            return f"{scalar_prefix}{width}"

        matrix_alias_match = re.fullmatch(r"(float64_t)([1-4])x([1-4])", text)
        if matrix_alias_match:
            scalar, rows, cols = matrix_alias_match.groups()
            scalar_prefix = {
                "float64_t": "double",
            }[scalar]
            return f"{scalar_prefix}{rows}x{cols}"

        match = re.fullmatch(r"(signed|unsigned)\s+(.+)", text)
        if not match:
            return text

        signedness, base = match.groups()
        base = base.strip()
        if signedness == "unsigned":
            if base == "int":
                return "uint"
            if base == "int1":
                return "uint1"
            if base == "dword":
                return "uint"
            if base == "min16int":
                return "min16uint"
            if base == "int16_t":
                return "uint16_t"
            if base == "int32_t":
                return "uint32_t"
            if base == "int64_t":
                return "uint64_t"
            if re.fullmatch(r"int[2-4]", base):
                return "u" + base
            dword_vector_match = re.fullmatch(r"dword([1-4])", base)
            if dword_vector_match:
                return f"uint{dword_vector_match.group(1)}"
            min16_vector_match = re.fullmatch(r"min16int([1-4])", base)
            if min16_vector_match:
                return f"min16uint{min16_vector_match.group(1)}"
            fixed_width_vector_match = re.fullmatch(r"int(16|32|64)_t([1-4])", base)
            if fixed_width_vector_match:
                width, lanes = fixed_width_vector_match.groups()
                return f"uint{width}_t{lanes}"
            int_matrix_match = re.fullmatch(r"int([1-4])x([1-4])", base)
            if int_matrix_match:
                rows, cols = int_matrix_match.groups()
                return f"uint{rows}x{cols}"
            dword_matrix_match = re.fullmatch(r"dword([1-4])x([1-4])", base)
            if dword_matrix_match:
                rows, cols = dword_matrix_match.groups()
                return f"uint{rows}x{cols}"
            if base in {"uint", "dword"} or re.fullmatch(r"uint[2-4]", base):
                return base
            if re.fullmatch(r"uint[1-4]x[1-4]", base):
                return base

        if signedness == "signed":
            if (
                base == "int"
                or base == "int1"
                or re.fullmatch(r"int[2-4]", base)
                or re.fullmatch(r"int[1-4]x[1-4]", base)
            ):
                return base

        dword_vector_match = re.fullmatch(r"dword([1-4])", text)
        if dword_vector_match:
            return f"uint{dword_vector_match.group(1)}"

        dword_matrix_match = re.fullmatch(r"dword([1-4])x([1-4])", text)
        if dword_matrix_match:
            rows, cols = dword_matrix_match.groups()
            return f"uint{rows}x{cols}"

        return text

    def map_hlsl_matrix_alias_type(self, type_name):
        matrix_match = re.fullmatch(
            r"(float|half|fixed|double|min16float|min10float|float16_t|int|uint|bool|"
            r"min16int|min12int|int16_t|min16uint|uint16_t)([1-4])x([1-4])",
            str(type_name).strip(),
        )
        if not matrix_match:
            return None

        scalar_type, rows_text, cols_text = matrix_match.groups()
        rows = int(rows_text)
        cols = int(cols_text)
        if rows == 1 and cols == 1:
            return self.map_template_vector_type(scalar_type, 1)
        if rows == 1:
            return self.map_template_vector_type(scalar_type, cols)
        if cols == 1:
            return self.map_template_vector_type(scalar_type, rows)
        return None

    def sanitize_scoped_identifier_name(self, name):
        """Convert HLSL scoped function declarations into CrossGL identifiers."""
        sanitized = str(name).strip(":")
        sanitized = (
            sanitized.replace("operator()", "operator_call")
            .replace("operator[]", "operator_index")
            .replace("operator ", "operator_")
            .replace("::", "_")
        )
        sanitized = re.sub(r"\W+", "_", sanitized).strip("_")
        if not sanitized:
            return "operator_overload"
        if sanitized[0].isdigit():
            sanitized = f"_{sanitized}"
        return sanitized

    def map_template_vector_or_matrix_type(self, base_type, generic_type):
        args = self.split_generic_arguments(generic_type)
        if base_type == "vector":
            scalar_type = args[0] if args else "float"
            components = self.parse_template_dimension(args[1]) if len(args) > 1 else 4
            return self.map_template_vector_type(scalar_type, components)
        if base_type == "matrix":
            scalar_type = args[0] if args else "float"
            rows = self.parse_template_dimension(args[1]) if len(args) > 1 else 4
            cols = self.parse_template_dimension(args[2]) if len(args) > 2 else 4
            return self.map_template_matrix_type(scalar_type, rows, cols)
        return None

    def split_generic_arguments(self, generic_type):
        args = []
        depth = 0
        current = []
        for char in generic_type:
            if char == "<":
                depth += 1
            elif char == ">" and depth:
                depth -= 1
            elif char == "," and depth == 0:
                args.append("".join(current).strip())
                current = []
                continue
            current.append(char)
        if current or generic_type.strip():
            args.append("".join(current).strip())
        return args

    def parse_template_dimension(self, value):
        try:
            return int(str(value).strip())
        except (TypeError, ValueError):
            return self.evaluate_integer_constant_expression_text(value)

    def evaluate_integer_constant_expression_text(self, value):
        text = str(value).strip()
        if not text:
            return None
        text = re.sub(r"(?i)(?<=\d)(ull|llu|ul|lu|ll|u|l)\b", "", text)
        try:
            expression = python_ast.parse(text, mode="eval").body
        except (SyntaxError, ValueError):
            return None
        return self.evaluate_integer_python_ast(expression)

    def evaluate_integer_python_ast(self, node):
        if isinstance(node, python_ast.Constant):
            if isinstance(node.value, bool) or not isinstance(node.value, int):
                return None
            return node.value
        if isinstance(node, python_ast.Name):
            return self.integer_constant_values.get(node.id)
        if isinstance(node, python_ast.UnaryOp):
            value = self.evaluate_integer_python_ast(node.operand)
            if value is None:
                return None
            if isinstance(node.op, python_ast.UAdd):
                return value
            if isinstance(node.op, python_ast.USub):
                return -value
            if isinstance(node.op, python_ast.Invert):
                return ~value
            return None
        if isinstance(node, python_ast.BinOp):
            left = self.evaluate_integer_python_ast(node.left)
            right = self.evaluate_integer_python_ast(node.right)
            if left is None or right is None:
                return None
            return self.apply_integer_python_operator(left, node.op, right)
        return None

    def apply_integer_python_operator(self, left, operator, right):
        if isinstance(operator, python_ast.Add):
            return left + right
        if isinstance(operator, python_ast.Sub):
            return left - right
        if isinstance(operator, python_ast.Mult):
            return left * right
        if isinstance(operator, (python_ast.Div, python_ast.FloorDiv)) and right != 0:
            return left // right
        if isinstance(operator, python_ast.Mod) and right != 0:
            return left % right
        if isinstance(operator, python_ast.LShift):
            return left << right
        if isinstance(operator, python_ast.RShift):
            return left >> right
        if isinstance(operator, python_ast.BitAnd):
            return left & right
        if isinstance(operator, python_ast.BitOr):
            return left | right
        if isinstance(operator, python_ast.BitXor):
            return left ^ right
        return None

    def map_template_vector_type(self, scalar_type, components):
        if components is None or components < 1 or components > 4:
            return None
        scalar = self.canonical_composite_type(scalar_type)
        if components == 1:
            return self.map_type(scalar)
        if scalar in {"int64_t", "uint64_t"}:
            return self.map_type(f"{scalar}{components}")
        prefixes = {
            "float": "vec",
            "half": "f16vec",
            "fixed": "vec",
            "min16float": "f16vec",
            "min10float": "f16vec",
            "float16_t": "f16vec",
            "double": "dvec",
            "int": "ivec",
            "min16int": "i16vec",
            "min12int": "i16vec",
            "int16_t": "i16vec",
            "uint": "uvec",
            "min16uint": "u16vec",
            "uint16_t": "u16vec",
            "bool": "bvec",
        }
        prefix = prefixes.get(scalar)
        if prefix is None:
            if scalar in getattr(self, "current_template_type_names", ()):
                # ``vector<T, 3>`` of a function template is CrossGL's ``vec3<T>``.
                return f"vec{components}<{scalar}>"
            return None
        return f"{prefix}{components}"

    def map_template_matrix_type(self, scalar_type, rows, cols):
        if rows is None or cols is None or rows < 1 or rows > 4 or cols < 1 or cols > 4:
            return None
        if rows == 1 and cols == 1:
            return self.map_template_vector_type(scalar_type, 1)
        if rows == 1:
            return self.map_template_vector_type(scalar_type, cols)
        if cols == 1:
            return self.map_template_vector_type(scalar_type, rows)
        scalar_type = self.canonical_composite_type(scalar_type)
        prefixes = {
            "float": "mat",
            "half": "f16mat",
            "fixed": "mat",
            "min16float": "f16mat",
            "min10float": "f16mat",
            "float16_t": "f16mat",
            "double": "dmat",
        }
        if prefixes.get(str(scalar_type).strip()) is None:
            return None
        return crossgl_matrix_type_name(str(scalar_type).strip(), rows, cols)

    def map_rasterizer_ordered_buffer_type(self, base_type, element_type):
        buffer_type = {
            "RasterizerOrderedBuffer": "RWBuffer",
            "RasterizerOrderedStructuredBuffer": "RWStructuredBuffer",
        }.get(base_type)
        if buffer_type is None:
            return None
        return f"{buffer_type}<{self.canonical_composite_type(element_type)}>"

    def map_feedback_texture_type(self, base_type, feedback_type):
        texture_type = {
            "FeedbackTexture2D": "feedbackTexture2D",
            "FeedbackTexture2DArray": "feedbackTexture2DArray",
        }.get(base_type)
        if texture_type is None:
            return None
        return f"{texture_type}<{feedback_type}>"

    def map_rw_texture_type(self, base_type, element_type):
        image_type = {
            "RWTexture1D": "image1D",
            "RWTexture1DArray": "image1DArray",
            "RWTexture2D": "image2D",
            "RWTexture2DArray": "image2DArray",
            "RWTexture2DMS": "image2DMS",
            "RWTexture2DMSArray": "image2DMSArray",
            "RWTexture3D": "image3D",
            "RWTextureCube": "imageCube",
            "RWTextureCubeArray": "imageCubeArray",
            "RasterizerOrderedTexture1D": "image1D",
            "RasterizerOrderedTexture1DArray": "image1DArray",
            "RasterizerOrderedTexture2D": "image2D",
            "RasterizerOrderedTexture2DArray": "image2DArray",
            "RasterizerOrderedTexture3D": "image3D",
        }.get(base_type)
        if image_type is None:
            return None

        element = self.canonical_composite_type(element_type)
        if element.startswith("uint"):
            return f"u{image_type}"
        if element.startswith("int"):
            return f"i{image_type}"
        return image_type

    def is_fragment_position_input_parameter(
        self, semantic, function_qualifier=None, parameter=None
    ):
        if not semantic or str(semantic).upper() != "SV_POSITION":
            return False
        if str(function_qualifier or "").lower() not in {"fragment", "pixel"}:
            return False
        qualifiers = {
            str(qualifier).lower()
            for qualifier in getattr(parameter, "qualifiers", []) or []
        }
        return not qualifiers.intersection({"out", "inout"})

    def is_fragment_coverage_input_parameter(
        self, semantic, function_qualifier=None, parameter=None
    ):
        if not semantic or str(semantic).upper() != "SV_COVERAGE":
            return False
        if str(function_qualifier or "").lower() not in {"fragment", "pixel"}:
            return False
        qualifiers = {
            str(qualifier).lower()
            for qualifier in getattr(parameter, "qualifiers", []) or []
        }
        return not qualifiers.intersection({"out", "inout"})

    def is_barycentric_semantic(self, semantic):
        return bool(
            semantic and re.fullmatch(r"SV_BARYCENTRICS\d*", str(semantic).upper())
        )

    def is_noperspective_barycentric(self, semantic, node=None):
        if not self.is_barycentric_semantic(semantic):
            return False
        qualifiers = {
            str(qualifier).lower()
            for qualifier in getattr(node, "qualifiers", []) or []
        }
        return any("noperspective" in qualifier for qualifier in qualifiers)

    def map_semantic(self, semantic, function_qualifier=None, parameter=None):
        """Map an HLSL semantic to CrossGL semantic annotation syntax."""
        if not semantic:
            return ""
        payload_access = []
        if isinstance(semantic, str):
            for access_name, access_args in re.findall(
                r"\b(read|write)\s*\(([^)]*)\)", semantic
            ):
                args = ", ".join(
                    arg.strip() for arg in access_args.split(",") if arg.strip()
                )
                payload_access.append(f"@ ray_payload_{access_name}({args})")
        if payload_access:
            return " ".join(payload_access)
        if self.is_fragment_position_input_parameter(
            semantic, function_qualifier, parameter
        ):
            return "@ gl_FragCoord"
        if self.is_fragment_coverage_input_parameter(
            semantic, function_qualifier, parameter
        ):
            return "@ gl_SampleMaskIn"
        if self.is_barycentric_semantic(semantic):
            mapped = (
                "gl_BaryCoordNoPerspEXT"
                if self.is_noperspective_barycentric(semantic, parameter)
                else "gl_BaryCoordEXT"
            )
            return f"@ {mapped}"
        if isinstance(semantic, str):
            # SV_DepthGreaterEqual and SV_DepthLessEqual write gl_FragDepth with GLSL's conservative depth layout,
            # which CrossGL spells as a depth_greater or depth_less attribute next to the built-in.
            conservative_depth = CONSERVATIVE_DEPTH_LAYOUTS.get(semantic.upper())
            if conservative_depth is not None:
                return f"@ gl_FragDepth @ {conservative_depth}"
        mapped = self.semantic_map.get(semantic)
        if mapped is None and isinstance(semantic, str):
            semantic_upper = semantic.upper()
            mapped = self.semantic_map.get(semantic_upper)
            if mapped is None:
                mapped = self.semantic_map_upper.get(semantic_upper)
            if mapped is None:
                texcoord_match = re.fullmatch(r"TEXCOORD(\d+)", semantic_upper)
                if texcoord_match:
                    mapped = f"TexCoord{texcoord_match.group(1)}"
            if mapped is None:
                color_match = re.fullmatch(r"COLOR(\d+)", semantic_upper)
                if color_match:
                    mapped = f"Color{color_match.group(1)}"
            if mapped is None:
                target_match = re.fullmatch(r"SV_TARGET(\d*)", semantic_upper)
                if target_match:
                    target_index = target_match.group(1)
                    mapped = (
                        f"gl_FragData[{target_index}]"
                        if target_index
                        else "gl_FragColor"
                    )
            if mapped is None:
                depth_match = re.fullmatch(
                    r"SV_DEPTH(?:GREATEREQUAL|LESSEQUAL)?", semantic_upper
                )
                if depth_match:
                    mapped = "gl_FragDepth"
            if mapped is None:
                for hlsl_prefix, crossgl_prefix in (
                    ("NORMAL", "Normal"),
                    ("TANGENT", "Tangent"),
                    ("BINORMAL", "Binormal"),
                    ("BLENDINDICES", "BlendIndices"),
                    ("BLENDWEIGHT", "BlendWeight"),
                ):
                    indexed_match = re.fullmatch(rf"{hlsl_prefix}(\d+)", semantic_upper)
                    if indexed_match:
                        mapped = f"{crossgl_prefix}{indexed_match.group(1)}"
                        break
            if mapped is None:
                for hlsl_prefix, crossgl_semantic in (
                    ("SV_CLIPDISTANCE", "gl_ClipDistance"),
                    ("SV_CULLDISTANCE", "gl_CullDistance"),
                ):
                    if not semantic_upper.startswith(hlsl_prefix):
                        continue
                    suffix = semantic_upper[len(hlsl_prefix) :]
                    if not suffix or suffix.isdigit():
                        # The semantic index orders the value in the target's distance array.
                        mapped = distance_semantic_spelling(
                            crossgl_semantic, int(suffix or 0)
                        )
                        break
        mapped = mapped or semantic
        return f"@ {mapped}"
