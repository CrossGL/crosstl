"""CrossGL-to-Mojo code generator."""

import re

from ..ast import (
    ArrayNode,
    ArrayAccessNode,
    ArrayLiteralNode,
    AssignmentNode,
    BinaryOpNode,
    BreakNode,
    CaseNode,
    CbufferNode,
    ContinueNode,
    DoWhileNode,
    EnumNode,
    ForInNode,
    ForNode,
    FunctionCallNode,
    FunctionNode,
    IfNode,
    LiteralPatternNode,
    LoopNode,
    MatchNode,
    MemberAccessNode,
    ReturnNode,
    RangeNode,
    ShaderNode,
    StructNode,
    SwitchNode,
    TernaryOpNode,
    UnaryOpNode,
    VariableNode,
    WhileNode,
    WildcardPatternNode,
)
from .array_utils import parse_array_type, format_array_type, get_array_size_from_node

MOJO_VECTOR_TYPES = {
    "vec2": ("DType.float32", 2, 2, None),
    "vec3": ("DType.float32", 3, 4, "0.0"),
    "vec4": ("DType.float32", 4, 4, None),
    "vec2<f32>": ("DType.float32", 2, 2, None),
    "vec3<f32>": ("DType.float32", 3, 4, "0.0"),
    "vec4<f32>": ("DType.float32", 4, 4, None),
    "vec2<f64>": ("DType.float64", 2, 2, None),
    "vec3<f64>": ("DType.float64", 3, 4, "0.0"),
    "vec4<f64>": ("DType.float64", 4, 4, None),
    "vec2<i32>": ("DType.int32", 2, 2, None),
    "vec3<i32>": ("DType.int32", 3, 4, "0"),
    "vec4<i32>": ("DType.int32", 4, 4, None),
    "vec2<u32>": ("DType.uint32", 2, 2, None),
    "vec3<u32>": ("DType.uint32", 3, 4, "0"),
    "vec4<u32>": ("DType.uint32", 4, 4, None),
    "vec2<bool>": ("DType.bool", 2, 2, None),
    "vec3<bool>": ("DType.bool", 3, 4, "False"),
    "vec4<bool>": ("DType.bool", 4, 4, None),
    "ivec2": ("DType.int32", 2, 2, None),
    "ivec3": ("DType.int32", 3, 4, "0"),
    "ivec4": ("DType.int32", 4, 4, None),
    "uvec2": ("DType.uint32", 2, 2, None),
    "uvec3": ("DType.uint32", 3, 4, "0"),
    "uvec4": ("DType.uint32", 4, 4, None),
    "dvec2": ("DType.float64", 2, 2, None),
    "dvec3": ("DType.float64", 3, 4, "0.0"),
    "dvec4": ("DType.float64", 4, 4, None),
    "bvec2": ("DType.bool", 2, 2, None),
    "bvec3": ("DType.bool", 3, 4, "False"),
    "bvec4": ("DType.bool", 4, 4, None),
    "bool2": ("DType.bool", 2, 2, None),
    "bool3": ("DType.bool", 3, 4, "False"),
    "bool4": ("DType.bool", 4, 4, None),
}

MOJO_MATRIX_TYPES = {
    "mat2": ("DType.float32", 2, 2),
    "mat3": ("DType.float32", 3, 3),
    "mat4": ("DType.float32", 4, 4),
    "mat2x2": ("DType.float32", 2, 2),
    "mat2x3": ("DType.float32", 2, 3),
    "mat2x4": ("DType.float32", 2, 4),
    "mat3x2": ("DType.float32", 3, 2),
    "mat3x3": ("DType.float32", 3, 3),
    "mat3x4": ("DType.float32", 3, 4),
    "mat4x2": ("DType.float32", 4, 2),
    "mat4x3": ("DType.float32", 4, 3),
    "mat4x4": ("DType.float32", 4, 4),
    "dmat2": ("DType.float64", 2, 2),
    "dmat3": ("DType.float64", 3, 3),
    "dmat4": ("DType.float64", 4, 4),
    "dmat2x2": ("DType.float64", 2, 2),
    "dmat2x3": ("DType.float64", 2, 3),
    "dmat2x4": ("DType.float64", 2, 4),
    "dmat3x2": ("DType.float64", 3, 2),
    "dmat3x3": ("DType.float64", 3, 3),
    "dmat3x4": ("DType.float64", 3, 4),
    "dmat4x2": ("DType.float64", 4, 2),
    "dmat4x3": ("DType.float64", 4, 3),
    "dmat4x4": ("DType.float64", 4, 4),
}

SWIZZLE_SETS = {
    "xyzw": {"x": 0, "y": 1, "z": 2, "w": 3},
    "rgba": {"r": 0, "g": 1, "b": 2, "a": 3},
}

MOJO_DTYPE_INFO = {
    "DType.float32": ("float", "vec", "0.0"),
    "DType.float64": ("double", "dvec", "0.0"),
    "DType.int32": ("int", "ivec", "0"),
    "DType.uint32": ("uint", "uvec", "0"),
    "DType.bool": ("bool", "bvec", "False"),
}

MOJO_DTYPE_SUFFIX = {
    "DType.float32": "f32",
    "DType.float64": "f64",
    "DType.int32": "i32",
    "DType.uint32": "u32",
    "DType.bool": "bool",
}

MOJO_SCALAR_DTYPES = {
    "float": "DType.float32",
    "double": "DType.float64",
    "int": "DType.int32",
    "uint": "DType.uint32",
    "bool": "DType.bool",
}

MOJO_RESOURCE_TYPE_MAPPING = {
    "sampler1D": "Texture1D",
    "sampler1DArray": "Texture1DArray",
    "sampler2D": "Texture2D",
    "sampler2DArray": "Texture2DArray",
    "sampler2DMS": "Texture2DMS",
    "sampler2DMSArray": "Texture2DMSArray",
    "sampler3D": "Texture3D",
    "samplerCube": "TextureCube",
    "samplerCubeArray": "TextureCubeArray",
    "sampler": "Sampler",
    "image1D": "Image1D",
    "image1DArray": "Image1DArray",
    "image2D": "Image2D",
    "image2DArray": "Image2DArray",
    "image2DMS": "Image2DMS",
    "image2DMSArray": "Image2DMSArray",
    "image3D": "Image3D",
    "imageCube": "ImageCube",
    "iimage1D": "IImage1D",
    "iimage1DArray": "IImage1DArray",
    "iimage2D": "IImage2D",
    "iimage2DArray": "IImage2DArray",
    "iimage2DMS": "IImage2DMS",
    "iimage2DMSArray": "IImage2DMSArray",
    "iimage3D": "IImage3D",
    "uimage1D": "UImage1D",
    "uimage1DArray": "UImage1DArray",
    "uimage2D": "UImage2D",
    "uimage2DArray": "UImage2DArray",
    "uimage2DMS": "UImage2DMS",
    "uimage2DMSArray": "UImage2DMSArray",
    "uimage3D": "UImage3D",
}

MOJO_RESOURCE_SAMPLE_COORDS = {
    "Texture1D": "Float32",
    "Texture1DArray": "SIMD[DType.float32, 2]",
    "Texture2D": "SIMD[DType.float32, 2]",
    "Texture2DArray": "SIMD[DType.float32, 4]",
    "Texture3D": "SIMD[DType.float32, 4]",
    "TextureCube": "SIMD[DType.float32, 4]",
    "TextureCubeArray": "SIMD[DType.float32, 4]",
}

MOJO_RESOURCE_TEXEL_COORDS = {
    "Texture1D": "Int32",
    "Texture1DArray": "SIMD[DType.int32, 2]",
    "Texture2D": "SIMD[DType.int32, 2]",
    "Texture2DArray": "SIMD[DType.int32, 4]",
    "Texture2DMS": "SIMD[DType.int32, 2]",
    "Texture2DMSArray": "SIMD[DType.int32, 4]",
    "Texture3D": "SIMD[DType.int32, 4]",
    "TextureCube": "SIMD[DType.int32, 4]",
    "TextureCubeArray": "SIMD[DType.int32, 4]",
    "Image1D": "Int32",
    "Image1DArray": "SIMD[DType.int32, 2]",
    "Image2D": "SIMD[DType.int32, 2]",
    "Image2DArray": "SIMD[DType.int32, 4]",
    "Image2DMS": "SIMD[DType.int32, 2]",
    "Image2DMSArray": "SIMD[DType.int32, 4]",
    "Image3D": "SIMD[DType.int32, 4]",
    "ImageCube": "SIMD[DType.int32, 4]",
    "IImage1D": "Int32",
    "IImage1DArray": "SIMD[DType.int32, 2]",
    "IImage2D": "SIMD[DType.int32, 2]",
    "IImage2DArray": "SIMD[DType.int32, 4]",
    "IImage2DMS": "SIMD[DType.int32, 2]",
    "IImage2DMSArray": "SIMD[DType.int32, 4]",
    "IImage3D": "SIMD[DType.int32, 4]",
    "UImage1D": "Int32",
    "UImage1DArray": "SIMD[DType.int32, 2]",
    "UImage2D": "SIMD[DType.int32, 2]",
    "UImage2DArray": "SIMD[DType.int32, 4]",
    "UImage2DMS": "SIMD[DType.int32, 2]",
    "UImage2DMSArray": "SIMD[DType.int32, 4]",
    "UImage3D": "SIMD[DType.int32, 4]",
}

MOJO_RESOURCE_SIZE_RETURNS = {
    "Texture1D": "Int32",
    "Texture1DArray": "SIMD[DType.int32, 2]",
    "Texture2D": "SIMD[DType.int32, 2]",
    "Texture2DArray": "SIMD[DType.int32, 4]",
    "Texture2DMS": "SIMD[DType.int32, 2]",
    "Texture2DMSArray": "SIMD[DType.int32, 4]",
    "Texture3D": "SIMD[DType.int32, 4]",
    "TextureCube": "SIMD[DType.int32, 2]",
    "TextureCubeArray": "SIMD[DType.int32, 4]",
    "Image1D": "Int32",
    "Image1DArray": "SIMD[DType.int32, 2]",
    "Image2D": "SIMD[DType.int32, 2]",
    "Image2DArray": "SIMD[DType.int32, 4]",
    "Image2DMS": "SIMD[DType.int32, 2]",
    "Image2DMSArray": "SIMD[DType.int32, 4]",
    "Image3D": "SIMD[DType.int32, 4]",
    "ImageCube": "SIMD[DType.int32, 2]",
    "IImage1D": "Int32",
    "IImage1DArray": "SIMD[DType.int32, 2]",
    "IImage2D": "SIMD[DType.int32, 2]",
    "IImage2DArray": "SIMD[DType.int32, 4]",
    "IImage2DMS": "SIMD[DType.int32, 2]",
    "IImage2DMSArray": "SIMD[DType.int32, 4]",
    "IImage3D": "SIMD[DType.int32, 4]",
    "UImage1D": "Int32",
    "UImage1DArray": "SIMD[DType.int32, 2]",
    "UImage2D": "SIMD[DType.int32, 2]",
    "UImage2DArray": "SIMD[DType.int32, 4]",
    "UImage2DMS": "SIMD[DType.int32, 2]",
    "UImage2DMSArray": "SIMD[DType.int32, 4]",
    "UImage3D": "SIMD[DType.int32, 4]",
}

MOJO_INTEGER_INDEX_TYPES = {"int", "uint", "short", "ushort", "long", "ulong"}

MOJO_VECTOR_ARITHMETIC_OPS = {
    "+": "add",
    "-": "sub",
    "*": "mul",
    "/": "div",
}


class MojoCodeGen:
    """Emit Mojo-like shader source from the shared CrossGL AST."""

    def __init__(self):
        """Initialize Mojo type maps and helper-generation state."""
        self.vector_constructor_info = MOJO_VECTOR_TYPES
        self.struct_types = {}
        self.function_return_types = {}
        self.variable_types = {}
        self.enum_types = {}
        self.enum_variant_aliases = {}
        self.enum_variant_values = {}
        self.current_enum_value_aliases = {}
        self.required_resource_types = set()
        self.required_resource_sample_types = set()
        self.required_resource_lod_types = set()
        self.required_resource_grad_types = set()
        self.required_resource_size_types = set()
        self.required_resource_query_level_types = set()
        self.required_resource_texel_fetch_types = set()
        self.required_image_load_types = set()
        self.required_image_store_types = set()
        self.required_helpers = set()
        self.required_splat_helpers = set()
        self.required_swizzle_helpers = set()
        self.required_constructor_helpers = {}
        self.required_select_helpers = set()
        self.required_matrix_types = set()
        self.required_matrix_constructor_helpers = {}
        self.required_fract_helpers = set()
        self.required_saturate_helpers = set()
        self.current_return_type = None
        self.current_shader = None
        self.do_while_contexts = []
        self.for_contexts = []
        self.loop_depth = 0
        self.do_while_counter = 0
        self.lambda_counter = 0
        self.expression_prelude_stack = []
        self.type_mapping = {
            # Scalar Types
            "void": "None",
            "int": "Int32",
            "short": "Int16",
            "long": "Int64",
            "uint": "UInt32",
            "ushort": "UInt16",
            "ulong": "UInt64",
            "float": "Float32",
            "double": "Float64",
            "half": "Float16",
            "bool": "Bool",
            "string": "String",
            "char": "String",
            **{
                name: f"SIMD[{dtype}, {storage_width}]"
                for name, (dtype, _, storage_width, _) in MOJO_VECTOR_TYPES.items()
            },
            **{
                name: self.matrix_type_name(dtype, columns, rows)
                for name, (dtype, columns, rows) in MOJO_MATRIX_TYPES.items()
            },
            # Texture/resource placeholders for Mojo compile-smoke support.
            **MOJO_RESOURCE_TYPE_MAPPING,
        }
        self.scalar_constructor_map = {
            name: mapped
            for name, mapped in self.type_mapping.items()
            if mapped
            in {
                "Bool",
                "Float16",
                "Float32",
                "Float64",
                "Int16",
                "Int32",
                "Int64",
                "String",
                "UInt16",
                "UInt32",
                "UInt64",
            }
        }

        self.semantic_map = {
            # Vertex attributes
            "gl_VertexID": "vertex_id",
            "gl_InstanceID": "instance_id",
            "gl_Position": "position",
            "gl_PointSize": "point_size",
            "gl_ClipDistance": "clip_distance",
            # Fragment attributes
            "gl_FragColor": "color(0)",
            "gl_FragColor0": "color(0)",
            "gl_FragColor1": "color(1)",
            "gl_FragColor2": "color(2)",
            "gl_FragColor3": "color(3)",
            "gl_FragDepth": "depth(any)",
            "gl_FragCoord": "position",
            "gl_FrontFacing": "front_facing",
            "gl_PointCoord": "point_coord",
            # Standard vertex semantics
            "POSITION": "position",
            "NORMAL": "normal",
            "TANGENT": "tangent",
            "BINORMAL": "binormal",
            "TEXCOORD": "texcoord",
            "TEXCOORD0": "texcoord0",
            "TEXCOORD1": "texcoord1",
            "TEXCOORD2": "texcoord2",
            "TEXCOORD3": "texcoord3",
            "COLOR": "color",
            "COLOR0": "color0",
            "COLOR1": "color1",
        }

        # Function mapping for common shader functions
        self.function_map = {
            "texture": "sample",
            "normalize": "normalize",
            "dot": "dot_product",
            "cross": "cross_product",
            "length": "magnitude",
            "reflect": "reflect",
            "refract": "refract",
            "sin": "sin",
            "cos": "cos",
            "tan": "tan",
            "sqrt": "sqrt",
            "inversesqrt": "rsqrt",
            "pow": "power",
            "abs": "abs",
            "min": "min",
            "max": "max",
            "clamp": "clamp",
            "mix": "lerp",
            "smoothstep": "smoothstep",
            "step": "step",
        }

    def generate(self, ast):
        """Generate complete Mojo-like shader source for a CrossGL AST."""
        self.struct_types = {}
        self.function_return_types = {}
        self.variable_types = {}
        self.enum_types = {}
        self.enum_variant_aliases = {}
        self.enum_variant_values = {}
        self.current_enum_value_aliases = {}
        self.required_resource_types = set()
        self.required_resource_sample_types = set()
        self.required_resource_lod_types = set()
        self.required_resource_grad_types = set()
        self.required_resource_size_types = set()
        self.required_resource_query_level_types = set()
        self.required_resource_texel_fetch_types = set()
        self.required_image_load_types = set()
        self.required_image_store_types = set()
        self.required_helpers = set()
        self.required_splat_helpers = set()
        self.required_swizzle_helpers = set()
        self.required_constructor_helpers = {}
        self.required_select_helpers = set()
        self.required_matrix_types = set()
        self.required_matrix_constructor_helpers = {}
        self.required_fract_helpers = set()
        self.required_saturate_helpers = set()
        self.current_return_type = None
        self.do_while_contexts = []
        self.for_contexts = []
        self.loop_depth = 0
        self.do_while_counter = 0
        self.lambda_counter = 0
        self.expression_prelude_stack = []
        self.collect_function_return_types(ast)

        header = "# Generated Mojo Shader Code\n"
        header += "from math import *\n"
        header += "from simd import *\n"
        header += "from gpu import *\n\n"
        code = ""

        structs = getattr(ast, "structs", [])
        for node in structs:
            if isinstance(node, EnumNode):
                code += self.generate_enum(node)
                continue
            if isinstance(node, StructNode):
                code += self.generate_struct(node)

        global_vars = getattr(ast, "global_variables", [])
        for node in global_vars:
            if isinstance(node, ArrayNode):
                code += self.generate_array_declaration(node)
            else:
                if hasattr(node, "initial_value") and node.initial_value is not None:
                    vtype = self.variable_declared_type(node)
                    self.register_variable_type(
                        node.name,
                        vtype or self.expression_result_type(node.initial_value),
                    )
                    if (
                        isinstance(node.initial_value, ArrayLiteralNode)
                        and vtype is not None
                        and self.is_array_type_name(vtype)
                    ):
                        init_expr = self.generate_array_literal_expression(
                            node.initial_value, vtype
                        )
                    else:
                        init_expr = self.generate_expression(node.initial_value)
                    if vtype is None:
                        code += f"var {node.name} = {init_expr}\n"
                        continue
                    code += f"var {node.name}: {self.map_type(vtype)} = {init_expr}\n"
                    continue

                # Handle both old and new AST variable structures
                vtype = self.variable_declared_type(node) or "float"
                self.register_variable_type(node.name, vtype)
                if self.is_array_type_name(vtype):
                    code += (
                        f"var {node.name} = "
                        f"{self.array_initial_value_for_type(vtype)}\n"
                    )
                elif self.is_struct_type_name(vtype):
                    code += f"var {node.name} = {self.zero_value_for_type(vtype)}\n"
                elif self.is_resource_type_name(vtype):
                    mapped_type = self.map_type(vtype)
                    code += (
                        f"var {node.name}: {mapped_type} = "
                        f"{self.zero_value_for_type(vtype)}\n"
                    )
                else:
                    code += f"var {node.name}: {self.map_type(vtype)}\n"

        cbuffers = getattr(ast, "cbuffers", [])
        if cbuffers:
            code += "# Constant Buffers\n"
            code += self.generate_cbuffers(ast)

        functions = getattr(ast, "functions", [])
        for func in functions:
            # Handle both old and new AST function structures
            if hasattr(func, "qualifiers") and func.qualifiers:
                qualifier = func.qualifiers[0] if func.qualifiers else None
            else:
                qualifier = getattr(func, "qualifier", None)

            if qualifier == "vertex":
                code += "# Vertex Shader\n"
                code += self.generate_function(func, shader_type="vertex")
            elif qualifier == "fragment":
                code += "# Fragment Shader\n"
                code += self.generate_function(func, shader_type="fragment")
            elif qualifier == "compute":
                code += "# Compute Shader\n"
                code += self.generate_function(func, shader_type="compute")
            else:
                code += self.generate_function(func)

        # Handle shader stages (new AST structure)
        if hasattr(ast, "stages") and ast.stages:
            emitted_local_functions = set()
            for stage_type, stage in ast.stages.items():
                if hasattr(stage, "entry_point"):
                    stage_name = (
                        str(stage_type).split(".")[-1].lower()
                    )  # Extract stage name from enum
                    code += f"# {stage_name.title()} Shader\n"
                    for func in getattr(stage, "local_functions", []):
                        if id(func) in emitted_local_functions:
                            continue
                        code += self.generate_function(func)
                        emitted_local_functions.add(id(func))
                    code += self.generate_function(
                        stage.entry_point, shader_type=stage_name
                    )

        return header + self.generate_required_helpers() + code

    def collect_function_return_types(self, ast):
        functions = list(getattr(ast, "functions", []))
        stages = getattr(ast, "stages", {})
        if stages:
            for stage in stages.values():
                entry_point = getattr(stage, "entry_point", None)
                if entry_point is not None:
                    functions.append(entry_point)
                functions.extend(getattr(stage, "local_functions", []))

        for func in functions:
            self.register_function_return_type(func)

    def register_function_return_type(self, func):
        if not hasattr(func, "name"):
            return

        if hasattr(func, "return_type"):
            return_type = self.convert_type_node_to_string(func.return_type)
        else:
            return_type = "void"
        self.function_return_types[func.name] = return_type

    def is_user_defined_function(self, func_name):
        return isinstance(func_name, str) and func_name in self.function_return_types

    def convert_type_node_to_string(self, type_node) -> str:
        """Convert new AST TypeNode to string representation."""
        if type_node.__class__.__name__ == "ArrayType":
            element_type = self.convert_type_node_to_string(type_node.element_type)
            size = self.format_array_size(type_node.size)
            return (
                f"{element_type}[{size}]" if size is not None else f"{element_type}[]"
            )
        if hasattr(type_node, "name"):
            generic_args = getattr(type_node, "generic_args", [])
            if generic_args:
                args = ", ".join(
                    self.convert_type_node_to_string(arg) for arg in generic_args
                )
                return f"{type_node.name}<{args}>"
            return type_node.name
        elif hasattr(type_node, "element_type") and hasattr(type_node, "size"):
            element_type = self.convert_type_node_to_string(type_node.element_type)
            size = type_node.size
            if element_type == "float":
                return f"vec{size}"
            elif element_type == "int":
                return f"ivec{size}"
            elif element_type == "uint":
                return f"uvec{size}"
            elif element_type == "double":
                return f"dvec{size}"
            elif element_type == "bool":
                return f"bvec{size}"
            else:
                return f"{element_type}{size}"
        elif hasattr(type_node, "element_type") and hasattr(type_node, "rows"):
            element_type = self.convert_type_node_to_string(type_node.element_type)
            prefix = "dmat" if element_type == "double" else "mat"
            if type_node.rows == type_node.cols:
                return f"{prefix}{type_node.rows}"
            return f"{prefix}{type_node.rows}x{type_node.cols}"
        else:
            return str(type_node)

    def variable_declared_type(self, node):
        """Return the explicit type on a variable declaration, if one exists."""
        var_type = getattr(node, "var_type", None)
        if var_type is not None:
            return self.convert_type_node_to_string(var_type)

        member_type = getattr(node, "member_type", None)
        if member_type is not None:
            return self.convert_type_node_to_string(member_type)

        vtype = getattr(node, "vtype", None)
        if vtype is None or vtype == "":
            return None
        if hasattr(vtype, "name") or hasattr(vtype, "element_type"):
            return self.convert_type_node_to_string(vtype)
        return vtype

    def format_array_size(self, size):
        if size is None:
            return None
        if hasattr(size, "value"):
            return size.value
        return size

    def extract_semantic_from_attributes(self, attributes):
        """Extract semantic information from new AST attributes."""
        semantic_attrs = [
            "position",
            "color",
            "texcoord",
            "normal",
            "tangent",
            "binormal",
            "POSITION",
            "COLOR",
            "TEXCOORD",
            "NORMAL",
            "TANGENT",
            "BINORMAL",
            "TEXCOORD0",
            "TEXCOORD1",
            "TEXCOORD2",
            "TEXCOORD3",
        ]

        for attr in attributes:
            if hasattr(attr, "name") and attr.name in semantic_attrs:
                return attr.name
        return None

    def generate_enum(self, node):
        """Lower a unit/numeric CrossGL enum to Mojo type and value aliases."""
        enum_type = self.map_enum_underlying_type(
            getattr(node, "underlying_type", None)
        )
        self.enum_types[node.name] = enum_type

        code = f"alias {node.name} = {enum_type}\n"
        next_value = 0
        local_aliases = {}
        local_values = {}
        for variant in getattr(node, "variants", []) or []:
            payload = getattr(variant, "data", None) or getattr(variant, "fields", None)
            if payload:
                raise ValueError(
                    "Unsupported enum payload for Mojo codegen; only unit/numeric "
                    f"enum variants are supported: {node.name}.{variant.name}"
                )

            alias_name = self.enum_variant_alias_name(node.name, variant.name)
            value = getattr(variant, "value", None)
            if value is None:
                value_text = str(next_value)
                resolved_value = next_value
            else:
                value_text = self.generate_enum_value_expression(value, local_aliases)
                resolved_value = self.evaluate_enum_integer_value(value, local_values)

            self.enum_variant_aliases[f"{node.name}::{variant.name}"] = alias_name
            self.enum_variant_aliases[f"{node.name}.{variant.name}"] = alias_name
            self.enum_variant_values[f"{node.name}::{variant.name}"] = resolved_value
            self.enum_variant_values[f"{node.name}.{variant.name}"] = resolved_value
            local_aliases[variant.name] = alias_name
            local_values[variant.name] = resolved_value
            code += f"alias {alias_name} = {value_text}\n"

            literal_value = resolved_value
            if literal_value is None:
                literal_value = self.literal_int_value(value_text)
            next_value = (
                literal_value + 1 if literal_value is not None else next_value + 1
            )

        return code + "\n"

    def generate_enum_value_expression(self, value, local_aliases):
        previous_aliases = self.current_enum_value_aliases
        self.current_enum_value_aliases = local_aliases
        try:
            return self.generate_expression(value)
        finally:
            self.current_enum_value_aliases = previous_aliases

    def evaluate_enum_integer_value(self, expr, local_values):
        if expr is None:
            return None
        if isinstance(expr, bool):
            return int(expr)
        if isinstance(expr, int):
            return expr
        if hasattr(expr, "value"):
            try:
                return int(expr.value)
            except (TypeError, ValueError):
                return None
        expr_name = getattr(expr, "name", None)
        if expr_name is not None and not isinstance(expr, VariableNode):
            return local_values.get(expr_name)
        if isinstance(expr, MemberAccessNode):
            reference = self.enum_member_reference_name(expr)
            if reference is None:
                return None
            return self.enum_variant_values.get(reference)
        if isinstance(expr, UnaryOpNode):
            operand = self.evaluate_enum_integer_value(expr.operand, local_values)
            if operand is None:
                return None
            op = self.map_operator(expr.op)
            if op == "+":
                return operand
            if op == "-":
                return -operand
            if op == "~":
                return ~operand
            return None
        if isinstance(expr, BinaryOpNode):
            left = self.evaluate_enum_integer_value(expr.left, local_values)
            right = self.evaluate_enum_integer_value(expr.right, local_values)
            if left is None or right is None:
                return None
            op = self.map_operator(expr.op)
            try:
                return self.evaluate_enum_binary_op(left, op, right)
            except ZeroDivisionError:
                return None
        return None

    def evaluate_enum_binary_op(self, left, op, right):
        if op == "+":
            return left + right
        if op == "-":
            return left - right
        if op == "*":
            return left * right
        if op == "/":
            return left // right
        if op == "%":
            return left % right
        if op == "<<":
            return left << right
        if op == ">>":
            return left >> right
        if op == "|":
            return left | right
        if op == "&":
            return left & right
        if op == "^":
            return left ^ right
        return None

    def enum_member_reference_name(self, expr):
        if not isinstance(expr, MemberAccessNode):
            return None
        obj = getattr(expr, "object", None)
        obj_name = getattr(obj, "name", obj if isinstance(obj, str) else None)
        if obj_name is None:
            return None
        return f"{obj_name}.{expr.member}"

    def map_enum_underlying_type(self, underlying_type):
        if underlying_type is None:
            return "Int32"
        mapped = self.map_type(self.convert_type_node_to_string(underlying_type))
        if mapped in {"Int", "Int16", "Int32", "Int64", "UInt16", "UInt32", "UInt64"}:
            return mapped
        return "Int32"

    def enum_variant_alias_name(self, enum_name, variant_name):
        return f"{enum_name}_{variant_name}"

    def map_enum_variant_reference(self, name):
        if not isinstance(name, str):
            return name
        if name in self.current_enum_value_aliases:
            return self.current_enum_value_aliases[name]
        return self.enum_variant_aliases.get(name, name)

    def generate_struct(self, node):
        code = f"@value\nstruct {node.name}:\n"
        self.struct_types[node.name] = {}

        members = getattr(node, "members", [])
        for member in members:
            if isinstance(member, ArrayNode):
                element_type = getattr(
                    member, "element_type", getattr(member, "vtype", "float")
                )
                size = get_array_size_from_node(member)
                self.struct_types[node.name][member.name] = self.array_type_name(
                    element_type, size
                )
                code += (
                    f"    var {member.name}: "
                    f"{self.array_storage_type(element_type, size)}\n"
                )
            else:
                if hasattr(member, "member_type"):
                    member_type = self.convert_type_node_to_string(member.member_type)
                elif hasattr(member, "vtype"):
                    member_type = member.vtype
                else:
                    member_type = "float"

                self.struct_types[node.name][member.name] = member_type

                semantic = None
                if hasattr(member, "semantic"):
                    semantic = member.semantic
                elif hasattr(member, "attributes"):
                    semantic = self.extract_semantic_from_attributes(member.attributes)

                semantic_comment = (
                    f"  # {self.map_semantic(semantic)}" if semantic else ""
                )
                code += f"    var {member.name}: {self.map_type(member_type)}{semantic_comment}\n"

        code += "\n"
        return code

    def generate_cbuffers(self, ast):
        code = ""
        cbuffers = getattr(ast, "cbuffers", [])
        for node in cbuffers:
            if isinstance(node, StructNode):
                code += f"@value\nstruct {node.name}:\n"
                members = getattr(node, "members", [])
                for member in members:
                    if isinstance(member, ArrayNode):
                        element_type = getattr(
                            member, "element_type", getattr(member, "vtype", "float")
                        )
                        size = get_array_size_from_node(member)
                        code += (
                            f"    var {member.name}: "
                            f"{self.array_storage_type(element_type, size)}\n"
                        )
                    else:
                        member_type = self.variable_declared_type(member) or "float"
                        code += f"    var {member.name}: {self.map_type(member_type)}\n"
                code += "\n"
            elif hasattr(node, "name") and hasattr(node, "members"):  # CbufferNode
                code += f"@value\nstruct {node.name}:\n"
                for member in node.members:
                    if isinstance(member, ArrayNode):
                        element_type = getattr(
                            member, "element_type", getattr(member, "vtype", "float")
                        )
                        size = get_array_size_from_node(member)
                        code += (
                            f"    var {member.name}: "
                            f"{self.array_storage_type(element_type, size)}\n"
                        )
                    else:
                        member_type = self.variable_declared_type(member) or "float"
                        code += f"    var {member.name}: {self.map_type(member_type)}\n"
                code += "\n"
        return code

    def generate_function(self, func, indent=0, shader_type=None):
        """Render one CrossGL function or shader entry point as Mojo code."""
        code = ""
        "    " * indent
        previous_variable_types = self.variable_types.copy()
        previous_return_type = self.current_return_type

        param_list = getattr(func, "parameters", getattr(func, "params", []))
        param_names = {p.name for p in param_list if hasattr(p, "name")}
        mutated_params = self.collect_mutated_parameters(
            getattr(func, "body", []), param_names
        )
        params = []
        for p in param_list:
            if hasattr(p, "param_type"):
                param_type = self.convert_type_node_to_string(p.param_type)
            elif hasattr(p, "vtype"):
                param_type = p.vtype
            else:
                param_type = "float"

            semantic = None
            if hasattr(p, "semantic"):
                semantic = p.semantic
            elif hasattr(p, "attributes"):
                semantic = self.extract_semantic_from_attributes(p.attributes)

            self.register_variable_type(p.name, param_type)
            param_semantic = f"  # {self.map_semantic(semantic)}" if semantic else ""
            ownership = "owned " if p.name in mutated_params else ""
            params.append(
                f"{ownership}{p.name}: {self.map_type(param_type)}{param_semantic}"
            )

        params_str = ", ".join(params) if params else ""

        if hasattr(func, "return_type"):
            return_type = self.convert_type_node_to_string(func.return_type)
        else:
            return_type = "void"
        self.function_return_types[func.name] = return_type
        self.current_return_type = return_type

        if shader_type == "vertex":
            code += f"@vertex_shader\n"
        elif shader_type == "fragment":
            code += f"@fragment_shader\n"
        elif shader_type == "compute":
            code += f"@compute_shader\n"

        code += f"fn {func.name}({params_str}) -> {self.map_type(return_type)}:\n"

        body = getattr(func, "body", [])
        statements = None
        if hasattr(body, "statements"):
            statements = body.statements
            for stmt in body.statements:
                code += self.generate_statement(stmt, indent + 1)
        elif isinstance(body, list):
            statements = body
            for stmt in body:
                code += self.generate_statement(stmt, indent + 1)
        else:
            code += "    pass\n"

        if statements is not None and not statements:
            code += "    pass\n"

        code += "\n"
        self.variable_types = previous_variable_types
        self.current_return_type = previous_return_type
        return code

    def collect_mutated_parameters(self, body, param_names):
        mutated = set()
        for stmt in self.body_statements(body):
            self.collect_mutated_parameters_from_node(stmt, param_names, mutated)
        return mutated

    def body_statements(self, body):
        if hasattr(body, "statements"):
            return body.statements
        if isinstance(body, list):
            return body
        if body is None:
            return []
        return [body]

    def collect_mutated_parameters_from_node(self, node, param_names, mutated):
        if node is None:
            return

        if isinstance(node, AssignmentNode):
            root_name = self.assignment_target_root(node.left)
            if root_name in param_names:
                mutated.add(root_name)
            self.collect_mutated_parameters_from_node(node.right, param_names, mutated)
            return

        if isinstance(node, UnaryOpNode) and self.map_operator(node.op) in ["++", "--"]:
            root_name = self.assignment_target_root(node.operand)
            if root_name in param_names:
                mutated.add(root_name)
            return

        for child in self.node_children(node):
            self.collect_mutated_parameters_from_node(child, param_names, mutated)

    def node_children(self, node):
        children = []
        for attr in (
            "init",
            "condition",
            "update",
            "body",
            "then_branch",
            "if_body",
            "else_branch",
            "else_body",
            "value",
            "expression",
            "left",
            "right",
            "object",
            "object_expr",
            "array",
            "array_expr",
            "index",
            "index_expr",
            "operand",
            "vector_expr",
        ):
            if hasattr(node, attr):
                children.append(getattr(node, attr))

        for attr in ("statements", "args", "arguments"):
            if hasattr(node, attr):
                children.extend(getattr(node, attr))
        if isinstance(node, ArrayLiteralNode):
            children.extend(node.elements)

        return children

    def assignment_target_root(self, target):
        if isinstance(target, str):
            return target
        if isinstance(target, VariableNode) and hasattr(target, "name"):
            return target.name
        if isinstance(target, ArrayAccessNode):
            return self.assignment_target_root(target.array)
        if isinstance(target, MemberAccessNode):
            return self.assignment_target_root(target.object)
        if hasattr(target, "__class__") and "Identifier" in str(target.__class__):
            return getattr(target, "name", None)
        if hasattr(target, "__class__") and "Swizzle" in str(target.__class__):
            return self.assignment_target_root(getattr(target, "vector_expr", None))
        return None

    def generate_expression_with_prelude(self, expr, indent):
        self.expression_prelude_stack.append({"indent": indent, "lines": []})
        try:
            expression = self.generate_expression(expr)
            prelude = "".join(self.expression_prelude_stack[-1]["lines"])
            return prelude, expression
        finally:
            self.expression_prelude_stack.pop()

    def generate_assignment_statement(self, node, indent):
        indent_str = "    " * indent
        left = self.generate_expression(node.left)
        left_type = self.expression_result_type(node.left)
        if isinstance(node.right, ArrayLiteralNode) and self.is_array_type_name(
            left_type
        ):
            prelude = ""
            right = self.generate_array_literal_expression(node.right, left_type)
        else:
            prelude, right = self.generate_expression_with_prelude(node.right, indent)
        op = self.map_operator(node.operator)
        return f"{prelude}{indent_str}{left} {op} {right}\n"

    def generate_statement(self, stmt, indent=0):
        """Render a single CrossGL statement as Mojo code."""
        indent_str = "    " * indent

        if isinstance(stmt, VariableNode):
            var_type = self.variable_declared_type(stmt)
            if getattr(stmt, "vtype", None) and var_type is not None:
                # Old AST structure - check if this is actually an array declaration disguised as a variable
                vtype_str = str(stmt.vtype)
                if (
                    "ArrayAccessNode" in vtype_str
                    and "array=" in vtype_str
                    and "index=" in vtype_str
                ):
                    # This is likely an array declaration
                    array_match = re.search(r"array=(\w+).*?index=(\w+)", vtype_str)
                    if array_match:
                        array_match.group(1)
                        size = array_match.group(2)
                        base_type = "Float32"  # Default, could be improved
                        return (
                            f"{indent_str}var {stmt.name} = "
                            f"InlineArray[{base_type}, {size}]"
                            "(unsafe_uninitialized=True)\n"
                        )

            if hasattr(stmt, "initial_value") and stmt.initial_value is not None:
                self.register_variable_type(
                    stmt.name,
                    var_type or self.expression_result_type(stmt.initial_value),
                )
                if (
                    isinstance(stmt.initial_value, ArrayLiteralNode)
                    and var_type is not None
                    and self.is_array_type_name(var_type)
                ):
                    prelude = ""
                    init_expr = self.generate_array_literal_expression(
                        stmt.initial_value, var_type
                    )
                else:
                    increment_init = self.generate_increment_initializer_declaration(
                        stmt,
                        stmt.initial_value,
                        var_type,
                        indent,
                    )
                    if increment_init is not None:
                        return increment_init
                    prelude, init_expr = self.generate_expression_with_prelude(
                        stmt.initial_value,
                        indent,
                    )
                if var_type is None:
                    return f"{prelude}{indent_str}var {stmt.name} = {init_expr}\n"
                return (
                    f"{prelude}{indent_str}var {stmt.name}: "
                    f"{self.map_type(var_type)} = {init_expr}\n"
                )

            var_type = var_type or "float"
            self.register_variable_type(stmt.name, var_type)
            if self.is_array_type_name(var_type):
                return (
                    f"{indent_str}var {stmt.name} = "
                    f"{self.array_initial_value_for_type(var_type)}\n"
                )
            elif self.is_struct_type_name(var_type):
                return (
                    f"{indent_str}var {stmt.name} = "
                    f"{self.zero_value_for_type(var_type)}\n"
                )
            else:
                return f"{indent_str}var {stmt.name}: {self.map_type(var_type)}\n"
        elif isinstance(stmt, ArrayNode):
            return self.generate_array_declaration(stmt, indent)
        elif isinstance(stmt, AssignmentNode):
            return self.generate_assignment_statement(stmt, indent)
        elif isinstance(stmt, IfNode):
            return self.generate_if(stmt, indent)
        elif isinstance(stmt, ForNode):
            return self.generate_for(stmt, indent)
        elif isinstance(stmt, ForInNode):
            return self.generate_for_in(stmt, indent)
        elif isinstance(stmt, WhileNode):
            return self.generate_while(stmt, indent)
        elif isinstance(stmt, LoopNode):
            return self.generate_loop(stmt, indent)
        elif isinstance(stmt, DoWhileNode):
            return self.generate_do_while(stmt, indent)
        elif isinstance(stmt, MatchNode):
            return self.generate_match(stmt, indent)
        elif isinstance(stmt, SwitchNode):
            return self.generate_switch(stmt, indent)
        elif isinstance(stmt, ReturnNode):
            if isinstance(stmt.value, list):
                # Multiple return values
                values = ", ".join(self.generate_expression(val) for val in stmt.value)
                return f"{indent_str}return {values}\n"
            elif isinstance(stmt.value, ArrayLiteralNode) and self.is_array_type_name(
                self.current_return_type
            ):
                return_value = self.generate_array_literal_expression(
                    stmt.value, self.current_return_type
                )
                return f"{indent_str}return " f"{return_value}\n"
            else:
                prelude, return_value = self.generate_expression_with_prelude(
                    stmt.value,
                    indent,
                )
                return f"{prelude}{indent_str}return {return_value}\n"
        elif isinstance(stmt, BreakNode):
            context = self.active_do_while_context()
            if context:
                break_flag = context["break_flag"]
                return f"{indent_str}{break_flag} = True\n{indent_str}break\n"
            return f"{indent_str}break\n"
        elif isinstance(stmt, ContinueNode):
            if self.active_do_while_context():
                return f"{indent_str}break\n"
            context = self.active_for_context()
            if context:
                update = context["update"]
                return f"{indent_str}{update}\n{indent_str}continue\n"
            return f"{indent_str}continue\n"
        elif isinstance(stmt, ArrayAccessNode):
            # ArrayAccessNode should not appear as a statement by itself - it's likely a misclassified array declaration
            # Try to handle it gracefully
            return f"{indent_str}# Unhandled ArrayAccessNode: {stmt}\n"
        else:
            # Handle expressions that may be used as statements
            prelude, expr_result = self.generate_expression_with_prelude(stmt, indent)
            if expr_result.strip():
                return f"{prelude}{indent_str}{expr_result}\n"
            else:
                return f"{indent_str}# Unhandled statement: {type(stmt).__name__}\n"

    def generate_increment_initializer_declaration(
        self,
        stmt,
        initial_value,
        var_type,
        indent,
    ):
        if not isinstance(initial_value, UnaryOpNode):
            return None

        op = self.map_operator(
            getattr(initial_value, "operator", getattr(initial_value, "op", ""))
        )
        if op not in {"++", "--"}:
            return None

        operand = self.generate_expression(getattr(initial_value, "operand", ""))
        assignment_op = "+=" if op == "++" else "-="
        indent_str = "    " * indent
        update = f"{indent_str}{operand} {assignment_op} 1\n"
        if var_type is None:
            declaration = f"{indent_str}var {stmt.name} = {operand}\n"
        else:
            declaration = (
                f"{indent_str}var {stmt.name}: "
                f"{self.map_type(var_type)} = {operand}\n"
            )
        is_postfix = getattr(
            initial_value,
            "is_postfix",
            getattr(initial_value, "postfix", False),
        )
        if is_postfix:
            return declaration + update
        return update + declaration

    def generate_switch(self, node, indent):
        indent_str = "    " * indent
        expression = self.generate_expression(getattr(node, "expression", ""))
        code = ""
        emitted_condition = False
        default_body = None

        for case in getattr(node, "cases", []) or []:
            if not isinstance(case, CaseNode):
                continue
            value = getattr(case, "value", None)
            if value is None:
                default_body = getattr(case, "statements", [])
                continue

            keyword = "if" if not emitted_condition else "elif"
            condition = f"{expression} == {self.generate_expression(value)}"
            code += f"{indent_str}{keyword} {condition}:\n"
            code += self.generate_switch_case_body(
                getattr(case, "statements", []), indent + 1
            )
            emitted_condition = True

        explicit_default = getattr(node, "default_case", None)
        if explicit_default is not None:
            default_body = explicit_default

        if default_body is not None:
            code += f"{indent_str}else:\n"
            code += self.generate_switch_case_body(default_body, indent + 1)
        elif not emitted_condition:
            code += f"{indent_str}pass\n"

        return code

    def generate_switch_case_body(self, body, indent):
        statements = self.statement_list(body)
        code = ""
        for stmt in statements:
            if isinstance(stmt, BreakNode):
                continue
            code += self.generate_statement(stmt, indent)
        if not code:
            code = f"{'    ' * indent}pass\n"
        return code

    def generate_match(self, node, indent):
        indent_str = "    " * indent
        expression = self.generate_expression(getattr(node, "expression", ""))
        code = ""
        emitted_condition = False
        wildcard_body = None

        for arm in getattr(node, "arms", []) or []:
            if not self.is_supported_match_arm(arm):
                raise ValueError(
                    "Unsupported match arm for Mojo codegen; only unguarded "
                    "literal and wildcard patterns are supported"
                )

            pattern = getattr(arm, "pattern", None)
            body = getattr(arm, "body", [])
            if isinstance(pattern, WildcardPatternNode):
                wildcard_body = body
                continue

            keyword = "if" if not emitted_condition else "elif"
            condition = f"{expression} == {self.generate_expression(pattern.literal)}"
            code += f"{indent_str}{keyword} {condition}:\n"
            code += self.generate_switch_case_body(body, indent + 1)
            emitted_condition = True

        if wildcard_body is not None:
            code += f"{indent_str}else:\n"
            code += self.generate_switch_case_body(wildcard_body, indent + 1)
        elif not emitted_condition:
            code += f"{indent_str}pass\n"

        return code

    def is_supported_match_arm(self, arm):
        if getattr(arm, "guard", None) is not None:
            return False
        pattern = getattr(arm, "pattern", None)
        return isinstance(pattern, (LiteralPatternNode, WildcardPatternNode))

    def statement_list(self, body):
        if hasattr(body, "statements"):
            return body.statements
        if isinstance(body, list):
            return body
        if body is None:
            return []
        return [body]

    def active_do_while_context(self):
        if not self.do_while_contexts:
            return None
        context = self.do_while_contexts[-1]
        if context["loop_depth"] == self.loop_depth:
            return context
        return None

    def active_for_context(self):
        if not self.for_contexts:
            return None
        context = self.for_contexts[-1]
        if context["loop_depth"] == self.loop_depth:
            return context
        return None

    def statement_body_terminates_inner_loop(self, body):
        statements = self.statement_list(body)
        if not statements:
            return False
        return isinstance(statements[-1], (BreakNode, ContinueNode, ReturnNode))

    def generate_array_declaration(self, node, indent=0):
        indent_str = "    " * indent
        size = get_array_size_from_node(node)
        self.register_variable_type(
            node.name, self.array_type_name(node.element_type, size)
        )
        return (
            f"{indent_str}var {node.name} = "
            f"{self.array_initial_value(node.element_type, size)}\n"
        )

    def generate_assignment(self, node):
        left = self.generate_expression(node.left)
        left_type = self.expression_result_type(node.left)
        if isinstance(node.right, ArrayLiteralNode) and self.is_array_type_name(
            left_type
        ):
            right = self.generate_array_literal_expression(node.right, left_type)
        else:
            right = self.generate_expression(node.right)
        op = self.map_operator(node.operator)
        return f"{left} {op} {right}"

    def generate_if(self, node, indent):
        indent_str = "    " * indent
        condition = self.generate_expression(
            node.condition if hasattr(node, "condition") else node.if_condition
        )
        code = f"{indent_str}if {condition}:\n"

        if_body = getattr(node, "then_branch", getattr(node, "if_body", None))
        if hasattr(if_body, "statements"):
            for stmt in if_body.statements:
                code += self.generate_statement(stmt, indent + 1)
        elif isinstance(if_body, list):
            for stmt in if_body:
                code += self.generate_statement(stmt, indent + 1)

        else_branch = getattr(node, "else_branch", None)
        if else_branch:
            if hasattr(else_branch, "__class__") and "If" in str(else_branch.__class__):
                # Generate elif by recursively generating the nested if with elif prefix
                elif_condition = self.generate_expression(
                    else_branch.condition
                    if hasattr(else_branch, "condition")
                    else else_branch.if_condition
                )
                code += f"{indent_str}elif {elif_condition}:\n"

                # Generate elif body
                elif_body = getattr(
                    else_branch, "then_branch", getattr(else_branch, "if_body", None)
                )
                if hasattr(elif_body, "statements"):
                    for stmt in elif_body.statements:
                        code += self.generate_statement(stmt, indent + 1)
                elif isinstance(elif_body, list):
                    for stmt in elif_body:
                        code += self.generate_statement(stmt, indent + 1)

                nested_else = getattr(else_branch, "else_branch", None)
                if nested_else:
                    if hasattr(nested_else, "__class__") and "If" in str(
                        nested_else.__class__
                    ):
                        # Another elif
                        remaining_code = self.generate_if(nested_else, indent)
                        # Remove the "if" prefix and replace with "elif"
                        remaining_lines = remaining_code.split("\n")
                        if remaining_lines[0].strip().startswith("if "):
                            remaining_lines[0] = remaining_lines[0].replace(
                                "if ", "elif ", 1
                            )
                        code += "\n".join(remaining_lines)
                    else:
                        # Final else clause
                        code += f"{indent_str}else:\n"
                        if hasattr(nested_else, "statements"):
                            for stmt in nested_else.statements:
                                code += self.generate_statement(stmt, indent + 1)
                        elif isinstance(nested_else, list):
                            for stmt in nested_else:
                                code += self.generate_statement(stmt, indent + 1)
                        else:
                            code += self.generate_statement(nested_else, indent + 1)
            else:
                code += f"{indent_str}else:\n"
                if hasattr(else_branch, "statements"):
                    for stmt in else_branch.statements:
                        code += self.generate_statement(stmt, indent + 1)
                elif isinstance(else_branch, list):
                    for stmt in else_branch:
                        code += self.generate_statement(stmt, indent + 1)
                else:
                    code += self.generate_statement(else_branch, indent + 1)

        return code

    def generate_for(self, node, indent):
        indent_str = "    " * indent

        init = self.generate_statement(node.init, 0).strip()
        condition = self.generate_expression(node.condition)
        update = self.generate_expression(node.update)

        code = f"{indent_str}{init}\n"
        code += f"{indent_str}while {condition}:\n"

        self.loop_depth += 1
        self.for_contexts.append({"loop_depth": self.loop_depth, "update": update})
        try:
            for stmt in self.statement_list(node.body):
                code += self.generate_statement(stmt, indent + 1)
        finally:
            self.for_contexts.pop()
            self.loop_depth -= 1

        # Add update at the end of the loop
        code += f"{indent_str}    {update}\n"

        return code

    def generate_for_in(self, node, indent):
        indent_str = "    " * indent
        pattern = getattr(node, "pattern", "item")
        iterable = self.generate_for_in_iterable(getattr(node, "iterable", None))

        code = f"{indent_str}for {pattern} in {iterable}:\n"

        self.loop_depth += 1
        try:
            body_code = ""
            for stmt in self.statement_list(getattr(node, "body", [])):
                body_code += self.generate_statement(stmt, indent + 1)
        finally:
            self.loop_depth -= 1

        if body_code:
            code += body_code
        else:
            code += f"{indent_str}    pass\n"

        return code

    def generate_for_in_iterable(self, iterable_node):
        if isinstance(iterable_node, RangeNode):
            start = self.generate_expression(iterable_node.start)
            end = self.generate_expression(iterable_node.end)
            if iterable_node.inclusive:
                end = f"({end} + 1)"
            return f"range({start}, {end})"

        iterable = self.generate_expression(iterable_node)
        return f"range({iterable})"

    def generate_while(self, node, indent):
        indent_str = "    " * indent
        condition = self.generate_expression(node.condition)

        code = f"{indent_str}while {condition}:\n"

        self.loop_depth += 1
        try:
            for stmt in self.statement_list(node.body):
                code += self.generate_statement(stmt, indent + 1)
        finally:
            self.loop_depth -= 1

        return code

    def generate_loop(self, node, indent):
        indent_str = "    " * indent
        code = f"{indent_str}while True:\n"

        self.loop_depth += 1
        try:
            for stmt in self.statement_list(node.body):
                code += self.generate_statement(stmt, indent + 1)
        finally:
            self.loop_depth -= 1

        return code

    def generate_do_while(self, node, indent):
        indent_str = "    " * indent
        break_flag = f"__cgl_do_break_{self.do_while_counter}"
        self.do_while_counter += 1
        condition = self.generate_expression(node.condition)

        code = f"{indent_str}var {break_flag}: Bool = False\n"
        code += f"{indent_str}while True:\n"
        code += f"{indent_str}    while True:\n"

        self.loop_depth += 1
        self.do_while_contexts.append(
            {"loop_depth": self.loop_depth, "break_flag": break_flag}
        )
        try:
            for stmt in self.statement_list(node.body):
                code += self.generate_statement(stmt, indent + 2)
        finally:
            self.do_while_contexts.pop()
            self.loop_depth -= 1

        if not self.statement_body_terminates_inner_loop(node.body):
            code += f"{indent_str}        break\n"
        code += f"{indent_str}    if {break_flag}:\n"
        code += f"{indent_str}        break\n"
        code += f"{indent_str}    if not {condition}:\n"
        code += f"{indent_str}        break\n"

        return code

    def generate_lambda_expression(self, args):
        """Materialize supported CrossGL pseudo-lambdas as local Mojo functions."""
        if not args or not self.expression_prelude_stack:
            return None

        params = []
        param_types = {}
        for arg in args[:-1]:
            param = self.generate_lambda_parameter(arg)
            if param is None:
                return None
            type_name, param_name, mapped_type = param
            params.append((type_name, param_name, mapped_type))
            param_types[param_name] = type_name

        body_arg = args[-1]
        return_expr = self.lambda_return_expression(body_arg)
        return_type = self.infer_lambda_return_type(return_expr, param_types)
        if return_type is None:
            return None

        helper_name = self.next_lambda_helper_name()
        signature_params = ", ".join(
            f"{param_name}: {mapped_type}" for _, param_name, mapped_type in params
        )
        return_type_mapped = self.map_type(return_type)

        context = self.expression_prelude_stack[-1]
        indent = context["indent"]
        indent_str = "    " * indent
        body = self.generate_lambda_function_body(body_arg, indent + 1)
        if body is None:
            return None

        context["lines"].append(
            f"{indent_str}fn {helper_name}({signature_params}) -> "
            f"{return_type_mapped}:\n{body}"
        )
        return helper_name

    def generate_lambda_parameter(self, arg):
        raw = self.lambda_raw_argument_text(arg).strip()
        typed_param = self.split_lambda_typed_parameter(raw)
        if typed_param is None:
            return None

        type_name, param_name = typed_param
        mapped_type = self.lambda_parameter_type(type_name)
        if mapped_type is None:
            return None
        return type_name, param_name, mapped_type

    def generate_lambda_function_body(self, arg, indent):
        raw = self.lambda_raw_argument_text(arg).strip()
        if not raw:
            return None

        indent_str = "    " * indent
        if raw.startswith("{") and raw.endswith("}"):
            inner = raw[1:-1].strip()
            if not inner:
                return None
            lines = []
            for statement in inner.split(";"):
                statement = statement.strip()
                if not statement:
                    continue
                if statement.startswith("return "):
                    value = self.translate_lambda_raw_expression(
                        statement[len("return ") :].strip()
                    )
                    lines.append(f"{indent_str}return {value}\n")
                else:
                    lines.append(
                        f"{indent_str}{self.translate_lambda_raw_expression(statement)}\n"
                    )
            return "".join(lines) if lines else None

        expression = self.translate_lambda_raw_expression(raw)
        return f"{indent_str}return {expression}\n"

    def lambda_raw_argument_text(self, arg):
        if hasattr(arg, "name"):
            return arg.name
        if isinstance(arg, str):
            return arg
        return self.generate_expression(arg)

    def split_lambda_typed_parameter(self, raw):
        if not raw or ":" in raw:
            return None
        if any(char in raw for char in "{}()"):
            return None

        parts = raw.rsplit(None, 1)
        if len(parts) != 2:
            return None

        type_name, param_name = parts
        type_name = self.canonical_lambda_type(type_name)
        if not param_name.isidentifier():
            return None
        if not type_name:
            return None
        return type_name, param_name

    def canonical_lambda_type(self, type_name):
        if "<" in type_name or ">" in type_name:
            return "".join(type_name.split())
        return type_name

    def lambda_parameter_type(self, type_name):
        if any(char.isspace() for char in type_name):
            return None
        if any(char in type_name for char in "{},;[]()"):
            return None

        mapped_type = self.map_type(type_name)
        if "<" in type_name or ">" in type_name:
            if mapped_type == type_name:
                return None
        return mapped_type

    def lambda_return_expression(self, arg):
        raw = self.lambda_raw_argument_text(arg).strip()
        if not raw:
            return None
        if raw.startswith("{") and raw.endswith("}"):
            inner = raw[1:-1].strip()
            for statement in inner.split(";"):
                statement = statement.strip()
                if statement.startswith("return "):
                    return statement[len("return ") :].strip()
            return None
        return raw

    def infer_lambda_return_type(self, return_expr, param_types):
        if not return_expr:
            return None

        stripped = self.strip_wrapping_parentheses(return_expr.strip())
        literal_type = self.lambda_literal_type(stripped)
        if literal_type is not None:
            return literal_type

        if re.fullmatch(r"[A-Za-z_]\w*", stripped):
            return param_types.get(stripped) or self.variable_types.get(stripped)

        referenced_types = {
            type_name
            for name, type_name in param_types.items()
            if re.search(rf"\b{re.escape(name)}\b", stripped)
        }
        if len(referenced_types) == 1:
            return next(iter(referenced_types))
        return None

    def lambda_literal_type(self, value):
        if value in {"true", "false", "True", "False"}:
            return "bool"
        if re.fullmatch(r"[+-]?\d+", value):
            return "int"
        if re.fullmatch(r"[+-]?(\d+\.\d*|\.\d+)([eE][+-]?\d+)?", value):
            return "float"
        return None

    def strip_wrapping_parentheses(self, expression):
        while expression.startswith("(") and expression.endswith(")"):
            inner = expression[1:-1].strip()
            if not inner:
                break
            depth = 0
            wraps = True
            for index, char in enumerate(expression):
                if char == "(":
                    depth += 1
                elif char == ")":
                    depth -= 1
                    if depth == 0 and index != len(expression) - 1:
                        wraps = False
                        break
            if not wraps:
                break
            expression = inner
        return expression

    def translate_lambda_raw_expression(self, expression):
        translated = expression.strip()
        translated = re.sub(r"\btrue\b", "True", translated)
        translated = re.sub(r"\bfalse\b", "False", translated)
        translated = translated.replace("&&", " and ")
        translated = translated.replace("||", " or ")
        return translated

    def next_lambda_helper_name(self):
        while True:
            helper_name = f"_crossgl_lambda_{self.lambda_counter}"
            self.lambda_counter += 1
            if helper_name not in self.function_return_types:
                return helper_name

    def generate_expression(self, expr):
        """Render a CrossGL expression as Mojo expression syntax."""
        if isinstance(expr, str):
            return self.map_enum_variant_reference(expr)
        elif isinstance(expr, (int, float, bool)):
            return self.format_literal(expr)
        elif isinstance(expr, VariableNode):
            if hasattr(expr, "vtype") and expr.vtype and expr.name:
                return f"{expr.name}"
            elif hasattr(expr, "name"):
                return expr.name
            else:
                return str(expr)
        elif isinstance(expr, BinaryOpNode):
            vector_binary = self.generate_vector_binary_op(expr)
            if vector_binary is not None:
                return vector_binary
            left = self.generate_expression(expr.left)
            right = self.generate_expression(expr.right)
            op = self.map_operator(expr.op)
            return f"({left} {op} {right})"
        elif isinstance(expr, AssignmentNode):
            return self.generate_assignment(expr)
        elif isinstance(expr, ArrayLiteralNode):
            return self.generate_array_literal_expression(expr)
        elif isinstance(expr, UnaryOpNode):
            operand = self.generate_expression(expr.operand)
            op = self.map_operator(expr.op)
            if op in ["++", "--"]:
                assignment_op = "+=" if op == "++" else "-="
                return f"{operand} {assignment_op} 1"
            if op == "not":
                return f"(not {operand})"
            return f"({op}{operand})"
        elif isinstance(expr, ArrayAccessNode):
            # Handle array access properly
            if hasattr(expr, "array") and hasattr(expr, "index"):
                return self.generate_array_access_expression(expr)
            else:
                # Fallback for malformed ArrayAccessNode
                return str(expr)
        elif isinstance(expr, FunctionCallNode):
            # Extract function name properly (might be IdentifierNode)
            func_expr = getattr(expr, "function", None)
            if func_expr is None:
                func_expr = expr.name
            func_name = None
            if hasattr(func_expr, "name"):
                # It's an IdentifierNode, extract the name
                func_name = func_expr.name
                callee = func_name
            elif isinstance(func_expr, str):
                func_name = func_expr
                callee = func_expr
            else:
                callee = self.generate_expression(func_expr)

            if func_name == "lambda":
                lambda_expr = self.generate_lambda_expression(expr.args)
                if lambda_expr is not None:
                    return lambda_expr

            if self.is_user_defined_function(func_name):
                args = ", ".join(self.generate_expression(arg) for arg in expr.args)
                return f"{callee}({args})"

            if func_name in {"fract", "frac"}:
                return self.generate_fract_call(expr.args)
            if func_name == "mod":
                return self.generate_mod_call(expr.args)
            if func_name == "saturate":
                saturate_call = self.generate_saturate_call(expr.args)
                if saturate_call is not None:
                    return saturate_call
            if func_name == "mix":
                bool_mix_call = self.generate_bool_mix_call(expr.args)
                if bool_mix_call is not None:
                    return bool_mix_call
            if func_name == "texture":
                return self.generate_texture_call(expr.args, "sample")
            if func_name == "textureLod":
                return self.generate_texture_call(expr.args, "sample_lod")
            if func_name == "textureGrad":
                return self.generate_texture_call(expr.args, "sample_grad")
            if func_name == "textureSize":
                return self.generate_resource_size_call(expr.args, "texture_size")
            if func_name == "textureQueryLevels":
                return self.generate_resource_query_levels_call(expr.args)
            if func_name == "texelFetch":
                return self.generate_texel_fetch_call(expr.args)
            if func_name == "imageSize":
                return self.generate_resource_size_call(expr.args, "image_size")
            if func_name == "imageLoad":
                return self.generate_image_load_call(expr.args)
            if func_name == "imageStore":
                return self.generate_image_store_call(expr.args)

            # Map function names to Mojo equivalents
            func_name = self.function_map.get(func_name, func_name)
            if func_name in self.scalar_constructor_map:
                func_name = self.scalar_constructor_map[func_name]

            # Handle vector constructors
            if func_name in self.vector_constructor_info:
                return self.generate_vector_constructor(func_name, expr.args)

            if func_name in MOJO_MATRIX_TYPES:
                return self.generate_matrix_constructor(func_name, expr.args)

            # Handle standard function calls
            args = ", ".join(self.generate_expression(arg) for arg in expr.args)
            call_name = func_name if func_name is not None else callee
            return f"{call_name}({args})"
        elif isinstance(expr, MemberAccessNode):
            obj = self.generate_expression(expr.object)
            enum_variant = self.map_enum_variant_reference(f"{obj}.{expr.member}")
            if enum_variant != f"{obj}.{expr.member}":
                return enum_variant
            swizzle_indices = self.get_swizzle_indices(expr.member)
            if swizzle_indices is not None:
                obj_type = self.expression_result_type(expr.object)
                return self.generate_swizzle(
                    expr.object, obj, obj_type, expr.member, swizzle_indices
                )
            return f"{obj}.{expr.member}"
        elif isinstance(expr, TernaryOpNode):
            bool_vector_select = self.generate_bool_vector_select_expression(
                expr.condition, expr.true_expr, expr.false_expr
            )
            if bool_vector_select is not None:
                return bool_vector_select
            condition = self.generate_expression(expr.condition)
            true_expr = self.generate_expression(expr.true_expr)
            false_expr = self.generate_expression(expr.false_expr)
            return f"({true_expr} if {condition} else {false_expr})"
        elif hasattr(expr, "__class__") and "Literal" in str(expr.__class__):
            # Handle LiteralNode
            if hasattr(expr, "value"):
                literal_type = getattr(
                    getattr(expr, "literal_type", None), "name", None
                )
                return self.format_literal(expr.value, literal_type)
            return str(expr)
        elif hasattr(expr, "__class__") and "Identifier" in str(expr.__class__):
            # Handle IdentifierNode
            return self.map_enum_variant_reference(getattr(expr, "name", str(expr)))
        elif hasattr(expr, "__class__") and "ExpressionStatement" in str(
            expr.__class__
        ):
            # Handle ExpressionStatementNode
            if hasattr(expr, "expression"):
                return self.generate_expression(expr.expression)
            else:
                return self.generate_expression(expr)
        else:
            # For unknown expression types, handle special cases
            expr_str = str(expr)
            # Check if this looks like an array declaration being misinterpreted
            if (
                "ArrayAccessNode" in expr_str
                and "array=" in expr_str
                and "index=" in expr_str
            ):
                # Try to extract array name and size for array declarations
                import re

                array_match = re.search(r"array=(\w+).*?index=(\w+)", expr_str)
                if array_match:
                    array_name = array_match.group(1)
                    array_match.group(2)
                    return f"{array_name}"  # Just return the array name for now
            return expr_str

    def generate_vector_constructor(self, func_name, args):
        helper_call = self.generate_constructor_helper_call(func_name, args)
        if helper_call is not None:
            return helper_call

        dtype, source_width, storage_width, pad_literal = self.vector_constructor_info[
            func_name
        ]
        mojo_type = f"SIMD[{dtype}, {storage_width}]"
        emitted_args = []

        if len(args) == 1:
            arg = args[0]
            arg_components = self.vector_components_for_expression(arg, dtype)
            if arg_components is not None:
                emitted_args.extend(arg_components[:source_width])
            elif source_width == 3:
                arg_expr = self.generate_constructor_scalar_expression(arg, dtype)
                if self.is_duplicate_sensitive_expression(arg):
                    helper_name = self.vec3_splat_helper_name(dtype)
                    self.required_splat_helpers.add(dtype)
                    return f"{helper_name}({arg_expr})"
                emitted_args.extend([arg_expr] * source_width)
            else:
                emitted_args.append(
                    self.generate_constructor_scalar_expression(arg, dtype)
                )
        else:
            for arg in args:
                arg_components = self.vector_components_for_expression(arg, dtype)
                if arg_components is not None:
                    emitted_args.extend(arg_components)
                else:
                    emitted_args.append(
                        self.generate_constructor_scalar_expression(arg, dtype)
                    )

        if len(emitted_args) > source_width:
            emitted_args = emitted_args[:source_width]

        if source_width == 3 and len(emitted_args) == 3:
            emitted_args.append(pad_literal)

        return f"{mojo_type}({', '.join(emitted_args)})"

    def generate_matrix_constructor(self, func_name, args):
        dtype, columns, rows = MOJO_MATRIX_TYPES[func_name]
        matrix_key = (dtype, columns, rows)
        self.required_matrix_types.add(matrix_key)
        helper_call = self.generate_matrix_constructor_helper_call(
            dtype, columns, rows, args
        )
        if helper_call is not None:
            return helper_call

        component_count = columns * rows
        components = []
        for arg in args:
            arg_components = self.vector_components_for_expression(arg, dtype)
            if arg_components is not None:
                components.extend(arg_components)
            else:
                components.append(
                    self.generate_constructor_scalar_expression(arg, dtype)
                )

        if len(args) == 1 and len(components) == 1:
            scalar = components[0]
            components = [
                scalar if column == row else self.matrix_zero_literal(dtype)
                for column in range(columns)
                for row in range(rows)
            ]

        if len(components) > component_count:
            components = components[:component_count]
        elif len(components) < component_count:
            components.extend(
                self.matrix_zero_literal(dtype)
                for _ in range(component_count - len(components))
            )

        matrix_type = self.matrix_type_name(dtype, columns, rows)
        column_args = []
        storage_rows = self.matrix_storage_rows(rows)
        pad_literal = self.matrix_zero_literal(dtype)
        for column in range(columns):
            start = column * rows
            column_components = components[start : start + rows]
            if rows == 3:
                column_components.append(pad_literal)
            column_type = f"SIMD[{dtype}, {storage_rows}]"
            column_args.append(f"{column_type}({', '.join(column_components)})")

        return f"{matrix_type}({', '.join(column_args)})"

    def generate_fract_call(self, args):
        if not args:
            return "fract()"

        arg = args[0]
        arg_expr = self.generate_expression(arg)
        arg_type = self.expression_result_type(arg)
        vector_info = self.vector_type_info(arg_type)
        if vector_info is not None:
            dtype, source_width, storage_width, _ = vector_info
            if dtype in {"DType.float32", "DType.float64"}:
                self.required_fract_helpers.add(
                    ("vector", dtype, source_width, storage_width)
                )
                helper_name = self.fract_vector_helper_name(
                    dtype, source_width, storage_width
                )
                return f"{helper_name}({arg_expr})"

        dtype = self.expression_mojo_dtype(arg) or "DType.float32"
        if dtype not in {"DType.float32", "DType.float64"}:
            dtype = "DType.float32"
        self.required_fract_helpers.add(("scalar", dtype, 1, 1))
        return f"{self.fract_scalar_helper_name(dtype)}({arg_expr})"

    def generate_mod_call(self, args):
        generated_args = [self.generate_expression(arg) for arg in args]
        if len(generated_args) != 2:
            return f"fmod({', '.join(generated_args)})"

        left_type = self.expression_result_type(args[0])
        right_type = self.expression_result_type(args[1])
        if self.is_scalar_integer_type(left_type) and (
            right_type is None or self.is_scalar_integer_type(right_type)
        ):
            return f"({generated_args[0]} % {generated_args[1]})"

        return f"fmod({generated_args[0]}, {generated_args[1]})"

    def generate_saturate_call(self, args):
        if len(args) != 1:
            return None

        arg_type = self.expression_result_type(args[0])
        vector_info = self.vector_type_info(arg_type)
        if vector_info is not None:
            dtype, source_width, storage_width, _ = vector_info
            if dtype not in {"DType.float32", "DType.float64"}:
                return None
            arg_expr = self.generate_expression(args[0])
            self.required_saturate_helpers.add((dtype, source_width, storage_width))
            helper_name = self.saturate_vector_helper_name(
                dtype, source_width, storage_width
            )
            return f"{helper_name}({arg_expr})"

        arg_dtype = self.expression_mojo_dtype(args[0])
        if arg_dtype not in {"DType.float32", "DType.float64"}:
            return None

        arg_expr = self.generate_expression(args[0])
        return f"clamp({arg_expr}, 0.0, 1.0)"

    def generate_bool_mix_call(self, args):
        if len(args) != 3:
            return None

        factor_type = self.expression_result_type(args[2])
        factor_info = self.vector_type_info(factor_type)
        if factor_info is not None:
            if factor_info[0] != "DType.bool":
                return None
            return self.generate_bool_vector_select_expression(
                args[2], args[1], args[0]
            )

        if self.expression_mojo_dtype(args[2]) != "DType.bool":
            return None

        condition = self.generate_expression(args[2])
        true_value = self.generate_expression(args[1])
        false_value = self.generate_expression(args[0])
        return f"({true_value} if {condition} else {false_value})"

    def generate_texture_call(self, args, helper_name):
        if not args:
            return f"{helper_name}()"

        texture_type = self.expression_result_type(args[0])
        mapped_type = self.map_type(texture_type)
        if mapped_type in MOJO_RESOURCE_SAMPLE_COORDS:
            if helper_name == "sample":
                self.required_resource_sample_types.add(mapped_type)
            elif helper_name == "sample_lod":
                self.required_resource_lod_types.add(mapped_type)
            elif helper_name == "sample_grad":
                self.required_resource_grad_types.add(mapped_type)

        generated_args = ", ".join(self.generate_expression(arg) for arg in args)
        return f"{helper_name}({generated_args})"

    def generate_resource_size_call(self, args, helper_name):
        if not args:
            return f"{helper_name}()"

        resource_type = self.map_type(self.expression_result_type(args[0]))
        if resource_type in MOJO_RESOURCE_SIZE_RETURNS:
            self.required_resource_size_types.add(resource_type)

        generated_args = ", ".join(self.generate_expression(arg) for arg in args)
        return f"{helper_name}({generated_args})"

    def generate_resource_query_levels_call(self, args):
        if args:
            resource_type = self.map_type(self.expression_result_type(args[0]))
            if resource_type in MOJO_RESOURCE_SIZE_RETURNS:
                self.required_resource_query_level_types.add(resource_type)

        generated_args = ", ".join(self.generate_expression(arg) for arg in args)
        return f"texture_query_levels({generated_args})"

    def generate_texel_fetch_call(self, args):
        if args:
            resource_type = self.map_type(self.expression_result_type(args[0]))
            if resource_type in MOJO_RESOURCE_TEXEL_COORDS:
                self.required_resource_texel_fetch_types.add(resource_type)

        generated_args = ", ".join(self.generate_expression(arg) for arg in args)
        return f"texel_fetch({generated_args})"

    def generate_image_load_call(self, args):
        if args:
            resource_type = self.map_type(self.expression_result_type(args[0]))
            if resource_type in MOJO_RESOURCE_TEXEL_COORDS:
                self.required_image_load_types.add(resource_type)

        generated_args = ", ".join(self.generate_expression(arg) for arg in args)
        return f"image_load({generated_args})"

    def generate_image_store_call(self, args):
        if args:
            resource_type = self.map_type(self.expression_result_type(args[0]))
            if resource_type in MOJO_RESOURCE_TEXEL_COORDS:
                self.required_image_store_types.add(resource_type)

        generated_args = ", ".join(self.generate_expression(arg) for arg in args)
        return f"image_store({generated_args})"

    def generate_bool_vector_select_expression(
        self, condition_expr, true_expr, false_expr
    ):
        condition_info = self.vector_type_info(
            self.expression_result_type(condition_expr)
        )
        if condition_info is None or condition_info[0] != "DType.bool":
            return None

        true_info = self.vector_type_info(self.expression_result_type(true_expr))
        false_info = self.vector_type_info(self.expression_result_type(false_expr))
        if (
            true_info is None
            or false_info is None
            or true_info[:3] != false_info[:3]
            or condition_info[1] != true_info[1]
        ):
            return None

        dtype, source_width, storage_width, _ = true_info
        helper_name = self.select_helper_name(dtype, source_width, storage_width)
        self.required_select_helpers.add((dtype, source_width, storage_width))
        condition = self.generate_expression(condition_expr)
        true_value = self.generate_expression(true_expr)
        false_value = self.generate_expression(false_expr)
        return f"{helper_name}({condition}, {true_value}, {false_value})"

    def is_scalar_integer_type(self, type_name):
        if type_name is None or self.vector_type_info(type_name) is not None:
            return False
        return str(type_name) in {
            "int",
            "uint",
            "short",
            "ushort",
            "long",
            "ulong",
            "i32",
            "u32",
            "Int",
            "UInt",
            "Int32",
            "UInt32",
        }

    def generate_matrix_constructor_helper_call(self, dtype, columns, rows, args):
        component_count = columns * rows
        pieces = []

        for arg in args:
            piece = self.constructor_piece_for_expression(arg, dtype)
            if piece is None:
                return None
            pieces.append(piece)

        if len(args) == 1 and pieces and pieces[0]["kind"] == "scalar":
            return None

        pieces = self.select_constructor_pieces(pieces, component_count)
        if pieces is None:
            return None

        has_duplicate_sensitive_vector = any(
            piece["kind"] == "vector" and piece["duplicate_sensitive"]
            for piece in pieces
        )
        if not has_duplicate_sensitive_vector:
            return None

        key = self.matrix_constructor_helper_key(dtype, columns, rows, pieces)
        helper_name = self.matrix_constructor_helper_name(key)
        self.required_matrix_constructor_helpers[key] = {
            "key": key,
            "dtype": dtype,
            "columns": columns,
            "rows": rows,
            "pieces": pieces,
        }

        call_args = [self.generate_expression(piece["expr"]) for piece in pieces]
        return f"{helper_name}({', '.join(call_args)})"

    def matrix_constructor_helper_key(self, dtype, columns, rows, pieces):
        signature = self.constructor_helper_key(dtype, columns * rows, columns, pieces)
        return (dtype, columns, rows, signature[3])

    def matrix_constructor_helper_name(self, key):
        dtype, columns, rows, signature = key
        vector_key = (dtype, columns * rows, columns, signature)
        suffix = self.constructor_helper_name(vector_key).split("_", 4)[4]
        return (
            f"_crossgl_construct_matrix_{MOJO_DTYPE_SUFFIX[dtype]}_"
            f"c{columns}_r{rows}_{suffix}"
        )

    def matrix_type_name(self, dtype, columns, rows):
        dtype_suffix = MOJO_DTYPE_SUFFIX[dtype].upper()
        return f"CrossGLMatrix{dtype_suffix}C{columns}R{rows}"

    def matrix_storage_rows(self, rows):
        return 4 if rows == 3 else rows

    def matrix_zero_literal(self, dtype):
        return MOJO_DTYPE_INFO[dtype][2]

    def generate_matrix_type(self, key):
        dtype, columns, rows = key
        name = self.matrix_type_name(dtype, columns, rows)
        storage_rows = self.matrix_storage_rows(rows)
        column_type = f"SIMD[{dtype}, {storage_rows}]"
        code = f"@value\nstruct {name}:\n"
        for column in range(columns):
            code += f"    var c{column}: {column_type}\n"
        code += "\n"
        for index_type in ("Int", "Int32", "UInt32"):
            code += f"    fn __getitem__(self, index: {index_type}) -> {column_type}:\n"
            for column in range(columns - 1):
                code += f"        if index == {column}:\n"
                code += f"            return self.c{column}\n"
            code += f"        return self.c{columns - 1}\n\n"

            code += (
                f"    fn __setitem__(inout self, index: {index_type}, "
                f"value: {column_type}):\n"
            )
            for column in range(columns - 1):
                code += f"        if index == {column}:\n"
                code += f"            self.c{column} = value\n"
                code += "            return\n"
            code += f"        self.c{columns - 1} = value\n\n"
        return code + "\n"

    def generate_matrix_constructor_helper(self, helper):
        dtype = helper["dtype"]
        columns = helper["columns"]
        rows = helper["rows"]
        matrix_type = self.matrix_type_name(dtype, columns, rows)
        scalar_type, _, _ = MOJO_DTYPE_INFO[dtype]
        mojo_scalar_type = self.map_type(scalar_type)
        params = []
        components = []
        prelude = []

        for index, piece in enumerate(helper["pieces"]):
            if piece["kind"] == "vector":
                param_name = f"v{index}"
                vector_type = f"SIMD[{piece['dtype']}, {piece['storage_width']}]"
                params.append(f"{param_name}: {vector_type}")
                vector_expr = param_name
                if piece["dtype"] != dtype:
                    vector_expr = f"{param_name}_cast"
                    prelude.append(
                        f"    var {vector_expr} = {param_name}.cast[{dtype}]()\n"
                    )
                components.extend(
                    f"{vector_expr}[{component_index}]"
                    for component_index in piece["indices"]
                )
            else:
                param_name = f"s{index}"
                piece_dtype = piece.get("dtype")
                param_scalar_type = mojo_scalar_type
                if piece_dtype is not None and piece_dtype != dtype:
                    scalar_type = MOJO_DTYPE_INFO[piece_dtype][0]
                    param_scalar_type = self.map_type(scalar_type)
                params.append(f"{param_name}: {param_scalar_type}")
                components.append(
                    self.cast_scalar_text(param_name, piece_dtype, dtype)
                    if piece_dtype is not None
                    else param_name
                )

        column_args = []
        storage_rows = self.matrix_storage_rows(rows)
        pad_literal = self.matrix_zero_literal(dtype)
        for column in range(columns):
            start = column * rows
            column_components = components[start : start + rows]
            if rows == 3:
                column_components.append(pad_literal)
            column_type = f"SIMD[{dtype}, {storage_rows}]"
            column_args.append(f"{column_type}({', '.join(column_components)})")

        helper_name = self.matrix_constructor_helper_name(helper["key"])
        code = f"fn {helper_name}({', '.join(params)}) -> {matrix_type}:\n"
        code += "".join(prelude)
        code += f"    return {matrix_type}({', '.join(column_args)})\n\n"
        return code

    def generate_vector_binary_op(self, expr):
        op = self.map_operator(expr.op)
        if op not in MOJO_VECTOR_ARITHMETIC_OPS:
            return None

        left_type = self.expression_result_type(expr.left)
        right_type = self.expression_result_type(expr.right)
        left_info = self.vector_type_info(left_type)
        right_info = self.vector_type_info(right_type)
        left_is_vec3 = left_info is not None and left_info[1] == 3
        right_is_vec3 = right_info is not None and right_info[1] == 3

        if not left_is_vec3 and not right_is_vec3:
            return None

        if left_is_vec3 and right_is_vec3:
            if left_info[0] != right_info[0]:
                return None
            dtype = left_info[0]
            helper_kind = "vv"
        elif left_is_vec3:
            if right_info is not None:
                return None
            dtype = left_info[0]
            helper_kind = "vs"
        else:
            if left_info is not None:
                return None
            dtype = right_info[0]
            helper_kind = "sv"

        if dtype == "DType.bool" or dtype not in MOJO_DTYPE_SUFFIX:
            return None

        left = self.generate_expression(expr.left)
        right = self.generate_expression(expr.right)
        helper_name = self.vector_binary_helper_name(dtype, op, helper_kind)
        self.required_helpers.add((dtype, op, helper_kind))
        return f"{helper_name}({left}, {right})"

    def generate_required_helpers(self):
        if (
            not self.required_helpers
            and not self.required_splat_helpers
            and not self.required_swizzle_helpers
            and not self.required_constructor_helpers
            and not self.required_select_helpers
            and not self.required_matrix_types
            and not self.required_matrix_constructor_helpers
            and not self.required_fract_helpers
            and not self.required_saturate_helpers
            and not self.required_resource_types
            and not self.required_resource_sample_types
            and not self.required_resource_lod_types
            and not self.required_resource_grad_types
            and not self.required_resource_size_types
            and not self.required_resource_query_level_types
            and not self.required_resource_texel_fetch_types
            and not self.required_image_load_types
            and not self.required_image_store_types
        ):
            return ""

        code = ""
        resource_sampled_types = (
            self.required_resource_sample_types
            | self.required_resource_lod_types
            | self.required_resource_grad_types
            | self.required_resource_size_types
            | self.required_resource_query_level_types
            | self.required_resource_texel_fetch_types
            | self.required_image_load_types
            | self.required_image_store_types
        )
        if self.required_resource_types or resource_sampled_types:
            code += "# CrossGL resource placeholders\n"
            for resource_type in sorted(
                self.required_resource_types | resource_sampled_types
            ):
                code += self.generate_resource_type(resource_type)
            for resource_type in sorted(self.required_resource_sample_types):
                code += self.generate_resource_sample_helper(resource_type)
            for resource_type in sorted(self.required_resource_lod_types):
                code += self.generate_resource_lod_helper(resource_type)
            for resource_type in sorted(self.required_resource_grad_types):
                code += self.generate_resource_grad_helper(resource_type)
            for resource_type in sorted(self.required_resource_size_types):
                code += self.generate_resource_size_helper(resource_type)
            for resource_type in sorted(self.required_resource_query_level_types):
                code += self.generate_resource_query_levels_helper(resource_type)
            for resource_type in sorted(self.required_resource_texel_fetch_types):
                code += self.generate_texel_fetch_helper(resource_type)
            for resource_type in sorted(self.required_image_load_types):
                code += self.generate_image_load_helper(resource_type)
            for resource_type in sorted(self.required_image_store_types):
                code += self.generate_image_store_helper(resource_type)
            code += "\n"

        if self.required_fract_helpers or self.required_saturate_helpers:
            code += "# CrossGL math helpers\n"
            for key in sorted(self.required_fract_helpers):
                code += self.generate_fract_helper(key)
            for key in sorted(self.required_saturate_helpers):
                code += self.generate_saturate_helper(key)
            code += "\n"

        if self.required_matrix_types:
            code += "# CrossGL matrix types\n"
            for key in sorted(self.required_matrix_types):
                code += self.generate_matrix_type(key)
            code += "\n"

        if (
            self.required_helpers
            or self.required_splat_helpers
            or self.required_swizzle_helpers
            or self.required_constructor_helpers
            or self.required_select_helpers
            or self.required_matrix_constructor_helpers
        ):
            code += "# CrossGL vector helpers\n"
        for dtype, op, helper_kind in sorted(self.required_helpers):
            code += self.generate_vector_binary_helper(dtype, op, helper_kind)
        for dtype in sorted(self.required_splat_helpers):
            code += self.generate_vec3_splat_helper(dtype)
        for dtype, source_width, member in sorted(self.required_swizzle_helpers):
            code += self.generate_swizzle_helper(dtype, source_width, member)
        for key in sorted(self.required_constructor_helpers):
            code += self.generate_constructor_helper(
                self.required_constructor_helpers[key]
            )
        for key in sorted(self.required_select_helpers):
            code += self.generate_select_helper(key)
        for key in sorted(self.required_matrix_constructor_helpers):
            code += self.generate_matrix_constructor_helper(
                self.required_matrix_constructor_helpers[key]
            )
        return code + "\n"

    def generate_resource_type(self, resource_type):
        code = f"@value\nstruct {resource_type}:\n"
        code += "    pass\n\n"
        return code

    def generate_resource_sample_helper(self, resource_type):
        coord_type = MOJO_RESOURCE_SAMPLE_COORDS[resource_type]
        code = (
            f"fn sample(tex: {resource_type}, coord: {coord_type}) -> "
            "SIMD[DType.float32, 4]:\n"
        )
        code += "    return SIMD[DType.float32, 4](0.0, 0.0, 0.0, 1.0)\n\n"
        return code

    def generate_resource_lod_helper(self, resource_type):
        coord_type = MOJO_RESOURCE_SAMPLE_COORDS[resource_type]
        code = (
            f"fn sample_lod(tex: {resource_type}, coord: {coord_type}, "
            "lod: Float32) -> SIMD[DType.float32, 4]:\n"
        )
        code += "    return SIMD[DType.float32, 4](0.0, 0.0, lod, 1.0)\n\n"
        return code

    def generate_resource_grad_helper(self, resource_type):
        coord_type = MOJO_RESOURCE_SAMPLE_COORDS[resource_type]
        code = (
            f"fn sample_grad(tex: {resource_type}, coord: {coord_type}, "
            f"ddx: {coord_type}, ddy: {coord_type}) -> SIMD[DType.float32, 4]:\n"
        )
        code += "    return SIMD[DType.float32, 4](0.0, 0.0, 0.0, 1.0)\n\n"
        return code

    def generate_resource_size_helper(self, resource_type):
        return_type = MOJO_RESOURCE_SIZE_RETURNS[resource_type]
        zero_value = self.zero_mojo_value(return_type)
        code = f"fn texture_size(tex: {resource_type}) -> {return_type}:\n"
        code += f"    return {zero_value}\n\n"
        code += f"fn texture_size(tex: {resource_type}, lod: Int32) -> {return_type}:\n"
        code += f"    return {zero_value}\n\n"
        code += f"fn image_size(image: {resource_type}) -> {return_type}:\n"
        code += f"    return {zero_value}\n\n"
        return code

    def generate_resource_query_levels_helper(self, resource_type):
        code = f"fn texture_query_levels(tex: {resource_type}) -> Int32:\n"
        code += "    return 1\n\n"
        return code

    def generate_texel_fetch_helper(self, resource_type):
        coord_type = MOJO_RESOURCE_TEXEL_COORDS[resource_type]
        code = (
            f"fn texel_fetch(tex: {resource_type}, coord: {coord_type}, "
            "lod: Int32) -> SIMD[DType.float32, 4]:\n"
        )
        code += "    return SIMD[DType.float32, 4](0.0, 0.0, 0.0, 1.0)\n\n"
        return code

    def generate_image_load_helper(self, resource_type):
        coord_type = MOJO_RESOURCE_TEXEL_COORDS[resource_type]
        value_type = self.image_value_type(resource_type)
        if self.is_multisample_resource_type(resource_type):
            code = (
                f"fn image_load(image: {resource_type}, coord: {coord_type}, "
                f"sample: Int32) -> {value_type}:\n"
            )
        else:
            code = (
                f"fn image_load(image: {resource_type}, coord: {coord_type}) -> "
                f"{value_type}:\n"
            )
        code += f"    return {self.zero_mojo_value(value_type)}\n\n"
        return code

    def generate_image_store_helper(self, resource_type):
        coord_type = MOJO_RESOURCE_TEXEL_COORDS[resource_type]
        value_type = self.image_value_type(resource_type)
        if self.is_multisample_resource_type(resource_type):
            code = (
                f"fn image_store(image: {resource_type}, coord: {coord_type}, "
                f"sample: Int32, value: {value_type}):\n"
            )
        else:
            code = (
                f"fn image_store(image: {resource_type}, coord: {coord_type}, "
                f"value: {value_type}):\n"
            )
        code += "    pass\n\n"
        return code

    def is_multisample_resource_type(self, resource_type):
        return "MS" in resource_type

    def image_value_type(self, resource_type):
        if resource_type.startswith("IImage"):
            return "Int32"
        if resource_type.startswith("UImage"):
            return "UInt32"
        return "SIMD[DType.float32, 4]"

    def zero_mojo_value(self, mojo_type):
        if mojo_type in {
            "Int",
            "Int16",
            "Int32",
            "Int64",
            "UInt16",
            "UInt32",
            "UInt64",
        }:
            return "0"
        if mojo_type in {"Float16", "Float32", "Float64"}:
            return "0.0"
        if mojo_type == "Bool":
            return "False"

        vector_match = re.fullmatch(r"SIMD\[(DType\.\w+), (\d+)\]", mojo_type)
        if vector_match:
            dtype = vector_match.group(1)
            width = int(vector_match.group(2))
            zero = MOJO_DTYPE_INFO.get(dtype, MOJO_DTYPE_INFO["DType.float32"])[2]
            return f"{mojo_type}({', '.join([zero] * width)})"

        return f"{mojo_type}()"

    def generate_fract_helper(self, key):
        kind, dtype, source_width, storage_width = key
        scalar_type, _, pad_literal = MOJO_DTYPE_INFO[dtype]
        mojo_scalar_type = self.map_type(scalar_type)

        if kind == "scalar":
            helper_name = self.fract_scalar_helper_name(dtype)
            code = f"fn {helper_name}(x: {mojo_scalar_type}) -> {mojo_scalar_type}:\n"
            code += "    return x - floor(x)\n\n"
            return code

        helper_name = self.fract_vector_helper_name(dtype, source_width, storage_width)
        vector_type = f"SIMD[{dtype}, {storage_width}]"
        components = [
            f"v[{index}] - floor(v[{index}])" for index in range(source_width)
        ]
        if storage_width > source_width:
            components.append(pad_literal)

        code = f"fn {helper_name}(v: {vector_type}) -> {vector_type}:\n"
        code += f"    return {vector_type}({', '.join(components)})\n\n"
        return code

    def fract_scalar_helper_name(self, dtype):
        return f"_crossgl_fract_{MOJO_DTYPE_SUFFIX[dtype]}"

    def fract_vector_helper_name(self, dtype, source_width, storage_width):
        dtype_suffix = MOJO_DTYPE_SUFFIX[dtype]
        return f"_crossgl_fract_{dtype_suffix}_{source_width}_{storage_width}"

    def generate_saturate_helper(self, key):
        dtype, source_width, storage_width = key
        _, _, pad_literal = MOJO_DTYPE_INFO[dtype]
        vector_type = f"SIMD[{dtype}, {storage_width}]"
        components = [f"clamp(v[{index}], 0.0, 1.0)" for index in range(source_width)]
        if storage_width > source_width:
            components.append(pad_literal)

        helper_name = self.saturate_vector_helper_name(
            dtype, source_width, storage_width
        )
        code = f"fn {helper_name}(v: {vector_type}) -> {vector_type}:\n"
        code += f"    return {vector_type}({', '.join(components)})\n\n"
        return code

    def saturate_vector_helper_name(self, dtype, source_width, storage_width):
        dtype_suffix = MOJO_DTYPE_SUFFIX[dtype]
        return f"_crossgl_saturate_{dtype_suffix}_{source_width}_{storage_width}"

    def generate_vector_binary_helper(self, dtype, op, helper_kind):
        scalar_type, _, pad_literal = MOJO_DTYPE_INFO[dtype]
        mojo_scalar_type = self.map_type(scalar_type)
        vector_type = f"SIMD[{dtype}, 4]"
        helper_name = self.vector_binary_helper_name(dtype, op, helper_kind)

        if helper_kind == "vv":
            params = f"a: {vector_type}, b: {vector_type}"
            components = [f"a[{index}] {op} b[{index}]" for index in range(3)]
        elif helper_kind == "vs":
            params = f"v: {vector_type}, s: {mojo_scalar_type}"
            components = [f"v[{index}] {op} s" for index in range(3)]
        else:
            params = f"s: {mojo_scalar_type}, v: {vector_type}"
            components = [f"s {op} v[{index}]" for index in range(3)]

        components.append(pad_literal)
        args = ", ".join(components)
        code = f"fn {helper_name}({params}) -> {vector_type}:\n"
        code += f"    return {vector_type}({args})\n\n"
        return code

    def vector_binary_helper_name(self, dtype, op, helper_kind):
        op_name = MOJO_VECTOR_ARITHMETIC_OPS[op]
        dtype_suffix = MOJO_DTYPE_SUFFIX[dtype]
        return f"_crossgl_vec3_{op_name}_{dtype_suffix}_{helper_kind}"

    def generate_select_helper(self, key):
        dtype, source_width, storage_width = key
        _, _, pad_literal = MOJO_DTYPE_INFO[dtype]
        vector_type = f"SIMD[{dtype}, {storage_width}]"
        mask_type = f"SIMD[DType.bool, {storage_width}]"
        components = [
            f"true_value[{index}] if mask[{index}] else false_value[{index}]"
            for index in range(source_width)
        ]
        if storage_width > source_width:
            components.append(pad_literal)

        helper_name = self.select_helper_name(dtype, source_width, storage_width)
        code = (
            f"fn {helper_name}(mask: {mask_type}, true_value: {vector_type}, "
            f"false_value: {vector_type}) -> {vector_type}:\n"
        )
        code += f"    return {vector_type}({', '.join(components)})\n\n"
        return code

    def select_helper_name(self, dtype, source_width, storage_width):
        dtype_suffix = MOJO_DTYPE_SUFFIX[dtype]
        return f"_crossgl_select_{dtype_suffix}_{source_width}_{storage_width}"

    def generate_vec3_splat_helper(self, dtype):
        scalar_type, _, pad_literal = MOJO_DTYPE_INFO[dtype]
        mojo_scalar_type = self.map_type(scalar_type)
        vector_type = f"SIMD[{dtype}, 4]"
        helper_name = self.vec3_splat_helper_name(dtype)
        code = f"fn {helper_name}(s: {mojo_scalar_type}) -> {vector_type}:\n"
        code += f"    return {vector_type}(s, s, s, {pad_literal})\n\n"
        return code

    def vec3_splat_helper_name(self, dtype):
        return f"_crossgl_vec3_splat_{MOJO_DTYPE_SUFFIX[dtype]}"

    def generate_swizzle_helper(self, dtype, source_width, member):
        _, _, pad_literal = MOJO_DTYPE_INFO[dtype]
        swizzle_indices = self.get_swizzle_indices(member)
        result_width = 2 if len(swizzle_indices) == 2 else 4
        source_type = f"SIMD[{dtype}, {source_width}]"
        result_type = f"SIMD[{dtype}, {result_width}]"
        helper_name = self.swizzle_helper_name(dtype, source_width, member)
        components = [f"v[{index}]" for index in swizzle_indices]
        if len(swizzle_indices) == 3:
            components.append(pad_literal)

        code = f"fn {helper_name}(v: {source_type}) -> {result_type}:\n"
        code += f"    return {result_type}({', '.join(components)})\n\n"
        return code

    def swizzle_helper_name(self, dtype, source_width, member):
        dtype_suffix = MOJO_DTYPE_SUFFIX[dtype]
        return f"_crossgl_swizzle_{dtype_suffix}_{source_width}_{member}"

    def generate_constructor_helper_call(self, func_name, args):
        dtype, source_width, storage_width, pad_literal = self.vector_constructor_info[
            func_name
        ]
        pieces = []

        for arg in args:
            piece = self.constructor_piece_for_expression(arg, dtype)
            if piece is None:
                return None
            pieces.append(piece)

        pieces = self.select_constructor_pieces(pieces, source_width)
        if pieces is None:
            return None

        has_duplicate_sensitive_vector = any(
            piece["kind"] == "vector" and piece["duplicate_sensitive"]
            for piece in pieces
        )

        if not has_duplicate_sensitive_vector:
            return None

        key = self.constructor_helper_key(dtype, source_width, storage_width, pieces)
        helper_name = self.constructor_helper_name(key)
        self.required_constructor_helpers[key] = {
            "key": key,
            "dtype": dtype,
            "storage_width": storage_width,
            "pad_literal": pad_literal,
            "pieces": pieces,
        }

        call_args = [self.generate_expression(piece["expr"]) for piece in pieces]
        return f"{helper_name}({', '.join(call_args)})"

    def select_constructor_pieces(self, pieces, source_width):
        selected = []
        remaining = source_width

        for piece in pieces:
            if remaining == 0:
                break
            if piece["kind"] == "vector":
                indices = piece["indices"][:remaining]
                if indices:
                    selected.append({**piece, "indices": tuple(indices)})
                    remaining -= len(indices)
            else:
                selected.append(piece)
                remaining -= 1

        if remaining != 0:
            return None
        return selected

    def constructor_piece_for_expression(self, expr, target_dtype):
        if isinstance(expr, MemberAccessNode):
            swizzle_indices = self.get_swizzle_indices(expr.member)
            if swizzle_indices is not None:
                source_type = self.expression_result_type(expr.object)
                source_info = self.vector_type_info(source_type)
                if source_info is None:
                    return None
                return {
                    "kind": "vector",
                    "dtype": source_info[0],
                    "storage_width": source_info[2],
                    "indices": tuple(swizzle_indices),
                    "expr": expr.object,
                    "duplicate_sensitive": self.is_duplicate_sensitive_expression(
                        expr.object
                    ),
                }

        expr_type = self.expression_result_type(expr)
        info = self.vector_type_info(expr_type)
        if info is not None:
            _, source_width, storage_width, _ = info
            return {
                "kind": "vector",
                "dtype": info[0],
                "storage_width": storage_width,
                "indices": tuple(range(source_width)),
                "expr": expr,
                "duplicate_sensitive": self.is_duplicate_sensitive_expression(expr),
            }

        return {
            "kind": "scalar",
            "expr": expr,
            "dtype": self.expression_mojo_dtype(expr),
        }

    def constructor_helper_key(self, dtype, source_width, storage_width, pieces):
        signature = []
        for piece in pieces:
            if piece["kind"] == "vector":
                signature.append(
                    (
                        "v",
                        piece["dtype"],
                        piece["storage_width"],
                        piece["indices"],
                    )
                )
            else:
                piece_dtype = piece.get("dtype")
                if piece_dtype is not None and piece_dtype != dtype:
                    signature.append(("s", piece_dtype))
                else:
                    signature.append(("s",))
        return (dtype, source_width, storage_width, tuple(signature))

    def constructor_helper_name(self, key):
        dtype, _, storage_width, signature = key
        parts = []
        for piece in signature:
            if piece[0] == "v":
                _, piece_dtype, piece_storage_width, indices = piece
                index_text = "".join(str(index) for index in indices)
                parts.append(
                    f"v{MOJO_DTYPE_SUFFIX[piece_dtype]}{piece_storage_width}_{index_text}"
                )
            elif len(piece) > 1:
                parts.append(f"s{MOJO_DTYPE_SUFFIX[piece[1]]}")
            else:
                parts.append("s")
        suffix = "_".join(parts)
        return f"_crossgl_construct_{MOJO_DTYPE_SUFFIX[dtype]}_{storage_width}_{suffix}"

    def generate_constructor_helper(self, helper):
        dtype = helper["dtype"]
        scalar_type, _, _ = MOJO_DTYPE_INFO[dtype]
        mojo_scalar_type = self.map_type(scalar_type)
        result_type = f"SIMD[{dtype}, {helper['storage_width']}]"
        params = []
        components = []
        prelude = []

        for index, piece in enumerate(helper["pieces"]):
            if piece["kind"] == "vector":
                param_name = f"v{index}"
                vector_type = f"SIMD[{piece['dtype']}, {piece['storage_width']}]"
                params.append(f"{param_name}: {vector_type}")
                vector_expr = param_name
                if piece["dtype"] != dtype:
                    vector_expr = f"{param_name}_cast"
                    prelude.append(
                        f"    var {vector_expr} = {param_name}.cast[{dtype}]()\n"
                    )
                components.extend(
                    f"{vector_expr}[{component_index}]"
                    for component_index in piece["indices"]
                )
            else:
                param_name = f"s{index}"
                piece_dtype = piece.get("dtype")
                param_scalar_type = mojo_scalar_type
                if piece_dtype is not None and piece_dtype != dtype:
                    scalar_type = MOJO_DTYPE_INFO[piece_dtype][0]
                    param_scalar_type = self.map_type(scalar_type)
                params.append(f"{param_name}: {param_scalar_type}")
                components.append(
                    self.cast_scalar_text(param_name, piece_dtype, dtype)
                    if piece_dtype is not None
                    else param_name
                )

        if helper["pad_literal"] is not None and len(components) == 3:
            components.append(helper["pad_literal"])

        helper_name = self.constructor_helper_name(helper["key"])
        code = f"fn {helper_name}({', '.join(params)}) -> {result_type}:\n"
        code += "".join(prelude)
        code += f"    return {result_type}({', '.join(components)})\n\n"
        return code

    def register_variable_type(self, name, var_type):
        if name and var_type:
            self.variable_types[name] = self.type_name(var_type)

    def type_name(self, type_value):
        if hasattr(type_value, "name") or hasattr(type_value, "element_type"):
            return self.convert_type_node_to_string(type_value)
        return str(type_value)

    def is_array_type_name(self, type_name):
        return type_name is not None and "[" in str(type_name) and "]" in str(type_name)

    def is_struct_type_name(self, type_name):
        if type_name is None:
            return False
        return self.type_name(type_name) in self.struct_types

    def array_type_name(self, element_type, size):
        element_type_name = self.type_name(element_type)
        if size is None:
            return f"{element_type_name}[]"
        return f"{element_type_name}[{size}]"

    def array_storage_type(self, element_type, size):
        element_type_name = self.map_type(element_type)
        if size is None:
            return f"List[{element_type_name}]"
        return f"InlineArray[{element_type_name}, {size}]"

    def array_initial_value(self, element_type, size):
        array_type = self.array_storage_type(element_type, size)
        if size is None:
            return f"{array_type}()"
        return f"{array_type}(unsafe_uninitialized=True)"

    def array_initial_value_for_type(self, type_name):
        element_type, size = parse_array_type(str(type_name))
        return self.array_initial_value(element_type, size)

    def array_element_type(self, type_name):
        if not self.is_array_type_name(type_name):
            return None
        element_type, _ = parse_array_type(str(type_name))
        return element_type

    def generate_array_literal_expression(self, expr, target_type=None):
        if target_type is not None and self.is_array_type_name(target_type):
            element_type, size = parse_array_type(str(target_type))
        else:
            element_type = self.infer_array_literal_element_type(expr)
            size = len(expr.elements)

        array_type = self.array_storage_type(element_type, size)
        elements = [
            self.generate_array_literal_element(element, element_type)
            for element in expr.elements
        ]

        if size is not None:
            size = int(size)
            elements = elements[:size]
            while len(elements) < size:
                elements.append(self.zero_value_for_type(element_type))

        return f"{array_type}({', '.join(elements)})"

    def infer_array_literal_element_type(self, expr):
        if not expr.elements:
            return "float"
        return self.expression_result_type(expr.elements[0]) or "float"

    def generate_array_literal_element(self, element, element_type):
        target_dtype = MOJO_SCALAR_DTYPES.get(self.type_name(element_type))
        if target_dtype is not None:
            return self.generate_constructor_scalar_expression(element, target_dtype)
        return self.generate_expression(element)

    def zero_value_for_type(self, type_name):
        type_name = self.type_name(type_name)
        if self.is_array_type_name(type_name):
            element_type, size = parse_array_type(type_name)
            return self.zero_array_value(element_type, size)

        if type_name in self.struct_types:
            return self.zero_struct_value(type_name)

        vector_info = self.vector_type_info(type_name)
        if vector_info is not None:
            dtype, source_width, storage_width, pad_literal = vector_info
            zero = MOJO_DTYPE_INFO[dtype][2]
            components = [zero] * source_width
            if pad_literal is not None and len(components) == 3:
                components.append(pad_literal)
            return f"SIMD[{dtype}, {storage_width}]({', '.join(components)})"

        matrix_info = self.matrix_type_info(type_name)
        if matrix_info is not None:
            dtype, columns, rows = matrix_info
            return self.zero_matrix_value(dtype, columns, rows)

        dtype = MOJO_SCALAR_DTYPES.get(type_name)
        if dtype is not None:
            return MOJO_DTYPE_INFO[dtype][2]
        return f"{self.map_type(type_name)}()"

    def zero_array_value(self, element_type, size):
        array_type = self.array_storage_type(element_type, size)
        if size is None:
            return f"{array_type}()"

        try:
            element_count = int(size)
        except (TypeError, ValueError):
            return f"{array_type}(unsafe_uninitialized=True)"

        values = [self.zero_value_for_type(element_type) for _ in range(element_count)]
        return f"{array_type}({', '.join(values)})"

    def zero_struct_value(self, type_name):
        fields = self.struct_types.get(type_name, {})
        values = [
            self.zero_value_for_type(field_type) for field_type in fields.values()
        ]
        return f"{type_name}({', '.join(values)})"

    def zero_matrix_value(self, dtype, columns, rows):
        self.required_matrix_types.add((dtype, columns, rows))
        matrix_type = self.matrix_type_name(dtype, columns, rows)
        storage_rows = self.matrix_storage_rows(rows)
        zero = self.matrix_zero_literal(dtype)
        column_type = f"SIMD[{dtype}, {storage_rows}]"
        column_values = []
        for _ in range(columns):
            components = [zero] * rows
            if rows == 3:
                components.append(zero)
            column_values.append(f"{column_type}({', '.join(components)})")
        return f"{matrix_type}({', '.join(column_values)})"

    def generate_array_access_expression(self, expr):
        array_type = self.expression_result_type(expr.array)
        matrix_info = self.matrix_type_info(array_type)
        vector_info = self.vector_type_info(array_type)
        array_element_type = self.array_element_type(array_type)
        array = self.generate_expression(expr.array)
        index = self.generate_array_index_expression(
            expr.index,
            cast_integer_index=vector_info is not None
            or array_element_type is not None,
        )

        if matrix_info is not None:
            column_index = self.literal_int_value(expr.index)
            if column_index is not None:
                return f"{array}.c{column_index}"

        return f"{array}[{index}]"

    def generate_array_index_expression(self, expr, cast_integer_index=False):
        index = self.generate_expression(expr)
        if not cast_integer_index or self.literal_int_value(expr) is not None:
            return index

        index_type = self.expression_result_type(expr)
        if index_type in MOJO_INTEGER_INDEX_TYPES:
            return f"int({index})"
        return index

    def literal_int_value(self, expr):
        if hasattr(expr, "value"):
            try:
                return int(expr.value)
            except (TypeError, ValueError):
                return None
        if isinstance(expr, str):
            try:
                return int(expr)
            except ValueError:
                return None
        return None

    def expression_result_type(self, expr):
        if isinstance(expr, str):
            return self.variable_types.get(expr)
        if isinstance(expr, VariableNode) and hasattr(expr, "name"):
            return self.variable_types.get(expr.name)
        if isinstance(expr, ArrayLiteralNode):
            element_type = self.infer_array_literal_element_type(expr)
            return self.array_type_name(element_type, len(expr.elements))
        if isinstance(expr, ArrayAccessNode):
            array_type = self.expression_result_type(expr.array)
            array_element_type = self.array_element_type(array_type)
            if array_element_type is not None:
                return array_element_type
            matrix_info = self.matrix_type_info(array_type)
            if matrix_info is not None:
                dtype, _, rows = matrix_info
                return self.vector_type_name_for_dtype_width(dtype, rows)
            vector_info = self.vector_type_info(array_type)
            if vector_info is not None:
                return MOJO_DTYPE_INFO[vector_info[0]][0]
            return None
        if isinstance(expr, BinaryOpNode):
            left_type = self.expression_result_type(expr.left)
            right_type = self.expression_result_type(expr.right)
            left_info = self.vector_type_info(left_type)
            right_info = self.vector_type_info(right_type)
            if left_info is not None and right_info is not None:
                return left_type if left_info == right_info else left_type
            if left_info is not None:
                return left_type
            if right_info is not None:
                return right_type
            return left_type if left_type == right_type else left_type or right_type
        if isinstance(expr, FunctionCallNode):
            func_name = self.function_call_name(expr)
            if func_name in self.vector_constructor_info:
                return func_name
            if func_name in MOJO_MATRIX_TYPES:
                return func_name
            if func_name in {"fract", "frac"} and expr.args:
                return self.expression_result_type(expr.args[0]) or "float"
            if func_name == "saturate" and expr.args:
                return self.expression_result_type(expr.args[0]) or "float"
            if func_name in {"texture", "textureLod", "textureGrad", "texelFetch"}:
                return "vec4"
            if func_name == "textureQueryLevels":
                return "int"
            if func_name in {"textureSize", "imageSize"} and expr.args:
                resource_type = self.map_type(self.expression_result_type(expr.args[0]))
                size_type = MOJO_RESOURCE_SIZE_RETURNS.get(resource_type)
                if size_type == "Int32":
                    return "int"
                if size_type == "SIMD[DType.int32, 2]":
                    return "ivec2"
                if size_type == "SIMD[DType.int32, 4]":
                    return "ivec3"
            if func_name == "imageLoad" and expr.args:
                resource_type = self.map_type(self.expression_result_type(expr.args[0]))
                value_type = self.image_value_type(resource_type)
                if value_type == "Int32":
                    return "int"
                if value_type == "UInt32":
                    return "uint"
                return "vec4"
            if func_name == "imageStore":
                return "void"
            return self.function_return_types.get(func_name)
        if isinstance(expr, MemberAccessNode):
            swizzle_indices = self.get_swizzle_indices(expr.member)
            if swizzle_indices is not None:
                obj_type = self.expression_result_type(expr.object)
                return self.swizzle_result_type(obj_type, len(swizzle_indices))

            obj_type = self.expression_result_type(expr.object)
            if obj_type in self.struct_types:
                return self.struct_types[obj_type].get(expr.member)
        if hasattr(expr, "__class__") and "Identifier" in str(expr.__class__):
            return self.variable_types.get(getattr(expr, "name", ""))
        if hasattr(expr, "__class__") and "Literal" in str(expr.__class__):
            literal_type = getattr(getattr(expr, "literal_type", None), "name", None)
            if literal_type:
                return literal_type
        return None

    def function_call_name(self, expr):
        func_expr = getattr(expr, "function", None)
        if func_expr is None:
            func_expr = expr.name
        if hasattr(func_expr, "name"):
            return func_expr.name
        if isinstance(func_expr, str):
            return func_expr
        return None

    def vector_type_info(self, type_name):
        if type_name in self.vector_constructor_info:
            return self.vector_constructor_info[type_name]
        return None

    def matrix_type_info(self, type_name):
        if type_name in MOJO_MATRIX_TYPES:
            return MOJO_MATRIX_TYPES[type_name]
        return None

    def vector_type_name_for_dtype_width(self, dtype, width):
        _, prefix, _ = MOJO_DTYPE_INFO[dtype]
        return f"{prefix}{width}"

    def swizzle_result_type(self, obj_type, component_count):
        info = self.vector_type_info(obj_type)
        dtype = info[0] if info else "DType.float32"
        scalar_type, prefix, _ = MOJO_DTYPE_INFO.get(
            dtype, MOJO_DTYPE_INFO["DType.float32"]
        )
        if component_count == 1:
            return scalar_type
        return f"{prefix}{component_count}"

    def get_swizzle_indices(self, member):
        if not member:
            return None
        for components in SWIZZLE_SETS.values():
            if all(component in components for component in member):
                return [components[component] for component in member]
        return None

    def expression_mojo_dtype(self, expr):
        expr_type = self.expression_result_type(expr)
        info = self.vector_type_info(expr_type)
        if info is not None:
            return info[0]
        return MOJO_SCALAR_DTYPES.get(expr_type)

    def is_literal_expression(self, expr):
        return hasattr(expr, "__class__") and "Literal" in str(expr.__class__)

    def cast_scalar_text(self, expr_text, source_dtype, target_dtype):
        if target_dtype is None or source_dtype is None or source_dtype == target_dtype:
            return expr_text
        return f"({expr_text}).cast[{target_dtype}]()"

    def cast_vector_component(self, component, source_dtype, target_dtype):
        if target_dtype is None or source_dtype is None or source_dtype == target_dtype:
            return component
        return f"{component}.cast[{target_dtype}]()"

    def generate_constructor_scalar_expression(self, expr, target_dtype):
        expr_text = self.generate_expression(expr)
        if self.is_literal_expression(expr):
            return expr_text
        return self.cast_scalar_text(
            expr_text, self.expression_mojo_dtype(expr), target_dtype
        )

    def vector_components_for_expression(self, expr, target_dtype=None):
        if isinstance(expr, MemberAccessNode):
            obj = self.generate_expression(expr.object)
            swizzle_indices = self.get_swizzle_indices(expr.member)
            if swizzle_indices is not None:
                source_info = self.vector_type_info(
                    self.expression_result_type(expr.object)
                )
                source_dtype = source_info[0] if source_info is not None else None
                return [
                    self.cast_vector_component(
                        f"{obj}[{index}]", source_dtype, target_dtype
                    )
                    for index in swizzle_indices
                ]

        expr_type = self.expression_result_type(expr)
        info = self.vector_type_info(expr_type)
        if info is None:
            return None

        source_dtype, source_width, _, _ = info
        if source_width <= 1:
            return None

        expr_text = self.generate_expression(expr)
        return [
            self.cast_vector_component(
                f"{expr_text}[{index}]", source_dtype, target_dtype
            )
            for index in range(source_width)
        ]

    def generate_swizzle(self, source_expr, obj, obj_type, member, swizzle_indices):
        if len(swizzle_indices) == 1:
            return f"{obj}[{swizzle_indices[0]}]"

        info = self.vector_type_info(obj_type)
        dtype = info[0] if info else "DType.float32"
        source_width = info[2] if info else 4
        if info is not None and self.is_duplicate_sensitive_expression(source_expr):
            helper_name = self.swizzle_helper_name(dtype, source_width, member)
            self.required_swizzle_helpers.add((dtype, source_width, member))
            return f"{helper_name}({obj})"

        _, _, pad_literal = MOJO_DTYPE_INFO.get(dtype, MOJO_DTYPE_INFO["DType.float32"])
        storage_width = 2 if len(swizzle_indices) == 2 else 4
        components = [f"{obj}[{index}]" for index in swizzle_indices]
        if len(swizzle_indices) == 3:
            components.append(pad_literal)

        return f"SIMD[{dtype}, {storage_width}]({', '.join(components)})"

    def is_duplicate_sensitive_expression(self, expr):
        if isinstance(expr, (FunctionCallNode, BinaryOpNode, TernaryOpNode)):
            return True
        if isinstance(expr, UnaryOpNode):
            return self.is_duplicate_sensitive_expression(expr.operand)
        if isinstance(expr, MemberAccessNode):
            return self.is_duplicate_sensitive_expression(expr.object)
        if isinstance(expr, ArrayAccessNode):
            return self.is_duplicate_sensitive_expression(
                expr.array
            ) or self.is_duplicate_sensitive_expression(expr.index)
        return False

    def format_literal(self, value, literal_type=None):
        if isinstance(value, bool):
            return "True" if value else "False"
        if literal_type == "bool" and isinstance(value, str):
            lower_value = value.lower()
            if lower_value == "true":
                return "True"
            if lower_value == "false":
                return "False"
        if isinstance(value, str):
            escaped = self.escape_literal(value)
            return f'"{escaped}"'
        return str(value)

    def escape_literal(self, value):
        text = str(value)
        escaped = []
        for index, char in enumerate(text):
            if char == "\n":
                escaped.append("\\n")
            elif char == "\r":
                escaped.append("\\r")
            elif char == "\t":
                escaped.append("\\t")
            elif char == '"' and (index == 0 or text[index - 1] != "\\"):
                escaped.append('\\"')
            else:
                escaped.append(char)
        return "".join(escaped)

    def map_type(self, vtype):
        """Map a CrossGL type name or type node to a Mojo type string."""
        if vtype is None:
            return "Float32"

        if hasattr(vtype, "name") or hasattr(vtype, "element_type"):
            vtype_str = self.convert_type_node_to_string(vtype)
        else:
            vtype_str = str(vtype)

        if "[" in vtype_str and "]" in vtype_str:
            base_type, size = parse_array_type(vtype_str)
            base_mapped = self.map_type(base_type)
            if size:
                return f"InlineArray[{base_mapped}, {size}]"
            else:
                return f"List[{base_mapped}]"

        if vtype_str in MOJO_MATRIX_TYPES:
            dtype, columns, rows = MOJO_MATRIX_TYPES[vtype_str]
            self.required_matrix_types.add((dtype, columns, rows))
            return self.matrix_type_name(dtype, columns, rows)

        if vtype_str in self.enum_types:
            return vtype_str

        mapped_type = self.type_mapping.get(vtype_str, vtype_str)
        if self.is_mojo_resource_type(mapped_type):
            self.required_resource_types.add(mapped_type)
        return mapped_type

    def is_resource_type_name(self, type_name):
        return self.is_mojo_resource_type(
            self.type_mapping.get(str(type_name), type_name)
        )

    def is_mojo_resource_type(self, type_name):
        return type_name in set(MOJO_RESOURCE_TYPE_MAPPING.values())

    def map_operator(self, op):
        op_map = {
            "PLUS": "+",
            "MINUS": "-",
            "MULTIPLY": "*",
            "DIVIDE": "/",
            "BITWISE_XOR": "^",
            "BITWISE_OR": "|",
            "BITWISE_AND": "&",
            "LESS_THAN": "<",
            "GREATER_THAN": ">",
            "ASSIGN_ADD": "+=",
            "ASSIGN_SUB": "-=",
            "ASSIGN_MUL": "*=",
            "ASSIGN_DIV": "/=",
            "ASSIGN_MOD": "%=",
            "ASSIGN_XOR": "^=",
            "ASSIGN_OR": "|=",
            "ASSIGN_AND": "&=",
            "LESS_EQUAL": "<=",
            "GREATER_EQUAL": ">=",
            "EQUAL": "==",
            "NOT_EQUAL": "!=",
            "AND": "and",
            "OR": "or",
            "&&": "and",
            "||": "or",
            "EQUALS": "=",
            "ASSIGN_SHIFT_LEFT": "<<=",
            "ASSIGN_SHIFT_RIGHT": ">>=",
            "LOGICAL_AND": "and",
            "LOGICAL_OR": "or",
            "BITWISE_SHIFT_RIGHT": ">>",
            "BITWISE_SHIFT_LEFT": "<<",
            "MOD": "%",
            "NOT": "not",
            "!": "not",
        }
        return op_map.get(op, op)

    def map_semantic(self, semantic):
        """Map a CrossGL semantic to the Mojo backend attribute name."""
        if semantic:
            return self.semantic_map.get(semantic, semantic)
        return ""
