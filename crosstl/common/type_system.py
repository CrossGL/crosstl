"""
Centralized Type System for CrossTL.
This module provides unified type definitions and mappings across all backends,
eliminating redundancy and ensuring consistency.
"""

from enum import Enum
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass


class BaseType(Enum):
    """Base primitive types supported across all backends."""
    
    VOID = "void"
    BOOL = "bool"
    
    # Integer types
    INT8 = "int8"
    UINT8 = "uint8"
    INT16 = "int16"
    UINT16 = "uint16"
    INT32 = "int32"
    UINT32 = "uint32"
    INT64 = "int64"
    UINT64 = "uint64"
    
    # Floating point types
    FLOAT16 = "float16"
    FLOAT32 = "float32"
    FLOAT64 = "float64"
    
    # Common aliases
    INT = "int"        # Usually int32
    UINT = "uint"      # Usually uint32
    FLOAT = "float"    # Usually float32
    DOUBLE = "double"  # Usually float64
    HALF = "half"      # Usually float16
    CHAR = "char"      # Usually int8


class VectorType(Enum):
    """Vector types with standard naming."""
    
    # Float vectors
    VEC2 = "vec2"
    VEC3 = "vec3"
    VEC4 = "vec4"
    
    # Integer vectors
    IVEC2 = "ivec2"
    IVEC3 = "ivec3"
    IVEC4 = "ivec4"
    
    # Unsigned integer vectors
    UVEC2 = "uvec2"
    UVEC3 = "uvec3"
    UVEC4 = "uvec4"
    
    # Boolean vectors
    BVEC2 = "bvec2"
    BVEC3 = "bvec3"
    BVEC4 = "bvec4"
    
    # Double vectors
    DVEC2 = "dvec2"
    DVEC3 = "dvec3"
    DVEC4 = "dvec4"


class MatrixType(Enum):
    """Matrix types with standard naming."""
    
    MAT2 = "mat2"
    MAT3 = "mat3"
    MAT4 = "mat4"
    MAT2X3 = "mat2x3"
    MAT2X4 = "mat2x4"
    MAT3X2 = "mat3x2"
    MAT3X4 = "mat3x4"
    MAT4X2 = "mat4x2"
    MAT4X3 = "mat4x3"
    
    # Double matrices
    DMAT2 = "dmat2"
    DMAT3 = "dmat3"
    DMAT4 = "dmat4"


class TextureType(Enum):
    """Texture types supported across backends."""
    
    SAMPLER1D = "sampler1D"
    SAMPLER2D = "sampler2D"
    SAMPLER3D = "sampler3D"
    SAMPLERCUBE = "samplerCube"
    SAMPLER2DARRAY = "sampler2DArray"
    SAMPLERCUBEARRAY = "samplerCubeArray"
    
    # Integer textures
    ISAMPLER1D = "isampler1D"
    ISAMPLER2D = "isampler2D"
    ISAMPLER3D = "isampler3D"
    ISAMPLERCUBE = "isamplerCube"
    
    # Unsigned integer textures
    USAMPLER1D = "usampler1D"
    USAMPLER2D = "usampler2D"
    USAMPLER3D = "usampler3D"
    USAMPLERCUBE = "usamplerCube"


@dataclass
class TypeDescriptor:
    """Comprehensive type descriptor for CrossTL types."""
    
    base_type: BaseType
    vector_size: Optional[int] = None
    matrix_rows: Optional[int] = None
    matrix_cols: Optional[int] = None
    array_size: Optional[int] = None
    is_pointer: bool = False
    is_reference: bool = False
    is_const: bool = False
    is_volatile: bool = False
    storage_class: Optional[str] = None  # For CUDA: __shared__, __constant__, etc.
    
    def is_vector(self) -> bool:
        """Check if this is a vector type."""
        return self.vector_size is not None and self.vector_size > 1
    
    def is_matrix(self) -> bool:
        """Check if this is a matrix type."""
        return self.matrix_rows is not None and self.matrix_cols is not None
    
    def is_array(self) -> bool:
        """Check if this is an array type."""
        return self.array_size is not None
    
    def is_scalar(self) -> bool:
        """Check if this is a scalar type."""
        return not (self.is_vector() or self.is_matrix() or self.is_array())
    
    def get_element_type(self) -> 'TypeDescriptor':
        """Get the element type for vectors, matrices, and arrays."""
        return TypeDescriptor(
            base_type=self.base_type,
            is_const=self.is_const,
            is_volatile=self.is_volatile
        )


class UniversalTypeMapper:
    """Universal type mapping system for all backends."""
    
    # Backend-specific type mappings
    BACKEND_MAPPINGS = {
        "cuda": {
            BaseType.VOID: "void",
            BaseType.BOOL: "bool",
            BaseType.INT8: "char",
            BaseType.UINT8: "unsigned char", 
            BaseType.INT16: "short",
            BaseType.UINT16: "unsigned short",
            BaseType.INT32: "int",
            BaseType.UINT32: "unsigned int",
            BaseType.INT64: "long long",
            BaseType.UINT64: "unsigned long long",
            BaseType.FLOAT16: "half",
            BaseType.FLOAT32: "float",
            BaseType.FLOAT64: "double",
            BaseType.INT: "int",
            BaseType.UINT: "unsigned int",
            BaseType.FLOAT: "float",
            BaseType.DOUBLE: "double",
            BaseType.HALF: "half",
            BaseType.CHAR: "char",
            
            # Vectors
            VectorType.VEC2: "float2",
            VectorType.VEC3: "float3",
            VectorType.VEC4: "float4", 
            VectorType.IVEC2: "int2",
            VectorType.IVEC3: "int3",
            VectorType.IVEC4: "int4",
            VectorType.UVEC2: "uint2",
            VectorType.UVEC3: "uint3",
            VectorType.UVEC4: "uint4",
            VectorType.DVEC2: "double2",
            VectorType.DVEC3: "double3",
            VectorType.DVEC4: "double4",
        },
        
        "metal": {
            BaseType.VOID: "void",
            BaseType.BOOL: "bool",
            BaseType.INT32: "int",
            BaseType.UINT32: "uint",
            BaseType.FLOAT16: "half",
            BaseType.FLOAT32: "float",
            BaseType.FLOAT64: "double",
            BaseType.INT: "int",
            BaseType.UINT: "uint", 
            BaseType.FLOAT: "float",
            BaseType.DOUBLE: "double",
            BaseType.HALF: "half",
            
            # Vectors
            VectorType.VEC2: "float2",
            VectorType.VEC3: "float3",
            VectorType.VEC4: "float4",
            VectorType.IVEC2: "int2",
            VectorType.IVEC3: "int3", 
            VectorType.IVEC4: "int4",
            VectorType.UVEC2: "uint2",
            VectorType.UVEC3: "uint3",
            VectorType.UVEC4: "uint4",
            
            # Matrices
            MatrixType.MAT2: "float2x2",
            MatrixType.MAT3: "float3x3",
            MatrixType.MAT4: "float4x4",
            
            # Textures
            TextureType.SAMPLER2D: "texture2d<float>",
            TextureType.SAMPLERCUBE: "texturecube<float>",
        },
        
        "directx": {
            BaseType.VOID: "void",
            BaseType.BOOL: "bool",
            BaseType.INT32: "int",
            BaseType.UINT32: "uint", 
            BaseType.FLOAT16: "half",
            BaseType.FLOAT32: "float",
            BaseType.FLOAT64: "double",
            BaseType.INT: "int",
            BaseType.UINT: "uint",
            BaseType.FLOAT: "float",
            BaseType.DOUBLE: "double",
            BaseType.HALF: "half",
            
            # Vectors
            VectorType.VEC2: "float2",
            VectorType.VEC3: "float3",
            VectorType.VEC4: "float4",
            VectorType.IVEC2: "int2",
            VectorType.IVEC3: "int3",
            VectorType.IVEC4: "int4",
            VectorType.UVEC2: "uint2",
            VectorType.UVEC3: "uint3",
            VectorType.UVEC4: "uint4",
            
            # Matrices
            MatrixType.MAT2: "float2x2",
            MatrixType.MAT3: "float3x3",
            MatrixType.MAT4: "float4x4",
            
            # Textures
            TextureType.SAMPLER2D: "Texture2D",
            TextureType.SAMPLERCUBE: "TextureCube",
        },
        
        "opengl": {
            # GLSL uses universal types mostly as-is
            BaseType.VOID: "void",
            BaseType.BOOL: "bool",
            BaseType.INT32: "int",
            BaseType.UINT32: "uint",
            BaseType.FLOAT32: "float",
            BaseType.FLOAT64: "double",
            BaseType.INT: "int",
            BaseType.UINT: "uint",
            BaseType.FLOAT: "float",
            BaseType.DOUBLE: "double",
            
            # Vectors (keep as-is)
            VectorType.VEC2: "vec2",
            VectorType.VEC3: "vec3",
            VectorType.VEC4: "vec4",
            VectorType.IVEC2: "ivec2",
            VectorType.IVEC3: "ivec3",
            VectorType.IVEC4: "ivec4",
            VectorType.UVEC2: "uvec2",
            VectorType.UVEC3: "uvec3",
            VectorType.UVEC4: "uvec4",
            VectorType.BVEC2: "bvec2",
            VectorType.BVEC3: "bvec3",
            VectorType.BVEC4: "bvec4",
            
            # Matrices
            MatrixType.MAT2: "mat2",
            MatrixType.MAT3: "mat3",
            MatrixType.MAT4: "mat4",
            
            # Textures
            TextureType.SAMPLER2D: "sampler2D",
            TextureType.SAMPLERCUBE: "samplerCube",
        },
        
        "vulkan": {
            # Vulkan uses GLSL-like types
            BaseType.VOID: "void",
            BaseType.BOOL: "bool", 
            BaseType.INT32: "int",
            BaseType.UINT32: "uint",
            BaseType.FLOAT32: "float",
            BaseType.FLOAT64: "double",
            BaseType.INT: "int",
            BaseType.UINT: "uint",
            BaseType.FLOAT: "float",
            BaseType.DOUBLE: "double",
            
            # Vectors
            VectorType.VEC2: "vec2",
            VectorType.VEC3: "vec3", 
            VectorType.VEC4: "vec4",
            VectorType.IVEC2: "ivec2",
            VectorType.IVEC3: "ivec3",
            VectorType.IVEC4: "ivec4",
            VectorType.UVEC2: "uvec2",
            VectorType.UVEC3: "uvec3",
            VectorType.UVEC4: "uvec4",
            
            # Matrices
            MatrixType.MAT2: "mat2",
            MatrixType.MAT3: "mat3",
            MatrixType.MAT4: "mat4",
        },
        
        "rust": {
            BaseType.VOID: "()",
            BaseType.BOOL: "bool",
            BaseType.INT8: "i8",
            BaseType.UINT8: "u8",
            BaseType.INT16: "i16", 
            BaseType.UINT16: "u16",
            BaseType.INT32: "i32",
            BaseType.UINT32: "u32",
            BaseType.INT64: "i64",
            BaseType.UINT64: "u64",
            BaseType.FLOAT32: "f32",
            BaseType.FLOAT64: "f64",
            BaseType.INT: "i32",
            BaseType.UINT: "u32",
            BaseType.FLOAT: "f32",
            BaseType.DOUBLE: "f64",
            BaseType.CHAR: "i8",
            
            # Vectors (using hypothetical Rust GPU types)
            VectorType.VEC2: "Vec2<f32>",
            VectorType.VEC3: "Vec3<f32>",
            VectorType.VEC4: "Vec4<f32>",
            VectorType.IVEC2: "Vec2<i32>",
            VectorType.IVEC3: "Vec3<i32>",
            VectorType.IVEC4: "Vec4<i32>",
            VectorType.UVEC2: "Vec2<u32>",
            VectorType.UVEC3: "Vec3<u32>",
            VectorType.UVEC4: "Vec4<u32>",
            
            # Matrices
            MatrixType.MAT2: "Mat2<f32>",
            MatrixType.MAT3: "Mat3<f32>",
            MatrixType.MAT4: "Mat4<f32>",
        },
        
        "mojo": {
            BaseType.VOID: "None",
            BaseType.BOOL: "Bool",
            BaseType.INT8: "Int8",
            BaseType.UINT8: "UInt8",
            BaseType.INT16: "Int16",
            BaseType.UINT16: "UInt16", 
            BaseType.INT32: "Int32",
            BaseType.UINT32: "UInt32",
            BaseType.INT64: "Int64",
            BaseType.UINT64: "UInt64",
            BaseType.FLOAT16: "Float16",
            BaseType.FLOAT32: "Float32",
            BaseType.FLOAT64: "Float64",
            BaseType.INT: "Int32",
            BaseType.UINT: "UInt32",
            BaseType.FLOAT: "Float32",
            BaseType.DOUBLE: "Float64",
            BaseType.HALF: "Float16",
            
            # Vectors (using SIMD types)
            VectorType.VEC2: "SIMD[DType.float32, 2]",
            VectorType.VEC3: "SIMD[DType.float32, 3]",
            VectorType.VEC4: "SIMD[DType.float32, 4]",
            VectorType.IVEC2: "SIMD[DType.int32, 2]",
            VectorType.IVEC3: "SIMD[DType.int32, 3]",
            VectorType.IVEC4: "SIMD[DType.int32, 4]",
            
            # Matrices
            MatrixType.MAT2: "Matrix[DType.float32, 2, 2]",
            MatrixType.MAT3: "Matrix[DType.float32, 3, 3]", 
            MatrixType.MAT4: "Matrix[DType.float32, 4, 4]",
        },
        
        "hip": {
            # HIP uses CUDA-like types
            BaseType.VOID: "void",
            BaseType.BOOL: "bool",
            BaseType.INT32: "int",
            BaseType.UINT32: "unsigned int",
            BaseType.FLOAT32: "float",
            BaseType.FLOAT64: "double",
            BaseType.INT: "int",
            BaseType.UINT: "unsigned int",
            BaseType.FLOAT: "float",
            BaseType.DOUBLE: "double",
            
            # Vectors
            VectorType.VEC2: "float2",
            VectorType.VEC3: "float3",
            VectorType.VEC4: "float4",
            VectorType.IVEC2: "int2",
            VectorType.IVEC3: "int3",
            VectorType.IVEC4: "int4",
            VectorType.UVEC2: "uint2",
            VectorType.UVEC3: "uint3",
            VectorType.UVEC4: "uint4",
        },
        
        "slang": {
            # Slang uses DirectX-like types
            BaseType.VOID: "void",
            BaseType.BOOL: "bool",
            BaseType.INT32: "int",
            BaseType.UINT32: "uint",
            BaseType.FLOAT32: "float",
            BaseType.FLOAT64: "double",
            BaseType.INT: "int",
            BaseType.UINT: "uint",
            BaseType.FLOAT: "float",
            BaseType.DOUBLE: "double",
            
            # Vectors
            VectorType.VEC2: "float2",
            VectorType.VEC3: "float3",
            VectorType.VEC4: "float4",
            VectorType.IVEC2: "int2",
            VectorType.IVEC3: "int3",
            VectorType.IVEC4: "int4",
            VectorType.UVEC2: "uint2",
            VectorType.UVEC3: "uint3",
            VectorType.UVEC4: "uint4",
            
            # Matrices  
            MatrixType.MAT2: "float2x2",
            MatrixType.MAT3: "float3x3",
            MatrixType.MAT4: "float4x4",
            
            # Textures
            TextureType.SAMPLER2D: "Texture2D",
            TextureType.SAMPLERCUBE: "TextureCube",
        }
    }
    
    @classmethod
    def map_type(cls, type_name: Union[str, BaseType, VectorType, MatrixType, TextureType], 
                 backend: str) -> str:
        """Map a universal type to backend-specific syntax."""
        backend = backend.lower()
        
        # Handle enum types
        if isinstance(type_name, (BaseType, VectorType, MatrixType, TextureType)):
            type_key = type_name
        else:
            # Try to find matching enum
            type_key = cls._find_type_enum(str(type_name))
        
        # Get backend mapping
        if backend in cls.BACKEND_MAPPINGS:
            return cls.BACKEND_MAPPINGS[backend].get(type_key, str(type_name))
        
        return str(type_name)
    
    @classmethod
    def _find_type_enum(cls, type_name: str) -> Union[BaseType, VectorType, MatrixType, TextureType, str]:
        """Find the corresponding enum for a string type name."""
        
        # Check base types
        for base_type in BaseType:
            if base_type.value == type_name:
                return base_type
        
        # Check vector types
        for vec_type in VectorType:
            if vec_type.value == type_name:
                return vec_type
        
        # Check matrix types
        for mat_type in MatrixType:
            if mat_type.value == type_name:
                return mat_type
        
        # Check texture types
        for tex_type in TextureType:
            if tex_type.value == type_name:
                return tex_type
        
        return type_name
    
    @classmethod
    def add_backend_mapping(cls, backend: str, mappings: Dict[Union[BaseType, VectorType, MatrixType, TextureType], str]):
        """Add or update mappings for a backend."""
        backend = backend.lower()
        if backend not in cls.BACKEND_MAPPINGS:
            cls.BACKEND_MAPPINGS[backend] = {}
        cls.BACKEND_MAPPINGS[backend].update(mappings)
    
    @classmethod
    def get_supported_backends(cls) -> List[str]:
        """Get list of supported backends."""
        return list(cls.BACKEND_MAPPINGS.keys())
    
    @classmethod
    def parse_type_descriptor(cls, type_string: str) -> TypeDescriptor:
        """Parse a type string into a TypeDescriptor."""
        # This is a simplified implementation - can be enhanced
        
        # Handle basic cases first
        if type_string in [bt.value for bt in BaseType]:
            base_type = next(bt for bt in BaseType if bt.value == type_string)
            return TypeDescriptor(base_type=base_type)
        
        # Handle vector types
        if type_string in [vt.value for vt in VectorType]:
            vec_type = next(vt for vt in VectorType if vt.value == type_string)
            
            # Extract vector size and base type
            if "vec" in type_string:
                size = int(type_string[-1])
                if type_string.startswith("i"):
                    base = BaseType.INT32
                elif type_string.startswith("u"):
                    base = BaseType.UINT32
                elif type_string.startswith("b"):
                    base = BaseType.BOOL
                elif type_string.startswith("d"):
                    base = BaseType.FLOAT64
                else:
                    base = BaseType.FLOAT32
                
                return TypeDescriptor(base_type=base, vector_size=size)
        
        # Handle matrix types
        if type_string in [mt.value for mt in MatrixType]:
            mat_type = next(mt for mt in MatrixType if mt.value == type_string)
            
            # Extract matrix dimensions
            if "mat" in type_string:
                if "x" in type_string:
                    parts = type_string.replace("mat", "").split("x")
                    rows, cols = int(parts[0]), int(parts[1])
                else:
                    # Square matrix
                    size = int(type_string[-1])
                    rows, cols = size, size
                
                base = BaseType.FLOAT64 if type_string.startswith("d") else BaseType.FLOAT32
                return TypeDescriptor(base_type=base, matrix_rows=rows, matrix_cols=cols)
        
        # Default fallback
        return TypeDescriptor(base_type=BaseType.VOID)


class FunctionMapper:
    """Universal function mapping system."""
    
    UNIVERSAL_FUNCTIONS = {
        # Math functions
        "abs": "abs",
        "min": "min", 
        "max": "max",
        "clamp": "clamp",
        "floor": "floor",
        "ceil": "ceil",
        "round": "round",
        "sqrt": "sqrt",
        "pow": "pow",
        "exp": "exp",
        "log": "log",
        "sin": "sin",
        "cos": "cos",
        "tan": "tan",
        "asin": "asin",
        "acos": "acos",
        "atan": "atan",
        
        # Vector functions
        "dot": "dot",
        "cross": "cross",
        "length": "length",
        "normalize": "normalize",
        "distance": "distance",
        "reflect": "reflect",
        "refract": "refract",
        
        # Matrix functions
        "transpose": "transpose",
        "determinant": "determinant", 
        "inverse": "inverse",
        
        # Texture functions
        "texture": "texture",
        "textureLod": "textureLod",
    }
    
    BACKEND_OVERRIDES = {
        "cuda": {
            "abs": "fabsf",  # For float, fabs for double
            "sqrt": "sqrtf",
            "pow": "powf",
            "sin": "sinf",
            "cos": "cosf",
            "tan": "tanf",
            "floor": "floorf",
            "ceil": "ceilf",
        },
        "metal": {
            "texture": "sample",
        },
        "mojo": {
            "sqrt": "math.sqrt",
            "pow": "math.pow",
            "sin": "math.sin",
            "cos": "math.cos",
            "abs": "math.abs",
            "dot": "simd.dot",
            "cross": "simd.cross",
        }
    }
    
    @classmethod
    def map_function(cls, func_name: str, backend: str) -> str:
        """Map a universal function to backend-specific name."""
        backend = backend.lower()
        
        # Check for backend override
        if backend in cls.BACKEND_OVERRIDES:
            override = cls.BACKEND_OVERRIDES[backend].get(func_name)
            if override:
                return override
        
        # Return universal mapping
        return cls.UNIVERSAL_FUNCTIONS.get(func_name, func_name)


class SemanticMapper:
    """Universal semantic mapping system."""
    
    UNIVERSAL_SEMANTICS = {
        # Vertex input semantics
        "position": "POSITION",
        "normal": "NORMAL",
        "tangent": "TANGENT", 
        "binormal": "BINORMAL",
        "texcoord": "TEXCOORD",
        "color": "COLOR",
        "vertex_id": "VERTEX_ID",
        "instance_id": "INSTANCE_ID",
        
        # Fragment output semantics
        "color_output": "COLOR_OUTPUT",
        "depth_output": "DEPTH_OUTPUT",
    }
    
    BACKEND_MAPPINGS = {
        "directx": {
            "POSITION": "SV_Position", 
            "VERTEX_ID": "SV_VertexID",
            "INSTANCE_ID": "SV_InstanceID",
            "COLOR_OUTPUT": "SV_Target",
            "DEPTH_OUTPUT": "SV_Depth",
        },
        "metal": {
            "POSITION": "[[position]]",
            "VERTEX_ID": "[[vertex_id]]",
            "INSTANCE_ID": "[[instance_id]]",
            "COLOR_OUTPUT": "[[color(0)]]",
            "DEPTH_OUTPUT": "[[depth(any)]]",
        },
        "opengl": {
            "POSITION": "gl_Position",
            "COLOR_OUTPUT": "gl_FragColor", 
            "DEPTH_OUTPUT": "gl_FragDepth",
        },
        "vulkan": {
            "POSITION": "gl_Position",
            "COLOR_OUTPUT": "gl_FragColor",
            "DEPTH_OUTPUT": "gl_FragDepth",
        }
    }
    
    @classmethod
    def map_semantic(cls, semantic: str, backend: str) -> str:
        """Map a semantic to backend-specific syntax."""
        backend = backend.lower()
        
        # Normalize semantic to universal form
        universal = cls.UNIVERSAL_SEMANTICS.get(semantic.lower(), semantic.upper())
        
        # Map to backend-specific form
        if backend in cls.BACKEND_MAPPINGS:
            return cls.BACKEND_MAPPINGS[backend].get(universal, semantic)
        
        return semantic


# Convenience functions for type checking

def is_integer_type(type_name: str) -> bool:
    """Check if type is an integer type."""
    integer_types = {BaseType.INT8, BaseType.UINT8, BaseType.INT16, BaseType.UINT16,
                    BaseType.INT32, BaseType.UINT32, BaseType.INT64, BaseType.UINT64,
                    BaseType.INT, BaseType.UINT, BaseType.CHAR}
    type_enum = UniversalTypeMapper._find_type_enum(type_name)
    return type_enum in integer_types

def is_float_type(type_name: str) -> bool:
    """Check if type is a floating point type."""
    float_types = {BaseType.FLOAT16, BaseType.FLOAT32, BaseType.FLOAT64,
                  BaseType.FLOAT, BaseType.DOUBLE, BaseType.HALF}
    type_enum = UniversalTypeMapper._find_type_enum(type_name)
    return type_enum in float_types

def is_vector_type(type_name: str) -> bool:
    """Check if type is a vector type."""
    type_enum = UniversalTypeMapper._find_type_enum(type_name)
    return isinstance(type_enum, VectorType)

def is_matrix_type(type_name: str) -> bool:
    """Check if type is a matrix type."""
    type_enum = UniversalTypeMapper._find_type_enum(type_name)
    return isinstance(type_enum, MatrixType)

def get_vector_size(type_name: str) -> Optional[int]:
    """Get the size of a vector type."""
    if is_vector_type(type_name):
        if type_name.endswith("2"):
            return 2
        elif type_name.endswith("3"):
            return 3
        elif type_name.endswith("4"):
            return 4
    return None

def get_matrix_dimensions(type_name: str) -> Optional[Tuple[int, int]]:
    """Get the dimensions of a matrix type."""
    if is_matrix_type(type_name):
        if "x" in type_name:
            # Non-square matrix
            parts = type_name.replace("mat", "").replace("dmat", "").split("x")
            return (int(parts[0]), int(parts[1]))
        else:
            # Square matrix
            size = int(type_name[-1])
            return (size, size)
    return None
