"""
Production-grade Type System for CrossTL.
Provides comprehensive type analysis, inference, and conversion.
"""

from typing import Dict, List, Optional, Set, Union, Any, Tuple
from enum import Enum, auto
from dataclasses import dataclass
from abc import ABC, abstractmethod


class TypeCategory(Enum):
    """High-level type categories."""
    PRIMITIVE = auto()
    VECTOR = auto()
    MATRIX = auto()
    ARRAY = auto()
    POINTER = auto()
    REFERENCE = auto()
    FUNCTION = auto()
    STRUCT = auto()
    ENUM = auto()
    GENERIC = auto()
    TEXTURE = auto()
    SAMPLER = auto()
    BUFFER = auto()


class PrimitiveTypeKind(Enum):
    """Primitive type classifications."""
    VOID = "void"
    BOOL = "bool"
    
    # Signed integers
    INT8 = "int8"
    INT16 = "int16" 
    INT32 = "int32"
    INT64 = "int64"
    
    # Unsigned integers
    UINT8 = "uint8"
    UINT16 = "uint16"
    UINT32 = "uint32"
    UINT64 = "uint64"
    
    # Floating point
    FLOAT16 = "float16"
    FLOAT32 = "float32"
    FLOAT64 = "float64"
    
    # Common aliases
    CHAR = "char"
    SHORT = "short"
    INT = "int"
    LONG = "long"
    UINT = "uint"
    FLOAT = "float"
    DOUBLE = "double"
    HALF = "half"


class StorageClass(Enum):
    """Memory storage classes."""
    DEFAULT = "default"
    CONST = "const"
    STATIC = "static"
    EXTERN = "extern"
    REGISTER = "register"
    
    # GPU-specific
    SHARED = "shared"        # CUDA __shared__, Metal threadgroup
    CONSTANT = "constant"    # CUDA __constant__, Metal constant
    GLOBAL = "global"        # CUDA __global__
    DEVICE = "device"        # CUDA __device__, Metal device
    UNIFORM = "uniform"      # GLSL uniform
    VARYING = "varying"      # GLSL varying
    ATTRIBUTE = "attribute"  # GLSL attribute


@dataclass(frozen=True)
class TypeDescriptor:
    """Immutable type descriptor with complete type information."""
    
    category: TypeCategory
    primitive_kind: Optional[PrimitiveTypeKind] = None
    vector_size: Optional[int] = None
    matrix_rows: Optional[int] = None
    matrix_cols: Optional[int] = None
    array_size: Optional[Union[int, str]] = None  # str for dynamic
    pointee_type: Optional['TypeDescriptor'] = None
    element_type: Optional['TypeDescriptor'] = None
    return_type: Optional['TypeDescriptor'] = None
    param_types: Optional[List['TypeDescriptor']] = None
    struct_name: Optional[str] = None
    generic_args: Optional[List['TypeDescriptor']] = None
    storage_class: StorageClass = StorageClass.DEFAULT
    is_mutable: bool = True
    is_const: bool = False
    qualifiers: Optional[Set[str]] = None
    
    def __post_init__(self):
        if self.qualifiers is None:
            object.__setattr__(self, 'qualifiers', set())
    
    def is_numeric(self) -> bool:
        """Check if type is numeric."""
        if self.category != TypeCategory.PRIMITIVE:
            return False
        numeric_types = {
            PrimitiveTypeKind.INT8, PrimitiveTypeKind.INT16, PrimitiveTypeKind.INT32, PrimitiveTypeKind.INT64,
            PrimitiveTypeKind.UINT8, PrimitiveTypeKind.UINT16, PrimitiveTypeKind.UINT32, PrimitiveTypeKind.UINT64,
            PrimitiveTypeKind.FLOAT16, PrimitiveTypeKind.FLOAT32, PrimitiveTypeKind.FLOAT64,
            PrimitiveTypeKind.CHAR, PrimitiveTypeKind.SHORT, PrimitiveTypeKind.INT, PrimitiveTypeKind.LONG,
            PrimitiveTypeKind.UINT, PrimitiveTypeKind.FLOAT, PrimitiveTypeKind.DOUBLE, PrimitiveTypeKind.HALF
        }
        return self.primitive_kind in numeric_types
    
    def is_integer(self) -> bool:
        """Check if type is integer."""
        if self.category != TypeCategory.PRIMITIVE:
            return False
        integer_types = {
            PrimitiveTypeKind.INT8, PrimitiveTypeKind.INT16, PrimitiveTypeKind.INT32, PrimitiveTypeKind.INT64,
            PrimitiveTypeKind.UINT8, PrimitiveTypeKind.UINT16, PrimitiveTypeKind.UINT32, PrimitiveTypeKind.UINT64,
            PrimitiveTypeKind.CHAR, PrimitiveTypeKind.SHORT, PrimitiveTypeKind.INT, PrimitiveTypeKind.LONG, PrimitiveTypeKind.UINT
        }
        return self.primitive_kind in integer_types
    
    def is_float(self) -> bool:
        """Check if type is floating point."""
        if self.category != TypeCategory.PRIMITIVE:
            return False
        float_types = {
            PrimitiveTypeKind.FLOAT16, PrimitiveTypeKind.FLOAT32, PrimitiveTypeKind.FLOAT64,
            PrimitiveTypeKind.FLOAT, PrimitiveTypeKind.DOUBLE, PrimitiveTypeKind.HALF
        }
        return self.primitive_kind in float_types
    
    def is_vector(self) -> bool:
        """Check if type is vector."""
        return self.category == TypeCategory.VECTOR
    
    def is_matrix(self) -> bool:
        """Check if type is matrix."""
        return self.category == TypeCategory.MATRIX
    
    def is_array(self) -> bool:
        """Check if type is array."""
        return self.category == TypeCategory.ARRAY
    
    def is_pointer_like(self) -> bool:
        """Check if type is pointer or reference."""
        return self.category in {TypeCategory.POINTER, TypeCategory.REFERENCE}
    
    def get_element_type(self) -> Optional['TypeDescriptor']:
        """Get element type for composite types."""
        return self.element_type
    
    def get_size(self) -> Optional[int]:
        """Get size for vectors, matrices, or arrays."""
        if self.is_vector():
            return self.vector_size
        elif self.is_matrix():
            return self.matrix_rows * self.matrix_cols if self.matrix_rows and self.matrix_cols else None
        elif self.is_array():
            return self.array_size if isinstance(self.array_size, int) else None
        return None
    
    def with_storage_class(self, storage_class: StorageClass) -> 'TypeDescriptor':
        """Create a new descriptor with different storage class."""
        return TypeDescriptor(
            category=self.category,
            primitive_kind=self.primitive_kind,
            vector_size=self.vector_size,
            matrix_rows=self.matrix_rows,
            matrix_cols=self.matrix_cols,
            array_size=self.array_size,
            pointee_type=self.pointee_type,
            element_type=self.element_type,
            return_type=self.return_type,
            param_types=self.param_types,
            struct_name=self.struct_name,
            generic_args=self.generic_args,
            storage_class=storage_class,
            is_mutable=self.is_mutable,
            is_const=self.is_const,
            qualifiers=self.qualifiers
        )
    
    def with_const(self, is_const: bool = True) -> 'TypeDescriptor':
        """Create a new descriptor with different const qualifier."""
        return TypeDescriptor(
            category=self.category,
            primitive_kind=self.primitive_kind,
            vector_size=self.vector_size,
            matrix_rows=self.matrix_rows,
            matrix_cols=self.matrix_cols,
            array_size=self.array_size,
            pointee_type=self.pointee_type,
            element_type=self.element_type,
            return_type=self.return_type,
            param_types=self.param_types,
            struct_name=self.struct_name,
            generic_args=self.generic_args,
            storage_class=self.storage_class,
            is_mutable=self.is_mutable,
            is_const=is_const,
            qualifiers=self.qualifiers
        )


class TypeSystem:
    """Production-grade type system with inference and checking."""
    
    def __init__(self):
        self.type_cache: Dict[str, TypeDescriptor] = {}
        self.struct_definitions: Dict[str, Dict[str, TypeDescriptor]] = {}
        self.function_signatures: Dict[str, TypeDescriptor] = {}
        
        # Initialize built-in types
        self._initialize_builtin_types()
    
    def _initialize_builtin_types(self):
        """Initialize all built-in primitive types."""
        for primitive in PrimitiveTypeKind:
            desc = TypeDescriptor(
                category=TypeCategory.PRIMITIVE,
                primitive_kind=primitive
            )
            self.type_cache[primitive.value] = desc
    
    def get_primitive_type(self, kind: PrimitiveTypeKind) -> TypeDescriptor:
        """Get a primitive type descriptor."""
        return TypeDescriptor(
            category=TypeCategory.PRIMITIVE,
            primitive_kind=kind
        )
    
    def get_vector_type(self, element_type: TypeDescriptor, size: int) -> TypeDescriptor:
        """Get a vector type descriptor."""
        if not element_type.category == TypeCategory.PRIMITIVE:
            raise ValueError("Vector element type must be primitive")
        if size not in {2, 3, 4}:
            raise ValueError("Vector size must be 2, 3, or 4")
        
        return TypeDescriptor(
            category=TypeCategory.VECTOR,
            element_type=element_type,
            vector_size=size
        )
    
    def get_matrix_type(self, element_type: TypeDescriptor, rows: int, cols: int) -> TypeDescriptor:
        """Get a matrix type descriptor.""" 
        if not element_type.category == TypeCategory.PRIMITIVE:
            raise ValueError("Matrix element type must be primitive")
        if rows not in {2, 3, 4} or cols not in {2, 3, 4}:
            raise ValueError("Matrix dimensions must be 2, 3, or 4")
        
        return TypeDescriptor(
            category=TypeCategory.MATRIX,
            element_type=element_type,
            matrix_rows=rows,
            matrix_cols=cols
        )
    
    def get_array_type(self, element_type: TypeDescriptor, size: Union[int, str]) -> TypeDescriptor:
        """Get an array type descriptor."""
        return TypeDescriptor(
            category=TypeCategory.ARRAY,
            element_type=element_type,
            array_size=size
        )
    
    def get_pointer_type(self, pointee_type: TypeDescriptor, is_mutable: bool = True) -> TypeDescriptor:
        """Get a pointer type descriptor."""
        return TypeDescriptor(
            category=TypeCategory.POINTER,
            pointee_type=pointee_type,
            is_mutable=is_mutable
        )
    
    def get_function_type(self, return_type: TypeDescriptor, param_types: List[TypeDescriptor]) -> TypeDescriptor:
        """Get a function type descriptor."""
        return TypeDescriptor(
            category=TypeCategory.FUNCTION,
            return_type=return_type,
            param_types=param_types
        )
    
    def register_struct(self, name: str, members: Dict[str, TypeDescriptor]):
        """Register a struct type."""
        self.struct_definitions[name] = members
        struct_type = TypeDescriptor(
            category=TypeCategory.STRUCT,
            struct_name=name
        )
        self.type_cache[name] = struct_type
    
    def register_function(self, name: str, signature: TypeDescriptor):
        """Register a function signature."""
        if signature.category != TypeCategory.FUNCTION:
            raise ValueError("Function signature must be function type")
        self.function_signatures[name] = signature
    
    def parse_type_string(self, type_str: str) -> TypeDescriptor:
        """Parse a type string into a TypeDescriptor."""
        type_str = type_str.strip()
        
        # Check cache first
        if type_str in self.type_cache:
            return self.type_cache[type_str]
        
        # Parse storage qualifiers
        storage_class = StorageClass.DEFAULT
        is_const = False
        qualifiers = set()
        
        parts = type_str.split()
        core_type = parts[-1]  # Last part is the actual type
        
        for part in parts[:-1]:
            if part == "const":
                is_const = True
            elif part == "static":
                storage_class = StorageClass.STATIC
            elif part == "extern":
                storage_class = StorageClass.EXTERN
            elif part == "__shared__":
                storage_class = StorageClass.SHARED
            elif part == "__constant__":
                storage_class = StorageClass.CONSTANT
            elif part == "uniform":
                storage_class = StorageClass.UNIFORM
            else:
                qualifiers.add(part)
        
        # Parse core type
        desc = self._parse_core_type(core_type)
        
        # Apply qualifiers
        if storage_class != StorageClass.DEFAULT or is_const or qualifiers:
            desc = TypeDescriptor(
                category=desc.category,
                primitive_kind=desc.primitive_kind,
                vector_size=desc.vector_size,
                matrix_rows=desc.matrix_rows,
                matrix_cols=desc.matrix_cols,
                array_size=desc.array_size,
                pointee_type=desc.pointee_type,
                element_type=desc.element_type,
                return_type=desc.return_type,
                param_types=desc.param_types,
                struct_name=desc.struct_name,
                generic_args=desc.generic_args,
                storage_class=storage_class,
                is_mutable=desc.is_mutable,
                is_const=is_const,
                qualifiers=qualifiers
            )
        
        # Cache and return
        self.type_cache[type_str] = desc
        return desc
    
    def _parse_core_type(self, type_str: str) -> TypeDescriptor:
        """Parse the core type without qualifiers."""
        
        # Handle pointers
        if "*" in type_str:
            base_type_str = type_str.replace("*", "").strip()
            pointee_type = self._parse_core_type(base_type_str)
            return TypeDescriptor(
                category=TypeCategory.POINTER,
                pointee_type=pointee_type
            )
        
        # Handle arrays
        if "[" in type_str and "]" in type_str:
            bracket_start = type_str.index("[")
            bracket_end = type_str.rindex("]")
            base_type_str = type_str[:bracket_start].strip()
            size_str = type_str[bracket_start+1:bracket_end].strip()
            
            element_type = self._parse_core_type(base_type_str)
            
            # Parse array size
            array_size = None
            if size_str:
                try:
                    array_size = int(size_str)
                except ValueError:
                    array_size = size_str  # Dynamic size
            
            return TypeDescriptor(
                category=TypeCategory.ARRAY,
                element_type=element_type,
                array_size=array_size
            )
        
        # Handle generic types
        if "<" in type_str and ">" in type_str:
            generic_start = type_str.index("<")
            generic_end = type_str.rindex(">")
            base_name = type_str[:generic_start].strip()
            args_str = type_str[generic_start+1:generic_end].strip()
            
            # Parse generic arguments
            generic_args = []
            if args_str:
                for arg_str in args_str.split(","):
                    arg_type = self._parse_core_type(arg_str.strip())
                    generic_args.append(arg_type)
            
            # Handle special generic types
            if base_name in {"Vec", "Vector"}:
                if len(generic_args) >= 2:
                    element_type = generic_args[0]
                    size = generic_args[1]
                    if isinstance(size, int):
                        return TypeDescriptor(
                            category=TypeCategory.VECTOR,
                            element_type=element_type,
                            vector_size=size
                        )
            
            # Generic struct/type
            return TypeDescriptor(
                category=TypeCategory.GENERIC,
                struct_name=base_name,
                generic_args=generic_args
            )
        
        # Handle primitive types
        for primitive in PrimitiveTypeKind:
            if type_str == primitive.value:
                return TypeDescriptor(
                    category=TypeCategory.PRIMITIVE,
                    primitive_kind=primitive
                )
        
        # Handle vector types
        vector_patterns = {
            r"vec(\d)": (PrimitiveTypeKind.FLOAT32, None),
            r"ivec(\d)": (PrimitiveTypeKind.INT32, None),
            r"uvec(\d)": (PrimitiveTypeKind.UINT32, None),
            r"bvec(\d)": (PrimitiveTypeKind.BOOL, None),
            r"dvec(\d)": (PrimitiveTypeKind.FLOAT64, None),
            r"float(\d)": (PrimitiveTypeKind.FLOAT32, None),
            r"int(\d)": (PrimitiveTypeKind.INT32, None),
            r"uint(\d)": (PrimitiveTypeKind.UINT32, None),
            r"double(\d)": (PrimitiveTypeKind.FLOAT64, None),
        }
        
        for pattern, (element_kind, _) in vector_patterns.items():
            match = re.match(pattern, type_str)
            if match:
                size = int(match.group(1))
                element_type = TypeDescriptor(
                    category=TypeCategory.PRIMITIVE,
                    primitive_kind=element_kind
                )
                return TypeDescriptor(
                    category=TypeCategory.VECTOR,
                    element_type=element_type,
                    vector_size=size
                )
        
        # Handle matrix types
        matrix_patterns = {
            r"mat(\d)": (PrimitiveTypeKind.FLOAT32, None, None),
            r"mat(\d)x(\d)": (PrimitiveTypeKind.FLOAT32, None, None),
            r"dmat(\d)": (PrimitiveTypeKind.FLOAT64, None, None),
            r"float(\d)x(\d)": (PrimitiveTypeKind.FLOAT32, None, None),
            r"double(\d)x(\d)": (PrimitiveTypeKind.FLOAT64, None, None),
        }
        
        for pattern, (element_kind, _, _) in matrix_patterns.items():
            match = re.match(pattern, type_str)
            if match:
                if "x" in type_str:
                    rows = int(match.group(1))
                    cols = int(match.group(2))
                else:
                    # Square matrix
                    rows = cols = int(match.group(1))
                
                element_type = TypeDescriptor(
                    category=TypeCategory.PRIMITIVE,
                    primitive_kind=element_kind
                )
                return TypeDescriptor(
                    category=TypeCategory.MATRIX,
                    element_type=element_type,
                    matrix_rows=rows,
                    matrix_cols=cols
                )
        
        # Handle texture types
        texture_types = {
            "sampler1D", "sampler2D", "sampler3D", "samplerCube",
            "sampler2DArray", "samplerCubeArray", "sampler1DArray",
            "isampler1D", "isampler2D", "isampler3D", "isamplerCube",
            "usampler1D", "usampler2D", "usampler3D", "usamplerCube",
            "texture1d", "texture2d", "texture3d", "texturecube",
            "Texture1D", "Texture2D", "Texture3D", "TextureCube"
        }
        
        if type_str in texture_types:
            return TypeDescriptor(
                category=TypeCategory.TEXTURE,
                struct_name=type_str
            )
        
        # Assume it's a user-defined struct
        return TypeDescriptor(
            category=TypeCategory.STRUCT,
            struct_name=type_str
        )
    
    def are_compatible(self, from_type: TypeDescriptor, to_type: TypeDescriptor) -> bool:
        """Check if types are compatible for assignment/conversion."""
        # Exact match
        if from_type == to_type:
            return True
        
        # Same category compatibility
        if from_type.category == to_type.category:
            if from_type.category == TypeCategory.PRIMITIVE:
                return self._are_primitives_compatible(from_type, to_type)
            elif from_type.category == TypeCategory.VECTOR:
                return (from_type.vector_size == to_type.vector_size and
                       self.are_compatible(from_type.element_type, to_type.element_type))
            elif from_type.category == TypeCategory.MATRIX:
                return (from_type.matrix_rows == to_type.matrix_rows and
                       from_type.matrix_cols == to_type.matrix_cols and
                       self.are_compatible(from_type.element_type, to_type.element_type))
        
        # Cross-category compatibility
        return self._are_cross_category_compatible(from_type, to_type)
    
    def _are_primitives_compatible(self, from_type: TypeDescriptor, to_type: TypeDescriptor) -> bool:
        """Check primitive type compatibility."""
        # Same type
        if from_type.primitive_kind == to_type.primitive_kind:
            return True
        
        # Numeric conversions
        if from_type.is_numeric() and to_type.is_numeric():
            # Allow widening conversions
            return True  # Simplified - could be more strict
        
        return False
    
    def _are_cross_category_compatible(self, from_type: TypeDescriptor, to_type: TypeDescriptor) -> bool:
        """Check cross-category type compatibility."""
        # Pointer/reference dereference
        if from_type.category in {TypeCategory.POINTER, TypeCategory.REFERENCE}:
            return self.are_compatible(from_type.pointee_type, to_type)
        
        # Array to pointer decay
        if from_type.category == TypeCategory.ARRAY and to_type.category == TypeCategory.POINTER:
            return self.are_compatible(from_type.element_type, to_type.pointee_type)
        
        return False
    
    def can_implicitly_convert(self, from_type: TypeDescriptor, to_type: TypeDescriptor) -> bool:
        """Check if implicit conversion is allowed."""
        return self.are_compatible(from_type, to_type)
    
    def requires_explicit_cast(self, from_type: TypeDescriptor, to_type: TypeDescriptor) -> bool:
        """Check if explicit cast is required."""
        return not self.can_implicitly_convert(from_type, to_type)
    
    def get_common_type(self, types: List[TypeDescriptor]) -> Optional[TypeDescriptor]:
        """Find the common type for a list of types."""
        if not types:
            return None
        if len(types) == 1:
            return types[0]
        
        # Start with first type and find common with others
        common = types[0]
        for type_desc in types[1:]:
            common = self._find_common_type(common, type_desc)
            if common is None:
                return None
        
        return common
    
    def _find_common_type(self, type1: TypeDescriptor, type2: TypeDescriptor) -> Optional[TypeDescriptor]:
        """Find common type between two types."""
        # Same type
        if type1 == type2:
            return type1
        
        # Both numeric primitives
        if (type1.category == TypeCategory.PRIMITIVE and type2.category == TypeCategory.PRIMITIVE and
            type1.is_numeric() and type2.is_numeric()):
            
            # Promote to wider type
            if type1.is_float() or type2.is_float():
                # Float promotion
                if type1.primitive_kind in {PrimitiveTypeKind.FLOAT64, PrimitiveTypeKind.DOUBLE} or \
                   type2.primitive_kind in {PrimitiveTypeKind.FLOAT64, PrimitiveTypeKind.DOUBLE}:
                    return self.get_primitive_type(PrimitiveTypeKind.FLOAT64)
                else:
                    return self.get_primitive_type(PrimitiveTypeKind.FLOAT32)
            else:
                # Integer promotion - simplified
                return self.get_primitive_type(PrimitiveTypeKind.INT32)
        
        return None


class LanguageTypeMapper:
    """Maps universal types to language-specific representations."""
    
    # Comprehensive backend mappings
    LANGUAGE_MAPPINGS = {
        "cuda": {
            PrimitiveTypeKind.VOID: "void",
            PrimitiveTypeKind.BOOL: "bool",
            PrimitiveTypeKind.INT8: "char",
            PrimitiveTypeKind.UINT8: "unsigned char",
            PrimitiveTypeKind.INT16: "short",
            PrimitiveTypeKind.UINT16: "unsigned short", 
            PrimitiveTypeKind.INT32: "int",
            PrimitiveTypeKind.UINT32: "unsigned int",
            PrimitiveTypeKind.INT64: "long long",
            PrimitiveTypeKind.UINT64: "unsigned long long",
            PrimitiveTypeKind.FLOAT16: "half",
            PrimitiveTypeKind.FLOAT32: "float",
            PrimitiveTypeKind.FLOAT64: "double",
            PrimitiveTypeKind.CHAR: "char",
            PrimitiveTypeKind.SHORT: "short",
            PrimitiveTypeKind.INT: "int",
            PrimitiveTypeKind.LONG: "long long",
            PrimitiveTypeKind.UINT: "unsigned int",
            PrimitiveTypeKind.FLOAT: "float",
            PrimitiveTypeKind.DOUBLE: "double",
            PrimitiveTypeKind.HALF: "half",
        },
        
        "metal": {
            PrimitiveTypeKind.VOID: "void",
            PrimitiveTypeKind.BOOL: "bool",
            PrimitiveTypeKind.INT32: "int",
            PrimitiveTypeKind.UINT32: "uint",
            PrimitiveTypeKind.FLOAT16: "half",
            PrimitiveTypeKind.FLOAT32: "float",
            PrimitiveTypeKind.FLOAT64: "double",
            PrimitiveTypeKind.INT: "int",
            PrimitiveTypeKind.UINT: "uint",
            PrimitiveTypeKind.FLOAT: "float",
            PrimitiveTypeKind.DOUBLE: "double", 
            PrimitiveTypeKind.HALF: "half",
        },
        
        "directx": {
            PrimitiveTypeKind.VOID: "void",
            PrimitiveTypeKind.BOOL: "bool",
            PrimitiveTypeKind.INT32: "int",
            PrimitiveTypeKind.UINT32: "uint",
            PrimitiveTypeKind.FLOAT16: "half",
            PrimitiveTypeKind.FLOAT32: "float", 
            PrimitiveTypeKind.FLOAT64: "double",
            PrimitiveTypeKind.INT: "int",
            PrimitiveTypeKind.UINT: "uint",
            PrimitiveTypeKind.FLOAT: "float",
            PrimitiveTypeKind.DOUBLE: "double",
            PrimitiveTypeKind.HALF: "half",
        },
        
        "opengl": {
            PrimitiveTypeKind.VOID: "void",
            PrimitiveTypeKind.BOOL: "bool",
            PrimitiveTypeKind.INT32: "int", 
            PrimitiveTypeKind.UINT32: "uint",
            PrimitiveTypeKind.FLOAT32: "float",
            PrimitiveTypeKind.FLOAT64: "double",
            PrimitiveTypeKind.INT: "int",
            PrimitiveTypeKind.UINT: "uint",
            PrimitiveTypeKind.FLOAT: "float",
            PrimitiveTypeKind.DOUBLE: "double",
        },
        
        "vulkan": {
            PrimitiveTypeKind.VOID: "void",
            PrimitiveTypeKind.BOOL: "bool",
            PrimitiveTypeKind.INT32: "int",
            PrimitiveTypeKind.UINT32: "uint",
            PrimitiveTypeKind.FLOAT32: "float",
            PrimitiveTypeKind.FLOAT64: "double",
            PrimitiveTypeKind.INT: "int", 
            PrimitiveTypeKind.UINT: "uint",
            PrimitiveTypeKind.FLOAT: "float",
            PrimitiveTypeKind.DOUBLE: "double",
        },
        
        "rust": {
            PrimitiveTypeKind.VOID: "()",
            PrimitiveTypeKind.BOOL: "bool",
            PrimitiveTypeKind.INT8: "i8",
            PrimitiveTypeKind.UINT8: "u8",
            PrimitiveTypeKind.INT16: "i16",
            PrimitiveTypeKind.UINT16: "u16",
            PrimitiveTypeKind.INT32: "i32",
            PrimitiveTypeKind.UINT32: "u32",
            PrimitiveTypeKind.INT64: "i64",
            PrimitiveTypeKind.UINT64: "u64",
            PrimitiveTypeKind.FLOAT32: "f32",
            PrimitiveTypeKind.FLOAT64: "f64",
            PrimitiveTypeKind.CHAR: "i8",
            PrimitiveTypeKind.SHORT: "i16",
            PrimitiveTypeKind.INT: "i32",
            PrimitiveTypeKind.LONG: "i64",
            PrimitiveTypeKind.UINT: "u32",
            PrimitiveTypeKind.FLOAT: "f32",
            PrimitiveTypeKind.DOUBLE: "f64",
        },
        
        "mojo": {
            PrimitiveTypeKind.VOID: "None",
            PrimitiveTypeKind.BOOL: "Bool",
            PrimitiveTypeKind.INT8: "Int8",
            PrimitiveTypeKind.UINT8: "UInt8",
            PrimitiveTypeKind.INT16: "Int16",
            PrimitiveTypeKind.UINT16: "UInt16",
            PrimitiveTypeKind.INT32: "Int32",
            PrimitiveTypeKind.UINT32: "UInt32",
            PrimitiveTypeKind.INT64: "Int64",
            PrimitiveTypeKind.UINT64: "UInt64",
            PrimitiveTypeKind.FLOAT16: "Float16",
            PrimitiveTypeKind.FLOAT32: "Float32",
            PrimitiveTypeKind.FLOAT64: "Float64",
            PrimitiveTypeKind.INT: "Int32",
            PrimitiveTypeKind.UINT: "UInt32",
            PrimitiveTypeKind.FLOAT: "Float32",
            PrimitiveTypeKind.DOUBLE: "Float64",
            PrimitiveTypeKind.HALF: "Float16",
        },
        
        "hip": {
            # HIP uses CUDA-like types
            PrimitiveTypeKind.VOID: "void",
            PrimitiveTypeKind.BOOL: "bool",
            PrimitiveTypeKind.INT32: "int",
            PrimitiveTypeKind.UINT32: "unsigned int",
            PrimitiveTypeKind.FLOAT32: "float",
            PrimitiveTypeKind.FLOAT64: "double", 
            PrimitiveTypeKind.INT: "int",
            PrimitiveTypeKind.UINT: "unsigned int",
            PrimitiveTypeKind.FLOAT: "float",
            PrimitiveTypeKind.DOUBLE: "double",
        },
        
        "slang": {
            PrimitiveTypeKind.VOID: "void",
            PrimitiveTypeKind.BOOL: "bool",
            PrimitiveTypeKind.INT32: "int",
            PrimitiveTypeKind.UINT32: "uint",
            PrimitiveTypeKind.FLOAT32: "float",
            PrimitiveTypeKind.FLOAT64: "double",
            PrimitiveTypeKind.INT: "int",
            PrimitiveTypeKind.UINT: "uint",
            PrimitiveTypeKind.FLOAT: "float",
            PrimitiveTypeKind.DOUBLE: "double",
        }
    }
    
    # Vector type templates per language
    VECTOR_TEMPLATES = {
        "cuda": {
            2: ("{base}2", "make_{base}2"),
            3: ("{base}3", "make_{base}3"), 
            4: ("{base}4", "make_{base}4"),
        },
        "metal": {
            2: ("{base}2", "{base}2"),
            3: ("{base}3", "{base}3"),
            4: ("{base}4", "{base}4"),
        },
        "directx": {
            2: ("{base}2", "{base}2"),
            3: ("{base}3", "{base}3"),
            4: ("{base}4", "{base}4"),
        },
        "opengl": {
            2: ("vec2", "vec2"),
            3: ("vec3", "vec3"),
            4: ("vec4", "vec4"),
        },
        "vulkan": {
            2: ("vec2", "vec2"),
            3: ("vec3", "vec3"), 
            4: ("vec4", "vec4"),
        },
        "rust": {
            2: ("Vec2<{base}>", "Vec2::new"),
            3: ("Vec3<{base}>", "Vec3::new"),
            4: ("Vec4<{base}>", "Vec4::new"),
        },
        "mojo": {
            2: ("SIMD[DType.{base}, 2]", "SIMD[DType.{base}, 2]"),
            3: ("SIMD[DType.{base}, 3]", "SIMD[DType.{base}, 3]"),
            4: ("SIMD[DType.{base}, 4]", "SIMD[DType.{base}, 4]"),
        },
    }
    
    @classmethod
    def map_type(cls, type_desc: TypeDescriptor, target_language: str) -> str:
        """Map a type descriptor to target language syntax."""
        target_language = target_language.lower()
        
        if type_desc.category == TypeCategory.PRIMITIVE:
            mapping = cls.LANGUAGE_MAPPINGS.get(target_language, {})
            return mapping.get(type_desc.primitive_kind, type_desc.primitive_kind.value)
        
        elif type_desc.category == TypeCategory.VECTOR:
            return cls._map_vector_type(type_desc, target_language)
        
        elif type_desc.category == TypeCategory.MATRIX:
            return cls._map_matrix_type(type_desc, target_language)
        
        elif type_desc.category == TypeCategory.ARRAY:
            element_type = cls.map_type(type_desc.element_type, target_language)
            if type_desc.array_size:
                return f"{element_type}[{type_desc.array_size}]"
            else:
                return f"{element_type}[]"
        
        elif type_desc.category == TypeCategory.POINTER:
            pointee_type = cls.map_type(type_desc.pointee_type, target_language)
            if target_language == "rust":
                mutability = "mut " if type_desc.is_mutable else ""
                return f"&{mutability}{pointee_type}"
            else:
                return f"{pointee_type}*"
        
        elif type_desc.category == TypeCategory.STRUCT:
            return type_desc.struct_name
        
        elif type_desc.category == TypeCategory.TEXTURE:
            return cls._map_texture_type(type_desc, target_language)
        
        return "void"  # Fallback
    
    @classmethod
    def _map_vector_type(cls, type_desc: TypeDescriptor, target_language: str) -> str:
        """Map vector types to target language."""
        if target_language not in cls.VECTOR_TEMPLATES:
            return f"vec{type_desc.vector_size}"
        
        templates = cls.VECTOR_TEMPLATES[target_language]
        if type_desc.vector_size not in templates:
            return f"vec{type_desc.vector_size}"
        
        type_template, _ = templates[type_desc.vector_size]
        
        # Get base type name
        if type_desc.element_type and type_desc.element_type.primitive_kind:
            element_mapping = cls.LANGUAGE_MAPPINGS.get(target_language, {})
            base_type = element_mapping.get(type_desc.element_type.primitive_kind, "float")
            
            # Handle special cases
            if target_language == "opengl":
                # GLSL vector naming
                if type_desc.element_type.primitive_kind in {PrimitiveTypeKind.INT32, PrimitiveTypeKind.INT}:
                    return f"ivec{type_desc.vector_size}"
                elif type_desc.element_type.primitive_kind in {PrimitiveTypeKind.UINT32, PrimitiveTypeKind.UINT}:
                    return f"uvec{type_desc.vector_size}"
                elif type_desc.element_type.primitive_kind == PrimitiveTypeKind.BOOL:
                    return f"bvec{type_desc.vector_size}"
                else:
                    return f"vec{type_desc.vector_size}"
            
            return type_template.format(base=base_type)
        
        return type_template.format(base="float")
    
    @classmethod
    def _map_matrix_type(cls, type_desc: TypeDescriptor, target_language: str) -> str:
        """Map matrix types to target language."""
        rows = type_desc.matrix_rows
        cols = type_desc.matrix_cols
        
        if target_language in {"cuda", "metal", "directx", "hip", "slang"}:
            base_type = "float"
            if type_desc.element_type and type_desc.element_type.primitive_kind:
                element_mapping = cls.LANGUAGE_MAPPINGS.get(target_language, {})
                base_type = element_mapping.get(type_desc.element_type.primitive_kind, "float")
            
            if rows == cols:
                return f"{base_type}{rows}x{cols}"
            else:
                return f"{base_type}{rows}x{cols}"
        
        elif target_language in {"opengl", "vulkan"}:
            if rows == cols:
                return f"mat{rows}"
            else:
                return f"mat{rows}x{cols}"
        
        elif target_language == "rust":
            base_type = "f32"
            if type_desc.element_type and type_desc.element_type.primitive_kind:
                element_mapping = cls.LANGUAGE_MAPPINGS.get(target_language, {})
                base_type = element_mapping.get(type_desc.element_type.primitive_kind, "f32")
            
            return f"Mat{rows}x{cols}<{base_type}>"
        
        elif target_language == "mojo":
            base_type = "Float32"
            if type_desc.element_type and type_desc.element_type.primitive_kind:
                element_mapping = cls.LANGUAGE_MAPPINGS.get(target_language, {})
                base_type = element_mapping.get(type_desc.element_type.primitive_kind, "Float32")
            
            return f"Matrix[DType.{base_type.lower()}, {rows}, {cols}]"
        
        return f"mat{rows}x{cols}"
    
    @classmethod
    def _map_texture_type(cls, type_desc: TypeDescriptor, target_language: str) -> str:
        """Map texture types to target language."""
        texture_name = type_desc.struct_name
        
        texture_mappings = {
            "cuda": {
                "sampler2D": "texture<float4, 2>",
                "samplerCube": "textureCube<float4>",
                "sampler3D": "texture<float4, 3>",
            },
            "metal": {
                "sampler2D": "texture2d<float>",
                "samplerCube": "texturecube<float>",
                "sampler3D": "texture3d<float>",
            },
            "directx": {
                "sampler2D": "Texture2D",
                "samplerCube": "TextureCube",
                "sampler3D": "Texture3D",
            },
            "opengl": {
                "sampler2D": "sampler2D",
                "samplerCube": "samplerCube", 
                "sampler3D": "sampler3D",
            },
            "vulkan": {
                "sampler2D": "sampler2D",
                "samplerCube": "samplerCube",
                "sampler3D": "sampler3D",
            }
        }
        
        if target_language in texture_mappings:
            return texture_mappings[target_language].get(texture_name, texture_name)
        
        return texture_name
