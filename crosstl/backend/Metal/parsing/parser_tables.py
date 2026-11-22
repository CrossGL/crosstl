"""Token tables, qualifier sets, errors and helpers shared by the Metal parser modules."""

# Token groups for parsing
QUALIFIER_TOKENS = {
    "CONSTANT",
    "DEVICE",
    "THREADGROUP",
    "THREADGROUP_IMAGEBLOCK",
    "THREAD",
    "CONST",
    "CONSTEXPR",
    "STATIC",
    "INLINE",
    "VOLATILE",
    "RESTRICT",
    "READ",
    "WRITE",
    "READ_WRITE",
}
CAST_TYPE_QUALIFIER_NAMES = {
    "constant",
    "device",
    "threadgroup",
    "threadgroup_imageblock",
    "thread",
    "const",
    "constexpr",
    "static",
    "inline",
    "volatile",
    "restrict",
    "read",
    "write",
    "read_write",
}
TYPE_TOKENS = {
    "VOID",
    "FLOAT",
    "HALF",
    "DOUBLE",
    "INT",
    "UINT",
    "LONG",
    "ULONG",
    "SHORT",
    "USHORT",
    "CHAR",
    "UCHAR",
    "BOOL",
    "SIZE_T",
    "PTRDIFF_T",
    "INT64_T",
    "UINT64_T",
    "INT8_T",
    "UINT8_T",
    "INT16_T",
    "UINT16_T",
    "INT32_T",
    "UINT32_T",
    "VECTOR",
    "PACKED_VECTOR",
    "SIMD_VECTOR",
    "MATRIX",
    "SIMD_MATRIX",
    "ATOMIC_INT",
    "ATOMIC_UINT",
    "ATOMIC_BOOL",
    "TEXTURE1D",
    "TEXTURE1D_ARRAY",
    "TEXTURE2D",
    "TEXTURE2D_MS",
    "TEXTURE2D_MS_ARRAY",
    "TEXTURE3D",
    "TEXTURECUBE",
    "TEXTURECUBE_ARRAY",
    "TEXTURE2D_ARRAY",
    "TEXTUREBUFFER",
    "DEPTH2D",
    "DEPTH2D_ARRAY",
    "DEPTHCUBE",
    "DEPTHCUBE_ARRAY",
    "DEPTH2D_MS",
    "DEPTH2D_MS_ARRAY",
    "ACCELERATION_STRUCTURE",
    "INTERSECTION_FUNCTION_TABLE",
    "VISIBLE_FUNCTION_TABLE",
    "INDIRECT_COMMAND_BUFFER",
    "SAMPLER",
    "IDENTIFIER",
    "METAL",
    "ENUM",
    "TYPEDEF",
}
STAGE_TOKENS = {
    "VERTEX",
    "FRAGMENT",
    "KERNEL",
    "INTERSECTION",
    "ANYHIT",
    "CLOSESTHIT",
    "MISS",
    "CALLABLE",
    "MESH",
    "OBJECT",
    "AMPLIFICATION",
}
SCOPED_IDENTIFIER_PART_TOKENS = (
    TYPE_TOKENS
    | STAGE_TOKENS
    | {
        "METAL",
        "READ",
        "WRITE",
        "READ_WRITE",
    }
)
UNARY_KEYWORDS = {"SIZEOF", "ALIGNOF"}
# The Metal standard library headers define METAL_FUNC as ``inline __attribute__((__always_inline__))``. The
# importer does not read the SDK headers, so it keeps the macro as a function qualifier. Every other macro must
# be defined in the translated source or in a header on its include path; an undefined one is a syntax error.
MACRO_QUALIFIERS = {"METAL_FUNC"}
IDENTIFIER_TYPE_QUALIFIERS = MACRO_QUALIFIERS | {
    "__restrict",
    "__restrict__",
    "object_data",
}
IDENTIFIER_PREFIX_TYPE_QUALIFIERS = {"out", "inout"}
RAYTRACING_TYPE_QUALIFIERS = {"ray_data"}
TYPE_QUALIFIER_FUNCTIONS = {"coherent"}
SIGNED_TYPE_PREFIXES = {"signed", "unsigned"}
SIGNED_PREFIX_TYPE_TOKENS = {"CHAR", "SHORT", "INT", "LONG"}
# The Metal type ``unsigned`` makes of each integer a SIGNED_PREFIX_TYPE_TOKENS token spells (MSL 3.2 section 2.1).
UNSIGNED_INTEGER_TYPES = {
    "char": "uchar",
    "short": "ushort",
    "int": "uint",
    "long": "ulong",
}
KEYWORD_IDENTIFIER_TOKENS = {
    "BUFFER",
    "SAMPLER",
    "METAL",
    "RESTRICT",
    "READ",
    "WRITE",
    "READ_WRITE",
}
TYPE_IDENTIFIER_TOKENS = {"PACKED_VECTOR"}
GNU_EXTENSION_PREFIXES = {"__extension__"}
STRUCT_METHOD_PREFIXES = {"virtual"}
OPERATOR_OVERLOAD_TOKENS = {
    "PLUS",
    "MINUS",
    "MULTIPLY",
    "DIVIDE",
    "MOD",
    "EQUAL",
    "NOT_EQUAL",
    "LESS_THAN",
    "GREATER_THAN",
    "LESS_EQUAL",
    "GREATER_EQUAL",
    "PLUS_EQUALS",
    "MINUS_EQUALS",
    "MULTIPLY_EQUALS",
    "DIVIDE_EQUALS",
    "ASSIGN_MOD",
    "ASSIGN_AND",
    "ASSIGN_OR",
    "ASSIGN_XOR",
    "ASSIGN_SHIFT_LEFT",
    "ASSIGN_SHIFT_RIGHT",
    "AND",
    "OR",
    "NOT",
    "BITWISE_NOT",
    "BITWISE_AND",
    "BITWISE_OR",
    "BITWISE_XOR",
    "SHIFT_LEFT",
    "SHIFT_RIGHT",
    "INCREMENT",
    "DECREMENT",
    "EQUALS",
    "LBRACKET",
    "LPAREN",
}
TEMPLATE_VARIABLE_SUFFIX_FOLLOW_TOKENS = {
    "SEMICOLON",
    "COMMA",
    "RPAREN",
    "RBRACKET",
    "RBRACE",
    "QUESTION",
    "COLON",
    "AND",
    "OR",
    "BITWISE_OR",
    "BITWISE_XOR",
    "BITWISE_AND",
    "EQUAL",
    "NOT_EQUAL",
    "LESS_THAN",
    "GREATER_THAN",
    "LESS_EQUAL",
    "GREATER_EQUAL",
    "SHIFT_LEFT",
    "SHIFT_RIGHT",
    "PLUS",
    "MINUS",
    "MULTIPLY",
    "DIVIDE",
    "MOD",
}
TEMPLATE_IDENTIFIER_SUFFIX_FOLLOW_TOKENS = TEMPLATE_VARIABLE_SUFFIX_FOLLOW_TOKENS | {
    "LPAREN",
    "SCOPE",
    "LBRACE",
}
CONSTRUCTOR_TYPE_TOKENS = TYPE_TOKENS - {
    "VOID",
    "IDENTIFIER",
    "METAL",
    "ENUM",
    "TYPEDEF",
}
STAGE_ATTRIBUTE_NAMES = {
    "vertex",
    "fragment",
    "kernel",
    "intersection",
    "anyhit",
    "closesthit",
    "miss",
    "callable",
    "mesh",
    "object",
    "amplification",
}


class MetalCallOperatorAssociationError(SyntaxError):
    """Raised when a qualified call-operator body has no unique declaration."""

    project_diagnostic_code = (
        "project.translate.metal-call-operator-association-unresolved"
    )
    missing_capabilities = ("metal.out-of-line-call-operator-association",)

    def __init__(
        self,
        owner,
        signature,
        reason,
        *,
        definition_location=None,
        declaration_locations=(),
        candidates=(),
    ):
        self.owner = owner
        self.method_name = "operator()"
        self.signature = signature
        self.reason = reason
        self.definition_location = definition_location
        self.declaration_locations = tuple(declaration_locations)
        self.candidates = tuple(candidates)
        self.source_location = definition_location
        candidate_text = ", ".join(self.candidates) or "<none>"
        super().__init__(
            "Cannot associate out-of-line Metal call operator "
            f"'{owner}::{signature}': {reason}; declaration candidates are "
            f"{candidate_text}"
        )
