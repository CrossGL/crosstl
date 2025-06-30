import re
from collections import OrderedDict

TOKENS = OrderedDict(
    [
        # Comments
        ("COMMENT_SINGLE", r"//.*"),
        ("COMMENT_MULTI", r"/\*[\s\S]*?\*/"),
        # Keywords - Core Language
        ("SHADER", r"\bshader\b"),
        ("STRUCT", r"\bstruct\b"),
        ("ENUM", r"\benum\b"),
        ("IMPL", r"\bimpl\b"),
        ("TRAIT", r"\btrait\b"),
        ("CLASS", r"\bclass\b"),
        ("INTERFACE", r"\binterface\b"),
        ("NAMESPACE", r"\bnamespace\b"),
        ("MODULE", r"\bmodule\b"),
        ("IMPORT", r"\bimport\b"),
        ("USE", r"\buse\b"),
        ("FROM", r"\bfrom\b"),
        ("AS", r"\bas\b"),
        # Function/Method Keywords
        ("FUNCTION", r"\bfn\b"),
        ("VOID", r"\bvoid\b"),
        ("RETURN", r"\breturn\b"),
        ("YIELD", r"\byield\b"),
        ("ASYNC", r"\basync\b"),
        ("AWAIT", r"\bawait\b"),
        # Control Flow
        ("IF", r"\bif\b"),
        ("ELSE", r"\belse\b"),
        ("ELIF", r"\belif\b"),
        ("MATCH", r"\bmatch\b"),
        ("SWITCH", r"\bswitch\b"),
        ("CASE", r"\bcase\b"),
        ("DEFAULT", r"\bdefault\b"),
        ("FOR", r"\bfor\b"),
        ("WHILE", r"\bwhile\b"),
        ("LOOP", r"\bloop\b"),
        ("IN", r"\bin\b"),
        ("BREAK", r"\bbreak\b"),
        ("CONTINUE", r"\bcontinue\b"),
        # Variable/Memory Keywords
        ("LET", r"\blet\b"),
        ("VAR", r"\bvar\b"),
        ("MUT", r"\bmut\b"),
        ("CONST", r"\bconst\b"),
        ("STATIC", r"\bstatic\b"),
        ("EXTERN", r"\bextern\b"),
        ("UNIFORM", r"\buniform\b"),
        ("CBUFFER", r"\bcbuffer\b"),
        ("BUFFER", r"\bbuffer\b"),
        ("BUFFER", r"\bbuffer\b"),
        # Visibility/Access
        ("PUBLIC", r"\bpub\b"),
        ("PRIVATE", r"\bpriv\b"),
        ("PROTECTED", r"\bprotected\b"),
        ("INTERNAL", r"\binternal\b"),
        # Safety/Memory
        ("UNSAFE", r"\bunsafe\b"),
        ("SAFE", r"\bsafe\b"),
        ("REF", r"\bref\b"),
        ("BOX", r"\bbox\b"),
        ("MOVE", r"\bmove\b"),
        # Shader Stages
        ("VERTEX", r"\bvertex\b"),
        ("FRAGMENT", r"\bfragment\b"),
        ("COMPUTE", r"\bcompute\b"),
        ("GEOMETRY", r"\bgeometry\b"),
        ("TESSELLATION", r"\btessellation\b"),
        # GPU/Parallel Keywords
        ("KERNEL", r"\bkernel\b"),
        ("GLOBAL", r"\bglobal\b"),
        ("LOCAL", r"\blocal\b"),
        ("SHARED", r"\bshared\b"),
        ("THREADGROUP", r"\bthreadgroup\b"),
        ("WORKGROUP", r"\bworkgroup\b"),
        ("LAYOUT", r"\blayout\b"),
        # Types - Primitives
        ("BOOL", r"\bbool\b"),
        ("I8", r"\bi8\b"),
        ("I16", r"\bi16\b"),
        ("I32", r"\bi32\b"),
        ("I64", r"\bi64\b"),
        ("U8", r"\bu8\b"),
        ("U16", r"\bu16\b"),
        ("U32", r"\bu32\b"),
        ("U64", r"\bu64\b"),
        ("F16", r"\bf16\b"),
        ("F32", r"\bf32\b"),
        ("F64", r"\bf64\b"),
        ("INT", r"\bint\b"),
        ("UINT", r"\buint\b"),
        ("FLOAT", r"\bfloat\b"),
        ("DOUBLE", r"\bdouble\b"),
        ("HALF", r"\bhalf\b"),
        ("CHAR", r"\bchar\b"),
        ("STRING", r"\bstring\b"),
        # Types - Vectors
        ("VEC2", r"\bvec2\b"),
        ("VEC3", r"\bvec3\b"),
        ("VEC4", r"\bvec4\b"),
        ("IVEC2", r"\bivec2\b"),
        ("IVEC3", r"\bivec3\b"),
        ("IVEC4", r"\bivec4\b"),
        ("UVEC2", r"\buvec2\b"),
        ("UVEC3", r"\buvec3\b"),
        ("UVEC4", r"\buvec4\b"),
        ("BVEC2", r"\bbvec2\b"),
        ("BVEC3", r"\bbvec3\b"),
        ("BVEC4", r"\bbvec4\b"),
        # Types - Matrices
        ("MAT2", r"\bmat2\b"),
        ("MAT3", r"\bmat3\b"),
        ("MAT4", r"\bmat4\b"),
        ("MAT2X2", r"\bmat2x2\b"),
        ("MAT2X3", r"\bmat2x3\b"),
        ("MAT2X4", r"\bmat2x4\b"),
        ("MAT3X2", r"\bmat3x2\b"),
        ("MAT3X3", r"\bmat3x3\b"),
        ("MAT3X4", r"\bmat3x4\b"),
        ("MAT4X2", r"\bmat4x2\b"),
        ("MAT4X3", r"\bmat4x3\b"),
        ("MAT4X4", r"\bmat4x4\b"),
        # Types - Textures/Samplers
        ("TEXTURE1D", r"\btexture1d\b"),
        ("TEXTURE2D", r"\btexture2d\b"),
        ("TEXTURE3D", r"\btexture3d\b"),
        ("TEXTURECUBE", r"\btexturecube\b"),
        ("TEXTURE2DARRAY", r"\btexture2darray\b"),
        ("SAMPLER", r"\bsampler\b"),
        ("SAMPLER1D", r"\bsampler1d\b"),
        ("SAMPLER2D", r"\bsampler2d\b"),
        ("SAMPLER3D", r"\bsampler3d\b"),
        ("SAMPLERCUBE", r"\bsamplercube\b"),
        ("SAMPLER2DARRAY", r"\bsampler2darray\b"),
        # Generics/Templates
        ("WHERE", r"\bwhere\b"),
        ("IMPL_FOR", r"\bfor\b"),  # Different context from for loop
        # Attributes/Annotations
        ("ATTRIBUTE", r"@[a-zA-Z_][a-zA-Z_0-9]*"),
        ("HASH", r"#"),
        ("DOLLAR", r"\$"),
        # Literals
        ("FLOAT_NUMBER", r"\d*\.\d+[fF]?|\d+\.\d*[fF]?|\d+[fF]"),
        ("HEX_NUMBER", r"0[xX][0-9a-fA-F]+"),
        ("BIN_NUMBER", r"0[bB][01]+"),
        ("OCT_NUMBER", r"0[oO][0-7]+"),
        ("NUMBER", r"\d+"),
        ("STRING_LITERAL", r'"(?:[^"\\]|\\.)*"'),
        ("CHAR_LITERAL", r"'(?:[^'\\]|\\.)'"),
        # Operators - Assignment
        ("ASSIGN_ADD", r"\+="),
        ("ASSIGN_SUB", r"-="),
        ("ASSIGN_MUL", r"\*="),
        ("ASSIGN_DIV", r"/="),
        ("ASSIGN_MOD", r"%="),
        ("ASSIGN_AND", r"&="),
        ("ASSIGN_OR", r"\|="),
        ("ASSIGN_XOR", r"\^="),
        ("ASSIGN_SHIFT_LEFT", r"<<="),
        ("ASSIGN_SHIFT_RIGHT", r">>="),
        # Operators - Comparison
        ("EQUAL", r"=="),
        ("NOT_EQUAL", r"!="),
        ("LESS_EQUAL", r"<="),
        ("GREATER_EQUAL", r">="),
        ("SPACESHIP", r"<=>"),
        # Operators - Logical
        ("LOGICAL_AND", r"&&"),
        ("LOGICAL_OR", r"\|\|"),
        ("NOT", r"!"),
        # Operators - Bitwise
        ("BITWISE_SHIFT_LEFT", r"<<"),
        ("BITWISE_SHIFT_RIGHT", r">>"),
        ("BITWISE_AND", r"&"),
        ("BITWISE_OR", r"\|"),
        ("BITWISE_XOR", r"\^"),
        ("BITWISE_NOT", r"~"),
        # Operators - Arithmetic
        ("INCREMENT", r"\+\+"),
        ("DECREMENT", r"--"),
        ("PLUS", r"\+"),
        ("MINUS", r"-"),
        ("MULTIPLY", r"\*"),
        ("DIVIDE", r"/"),
        ("MOD", r"%"),
        ("POWER", r"\*\*"),
        # Operators - Other
        ("ARROW", r"->"),
        ("FAT_ARROW", r"=>"),
        ("DOUBLE_COLON", r"::"),
        ("RANGE", r"\.\."),
        ("RANGE_INCLUSIVE", r"\.\.="),
        ("ELVIS", r"\?:"),
        ("QUESTION", r"\?"),
        ("PIPE", r"\|"),
        # Punctuation
        ("SEMICOLON", r";"),
        ("COMMA", r","),
        ("DOT", r"\."),
        ("COLON", r":"),
        ("EQUALS", r"="),
        # Brackets
        ("LBRACE", r"\{"),
        ("RBRACE", r"\}"),
        ("LPAREN", r"\("),
        ("RPAREN", r"\)"),
        ("LBRACKET", r"\["),
        ("RBRACKET", r"\]"),
        ("LESS_THAN", r"<"),
        ("GREATER_THAN", r">"),
        # Special Characters
        ("AT", r"@"),
        ("AMPERSAND", r"&"),
        # Identifier (must be last)
        ("IDENTIFIER", r"[a-zA-Z_][a-zA-Z_0-9]*"),
        # Whitespace
        ("WHITESPACE", r"\s+"),
    ]
)

KEYWORDS = {
    # Core Language
    "shader": "SHADER",
    "struct": "STRUCT",
    "enum": "ENUM",
    "impl": "IMPL",
    "trait": "TRAIT",
    "class": "CLASS",
    "interface": "INTERFACE",
    "namespace": "NAMESPACE",
    "module": "MODULE",
    "import": "IMPORT",
    "use": "USE",
    "from": "FROM",
    "as": "AS",
    # Functions
    "fn": "FUNCTION",
    "void": "VOID",
    "return": "RETURN",
    "yield": "YIELD",
    "async": "ASYNC",
    "await": "AWAIT",
    # Control Flow
    "if": "IF",
    "else": "ELSE",
    "elif": "ELIF",
    "match": "MATCH",
    "switch": "SWITCH",
    "case": "CASE",
    "default": "DEFAULT",
    "for": "FOR",
    "while": "WHILE",
    "loop": "LOOP",
    "in": "IN",
    "break": "BREAK",
    "continue": "CONTINUE",
    # Variables
    "let": "LET",
    "var": "VAR",
    "mut": "MUT",
    "const": "CONST",
    "static": "STATIC",
    "extern": "EXTERN",
    "uniform": "UNIFORM",
    "cbuffer": "CBUFFER",
    "buffer": "BUFFER",
    # Visibility
    "pub": "PUBLIC",
    "priv": "PRIVATE",
    "protected": "PROTECTED",
    "internal": "INTERNAL",
    # Safety
    "unsafe": "UNSAFE",
    "safe": "SAFE",
    "ref": "REF",
    "box": "BOX",
    "move": "MOVE",
    # Shader Stages
    "vertex": "VERTEX",
    "fragment": "FRAGMENT",
    "compute": "COMPUTE",
    "geometry": "GEOMETRY",
    "tessellation": "TESSELLATION",
    # GPU
    "kernel": "KERNEL",
    "global": "GLOBAL",
    "local": "LOCAL",
    "shared": "SHARED",
    "threadgroup": "THREADGROUP",
    "workgroup": "WORKGROUP",
    "layout": "LAYOUT",
    # Types
    "bool": "BOOL",
    "i8": "I8",
    "i16": "I16",
    "i32": "I32",
    "i64": "I64",
    "u8": "U8",
    "u16": "U16",
    "u32": "U32",
    "u64": "U64",
    "f16": "F16",
    "f32": "F32",
    "f64": "F64",
    "int": "INT",
    "uint": "UINT",
    "float": "FLOAT",
    "double": "DOUBLE",
    "half": "HALF",
    "char": "CHAR",
    "string": "STRING",
    # Vectors
    "vec2": "VEC2",
    "vec3": "VEC3",
    "vec4": "VEC4",
    "ivec2": "IVEC2",
    "ivec3": "IVEC3",
    "ivec4": "IVEC4",
    "uvec2": "UVEC2",
    "uvec3": "UVEC3",
    "uvec4": "UVEC4",
    "bvec2": "BVEC2",
    "bvec3": "BVEC3",
    "bvec4": "BVEC4",
    # Matrices
    "mat2": "MAT2",
    "mat3": "MAT3",
    "mat4": "MAT4",
    "mat2x2": "MAT2X2",
    "mat2x3": "MAT2X3",
    "mat2x4": "MAT2X4",
    "mat3x2": "MAT3X2",
    "mat3x3": "MAT3X3",
    "mat3x4": "MAT3X4",
    "mat4x2": "MAT4X2",
    "mat4x3": "MAT4X3",
    "mat4x4": "MAT4X4",
    # Textures/Samplers
    "texture1d": "TEXTURE1D",
    "texture2d": "TEXTURE2D",
    "texture3d": "TEXTURE3D",
    "texturecube": "TEXTURECUBE",
    "texture2darray": "TEXTURE2DARRAY",
    "sampler": "SAMPLER",
    "sampler1d": "SAMPLER1D",
    "sampler2d": "SAMPLER2D",
    "sampler3d": "SAMPLER3D",
    "samplercube": "SAMPLERCUBE",
    "sampler2darray": "SAMPLER2DARRAY",
    # Generics
    "where": "WHERE",
}


class Lexer:
    """
    Production-ready lexer for CrossGL Universal IR.

    Supports comprehensive tokenization for all language features including:
    - Modern type systems (generics, traits, etc.)
    - Async/await patterns
    - Pattern matching
    - GPU programming constructs
    - Memory safety keywords
    - Complete operator set
    """

    def __init__(self, code):
        self.code = code
        self.tokens = []
        self.token_cache = {}
        self.regex_cache = self._compile_patterns()
        self.tokenize()

    def _compile_patterns(self):
        """Compile optimized regex with named groups for all tokens."""
        combined_pattern = "|".join(
            f"(?P<{name}>{pattern})" for name, pattern in TOKENS.items()
        )
        return re.compile(combined_pattern)

    def _get_cached_token(self, text, token_type):
        """Return cached token tuple for performance."""
        cache_key = (text, token_type)
        if cache_key not in self.token_cache:
            self.token_cache[cache_key] = (token_type, text)
        return self.token_cache[cache_key]

    def tokenize(self):
        """Tokenize the input code with comprehensive error handling."""
        pos = 0
        length = len(self.code)

        while pos < length:
            match = self.regex_cache.match(self.code, pos)
            if match:
                token_type = match.lastgroup
                text = match.group(token_type)

                # Handle keyword recognition
                if token_type == "IDENTIFIER" and text in KEYWORDS:
                    token_type = KEYWORDS[text]

                # Skip whitespace tokens
                if token_type != "WHITESPACE":
                    token = self._get_cached_token(text, token_type)
                    self.tokens.append(token)

                pos = match.end(0)
            else:
                # Enhanced error reporting
                bad_char = self.code[pos]
                line_num = self.code[:pos].count("\n") + 1
                col_num = pos - self.code.rfind("\n", 0, pos)

                # Show context around error
                line_start = self.code.rfind("\n", 0, pos) + 1
                line_end = self.code.find("\n", pos)
                if line_end == -1:
                    line_end = len(self.code)
                line_content = self.code[line_start:line_end]

                error_pointer = " " * (col_num - 1) + "^"

                raise SyntaxError(
                    f"Illegal character '{bad_char}' at line {line_num}, column {col_num}\n"
                    f"{line_content}\n{error_pointer}"
                )

        # Add EOF token
        self.tokens.append(self._get_cached_token(None, "EOF"))

    def get_tokens(self):
        """Return the list of tokens."""
        return self.tokens

    def debug_print(self):
        """Print tokens for debugging purposes."""
        for i, (token_type, text) in enumerate(self.tokens):
            print(f"{i:3d}: {token_type:20s} '{text}'")
