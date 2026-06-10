"""Lexer for tokenizing CrossGL source code."""

import ast
import re
from collections import OrderedDict
from pathlib import Path

TOKENS = OrderedDict(
    [
        ("COMMENT_SINGLE", r"//.*"),
        ("COMMENT_MULTI", r"/\*[\s\S]*?\*/"),
        ("PREPROCESSOR", r"#[^\n]*"),
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
        ("FUNCTION", r"\bfn\b"),
        ("VOID", r"\bvoid\b"),
        ("RETURN", r"\breturn\b"),
        ("YIELD", r"\byield\b"),
        ("ASYNC", r"\basync\b"),
        ("AWAIT", r"\bawait\b"),
        ("IF", r"\bif\b"),
        ("ELSE", r"\belse\b"),
        ("ELIF", r"\belif\b"),
        ("MATCH", r"\bmatch\b"),
        ("SWITCH", r"\bswitch\b"),
        ("CASE", r"\bcase\b"),
        ("DEFAULT", r"\bdefault\b"),
        ("FOR", r"\bfor\b"),
        ("DO", r"\bdo\b"),
        ("WHILE", r"\bwhile\b"),
        ("LOOP", r"\bloop\b"),
        ("IN", r"\bin\b"),
        ("BREAK", r"\bbreak\b"),
        ("CONTINUE", r"\bcontinue\b"),
        ("LET", r"\blet\b"),
        ("VAR", r"\bvar\b"),
        ("MUT", r"\bmut\b"),
        ("CONST", r"\bconst\b"),
        ("STATIC", r"\bstatic\b"),
        ("EXTERN", r"\bextern\b"),
        ("UNIFORM", r"\buniform\b"),
        ("CBUFFER", r"\bcbuffer\b"),
        ("BUFFER", r"\bbuffer\b"),
        ("PUBLIC", r"\bpub\b"),
        ("PRIVATE", r"\bpriv\b"),
        ("PROTECTED", r"\bprotected\b"),
        ("INTERNAL", r"\binternal\b"),
        ("UNSAFE", r"\bunsafe\b"),
        ("SAFE", r"\bsafe\b"),
        ("REF", r"\bref\b"),
        ("BOX", r"\bbox\b"),
        ("MOVE", r"\bmove\b"),
        ("VERTEX", r"\bvertex\b"),
        ("FRAGMENT", r"\bfragment\b"),
        ("COMPUTE", r"\bcompute\b"),
        ("GEOMETRY", r"\bgeometry\b"),
        ("TESSELLATION", r"\btessellation\b"),
        ("KERNEL", r"\bkernel\b"),
        ("GLOBAL", r"\bglobal\b"),
        ("LOCAL", r"\blocal\b"),
        ("SHARED", r"\bshared\b"),
        ("THREADGROUP_IMAGEBLOCK", r"\bthreadgroup_imageblock\b"),
        ("THREADGROUP", r"\bthreadgroup\b"),
        ("WORKGROUP", r"\bworkgroup\b"),
        ("LAYOUT", r"\blayout\b"),
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
        ("VEC2", r"\bvec2\b"),
        ("VEC3", r"\bvec3\b"),
        ("VEC4", r"\bvec4\b"),
        ("IVEC2", r"\bivec2\b"),
        ("IVEC3", r"\bivec3\b"),
        ("IVEC4", r"\bivec4\b"),
        ("UVEC2", r"\buvec2\b"),
        ("UVEC3", r"\buvec3\b"),
        ("UVEC4", r"\buvec4\b"),
        ("DVEC2", r"\bdvec2\b"),
        ("DVEC3", r"\bdvec3\b"),
        ("DVEC4", r"\bdvec4\b"),
        ("BVEC2", r"\bbvec2\b"),
        ("BVEC3", r"\bbvec3\b"),
        ("BVEC4", r"\bbvec4\b"),
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
        ("DMAT2", r"\bdmat2\b"),
        ("DMAT3", r"\bdmat3\b"),
        ("DMAT4", r"\bdmat4\b"),
        ("DMAT2X2", r"\bdmat2x2\b"),
        ("DMAT2X3", r"\bdmat2x3\b"),
        ("DMAT2X4", r"\bdmat2x4\b"),
        ("DMAT3X2", r"\bdmat3x2\b"),
        ("DMAT3X3", r"\bdmat3x3\b"),
        ("DMAT3X4", r"\bdmat3x4\b"),
        ("DMAT4X2", r"\bdmat4x2\b"),
        ("DMAT4X3", r"\bdmat4x3\b"),
        ("DMAT4X4", r"\bdmat4x4\b"),
        ("TEXTURE1D", r"\btexture1d\b"),
        ("TEXTURE2D", r"\btexture2d\b"),
        ("TEXTURE3D", r"\btexture3d\b"),
        ("TEXTURECUBE", r"\btexturecube\b"),
        ("TEXTURE2DARRAY", r"\btexture2darray\b"),
        ("SAMPLER", r"\bsampler\b"),
        ("SAMPLER1D", r"\bsampler1d\b"),
        ("SAMPLER1DARRAY", r"\bsampler1[Dd][Aa]rray\b"),
        ("SAMPLER2D", r"\bsampler2d\b"),
        ("SAMPLER3D", r"\bsampler3d\b"),
        ("SAMPLERCUBE", r"\bsamplercube\b"),
        ("SAMPLER2DARRAY", r"\bsampler2darray\b"),
        ("SAMPLER2DSHADOW", r"\bsampler2dshadow\b"),
        ("SAMPLER2DARRAYSHADOW", r"\bsampler2darrayshadow\b"),
        ("SAMPLERCUBESHADOW", r"\bsamplercubeshadow\b"),
        ("SAMPLERCUBEARRAY", r"\bsamplercubearray\b"),
        ("SAMPLERCUBEARRAYSHADOW", r"\bsamplercubearrayshadow\b"),
        ("SAMPLER2DMS", r"\bsampler2dms\b"),
        ("SAMPLER2DMSARRAY", r"\bsampler2dmsarray\b"),
        ("IIMAGE1D", r"\biimage1[Dd]\b"),
        ("IIMAGE1DARRAY", r"\biimage1[Dd][Aa]rray\b"),
        ("IIMAGE2D", r"\biimage2[Dd]\b"),
        ("IIMAGE3D", r"\biimage3[Dd]\b"),
        ("IIMAGECUBE", r"\biimage[Cc]ube\b"),
        ("IIMAGECUBEARRAY", r"\biimage[Cc]ube[Aa]rray\b"),
        ("IIMAGE2DARRAY", r"\biimage2[Dd][Aa]rray\b"),
        ("IIMAGE2DMS", r"\biimage2[Dd][Mm][Ss]\b"),
        ("IIMAGE2DMSARRAY", r"\biimage2[Dd][Mm][Ss][Aa]rray\b"),
        ("UIMAGE1D", r"\buimage1[Dd]\b"),
        ("UIMAGE1DARRAY", r"\buimage1[Dd][Aa]rray\b"),
        ("UIMAGE2D", r"\buimage2[Dd]\b"),
        ("UIMAGE3D", r"\buimage3[Dd]\b"),
        ("UIMAGECUBE", r"\buimage[Cc]ube\b"),
        ("UIMAGECUBEARRAY", r"\buimage[Cc]ube[Aa]rray\b"),
        ("UIMAGE2DARRAY", r"\buimage2[Dd][Aa]rray\b"),
        ("UIMAGE2DMS", r"\buimage2[Dd][Mm][Ss]\b"),
        ("UIMAGE2DMSARRAY", r"\buimage2[Dd][Mm][Ss][Aa]rray\b"),
        ("IMAGE1D", r"\bimage1[Dd]\b"),
        ("IMAGE1DARRAY", r"\bimage1[Dd][Aa]rray\b"),
        ("IMAGE2D", r"\bimage2[Dd]\b"),
        ("IMAGE3D", r"\bimage3[Dd]\b"),
        ("IMAGECUBE", r"\bimage[Cc]ube\b"),
        ("IMAGECUBEARRAY", r"\bimage[Cc]ube[Aa]rray\b"),
        ("IMAGE2DARRAY", r"\bimage2[Dd][Aa]rray\b"),
        ("IMAGE2DMS", r"\bimage2[Dd][Mm][Ss]\b"),
        ("IMAGE2DMSARRAY", r"\bimage2[Dd][Mm][Ss][Aa]rray\b"),
        ("WHERE", r"\bwhere\b"),
        ("IMPL_FOR", r"\bfor\b"),  # Different context from for loop
        ("ATTRIBUTE", r"@[a-zA-Z_][a-zA-Z_0-9]*"),
        ("HASH", r"#"),
        ("DOLLAR", r"\$"),
        (
            "FLOAT_NUMBER",
            r"0[xX](?:[0-9a-fA-F]+(?:\.[0-9a-fA-F]*)?|\.[0-9a-fA-F]+)[pP][+-]?\d+[fF]?"
            r"|(?:\d+\.\d*|\.\d+|\d+)[eE][+-]?\d+[fF]?"
            r"|\d*\.\d+[fF]?|\d+\.(?!\.)\d*[fF]?|\d+[fF]",
        ),
        ("HEX_NUMBER", r"0[xX][0-9a-fA-F]+[uU]?"),
        ("BIN_NUMBER", r"0[bB][01]+[uU]?"),
        ("OCT_NUMBER", r"0[oO][0-7]+[uU]?"),
        ("NUMBER", r"\d+[uU]?"),
        ("STRING_LITERAL", r'"(?:[^"\\]|\\.)*"'),
        ("CHAR_LITERAL", r"'(?:[^'\\]|\\.)'"),
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
        ("SPACESHIP", r"<=>"),
        ("EQUAL", r"=="),
        ("NOT_EQUAL", r"!="),
        ("LESS_EQUAL", r"<="),
        ("GREATER_EQUAL", r">="),
        # Operators - Logical
        ("LOGICAL_AND", r"&&"),
        ("LOGICAL_OR", r"\|\|"),
        ("NOT", r"!"),
        ("BITWISE_SHIFT_LEFT", r"<<"),
        ("BITWISE_SHIFT_RIGHT", r">>"),
        ("BITWISE_AND", r"&"),
        ("BITWISE_OR", r"\|"),
        ("BITWISE_XOR", r"\^"),
        ("BITWISE_NOT", r"~"),
        ("INCREMENT", r"\+\+"),
        ("DECREMENT", r"--"),
        ("ARROW", r"->"),
        ("POWER", r"\*\*"),
        ("PLUS", r"\+"),
        ("MINUS", r"-"),
        ("MULTIPLY", r"\*"),
        ("DIVIDE", r"/"),
        ("MOD", r"%"),
        ("FAT_ARROW", r"=>"),
        ("DOUBLE_COLON", r"::"),
        ("RANGE_INCLUSIVE", r"\.\.="),
        ("RANGE", r"\.\."),
        ("ELVIS", r"\?:"),
        ("QUESTION", r"\?"),
        ("PIPE", r"\|"),
        ("SEMICOLON", r";"),
        ("COMMA", r","),
        ("DOT", r"\."),
        ("COLON", r":"),
        ("EQUALS", r"="),
        ("LBRACE", r"\{"),
        ("RBRACE", r"\}"),
        ("LPAREN", r"\("),
        ("RPAREN", r"\)"),
        ("LBRACKET", r"\["),
        ("RBRACKET", r"\]"),
        ("LESS_THAN", r"<"),
        ("GREATER_THAN", r">"),
        ("AT", r"@"),
        ("AMPERSAND", r"&"),
        ("IDENTIFIER", r"[a-zA-Z_][a-zA-Z_0-9]*"),
        ("WHITESPACE", r"\s+"),
    ]
)

SKIP_TOKENS = {"WHITESPACE", "COMMENT_SINGLE", "COMMENT_MULTI"}
IDENTIFIER_RE = re.compile(r"\b[A-Za-z_][A-Za-z_0-9]*\b")
PREPROCESSOR_DIRECTIVE_RE = re.compile(r"^\s*#\s*(?P<name>[A-Za-z_]\w*)\b(?P<body>.*)$")
PREPROCESSOR_DEFINE_RE = re.compile(r"^\s*(?P<name>[A-Za-z_]\w*)(?P<body>.*)$")
PREPROCESSOR_INCLUDE_RE = re.compile(
    r'^(?P<target>"(?:[^"\\]|\\.)*"|<[^>\n]+>|[^\s/][^\s]*)'
)
PREPROCESSOR_STRING_OR_COMMENT_RE = re.compile(
    r'"(?:[^"\\]|\\.)*"|' r"'(?:[^'\\]|\\.)'|" r"//.*"
)
PREPROCESSOR_STRING_RE = re.compile(r'"(?:[^"\\]|\\.)*"|' r"'(?:[^'\\]|\\.)'")
SAFE_PREPROCESSOR_AST_NODES = (
    ast.Expression,
    ast.BoolOp,
    ast.UnaryOp,
    ast.BinOp,
    ast.Compare,
    ast.Constant,
    ast.And,
    ast.Or,
    ast.Not,
    ast.USub,
    ast.UAdd,
    ast.Invert,
    ast.Add,
    ast.Sub,
    ast.Mult,
    ast.Div,
    ast.FloorDiv,
    ast.Mod,
    ast.BitAnd,
    ast.BitOr,
    ast.BitXor,
    ast.LShift,
    ast.RShift,
    ast.Eq,
    ast.NotEq,
    ast.Lt,
    ast.LtE,
    ast.Gt,
    ast.GtE,
)


def _normalized_preprocessor_defines(defines):
    macros = {}
    for name, value in dict(defines or {}).items():
        name = str(name).strip()
        if not IDENTIFIER_RE.fullmatch(name):
            continue
        value = str(value).strip() if value is not None else "1"
        macros[name] = value if value else "1"
    return macros


def _preprocessor_expression_value(value):
    value = str(value).strip()
    if not value:
        return "1"
    lowered = value.lower()
    if lowered == "true":
        return "1"
    if lowered == "false":
        return "0"
    if re.fullmatch(r"[+-]?(?:0[xX][0-9a-fA-F]+|0[bB][01]+|0[oO][0-7]+|\d+)", value):
        return value
    if re.fullmatch(r"[+-]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][+-]?\d+)?", value):
        return value
    if PREPROCESSOR_STRING_RE.fullmatch(value):
        return value
    return "1"


def _replace_identifiers_outside_strings(expression, replacer):
    parts = []
    start = 0
    for match in PREPROCESSOR_STRING_RE.finditer(expression):
        parts.append(IDENTIFIER_RE.sub(replacer, expression[start : match.start()]))
        parts.append(match.group(0))
        start = match.end()
    parts.append(IDENTIFIER_RE.sub(replacer, expression[start:]))
    return "".join(parts)


def _safe_eval_preprocessor_expression(expression):
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as exc:
        raise SyntaxError(f"Unsupported preprocessor expression: {expression}") from exc

    for node in ast.walk(tree):
        if not isinstance(node, SAFE_PREPROCESSOR_AST_NODES):
            raise SyntaxError(f"Unsupported preprocessor expression: {expression}")
        if isinstance(node, ast.Constant) and not isinstance(
            node.value, (bool, int, float, str)
        ):
            raise SyntaxError(f"Unsupported preprocessor expression: {expression}")
    return bool(
        eval(compile(tree, "<crosstl-preprocessor>", "eval"), {"__builtins__": {}}, {})
    )


def _evaluate_preprocessor_condition(expression, macros):
    expression = expression.strip()
    if not expression:
        return False

    def replace_defined_call(match):
        return "1" if match.group(1) in macros else "0"

    def replace_defined_name(match):
        return "1" if match.group(1) in macros else "0"

    expression = re.sub(
        r"\bdefined\s*\(\s*([A-Za-z_]\w*)\s*\)",
        replace_defined_call,
        expression,
    )
    expression = re.sub(
        r"\bdefined\s+([A-Za-z_]\w*)",
        replace_defined_name,
        expression,
    )
    expression = expression.replace("&&", " and ").replace("||", " or ")
    expression = re.sub(r"!(?!=)", " not ", expression)

    def replace_identifier(match):
        name = match.group(0)
        if name in {"and", "or", "not"}:
            return name
        lowered = name.lower()
        if lowered == "true":
            return "1"
        if lowered == "false":
            return "0"
        if name in macros:
            return _preprocessor_expression_value(macros[name])
        return "0"

    expression = _replace_identifiers_outside_strings(expression, replace_identifier)
    return _safe_eval_preprocessor_expression(expression)


def _expand_object_macros(line, macros):
    if not macros:
        return line

    def replace_identifier(match):
        name = match.group(0)
        return str(macros.get(name, name))

    parts = []
    start = 0
    for match in PREPROCESSOR_STRING_OR_COMMENT_RE.finditer(line):
        parts.append(IDENTIFIER_RE.sub(replace_identifier, line[start : match.start()]))
        parts.append(match.group(0))
        start = match.end()
    parts.append(IDENTIFIER_RE.sub(replace_identifier, line[start:]))
    return "".join(parts)


def _active_preprocessor_block(stack):
    return all(frame["active"] for frame in stack)


def _parse_include_target(body, macros):
    expanded = _expand_object_macros(body.strip(), macros).strip()
    match = PREPROCESSOR_INCLUDE_RE.match(expanded)
    if match is None:
        return None

    target = match.group("target")
    if target.startswith('"') and target.endswith('"'):
        return target[1:-1], False
    if target.startswith("<") and target.endswith(">"):
        return target[1:-1], True
    return target, False


def _normalize_include_paths(include_paths):
    normalized = []
    for include_path in include_paths or ():
        try:
            normalized.append(Path(include_path).expanduser().resolve())
        except OSError:
            normalized.append(Path(include_path).expanduser())
    return normalized


def _resolve_include_target(target, system_include, current_file, include_paths):
    candidate = Path(target)
    if candidate.is_absolute():
        return candidate.resolve() if candidate.is_file() else None

    search_dirs = []
    if not system_include and current_file is not None:
        search_dirs.append(current_file.parent)
    search_dirs.extend(include_paths)

    for search_dir in search_dirs:
        candidate = search_dir / target
        if candidate.is_file():
            return candidate.resolve()
    return None


def _preprocess_code_with_defines(code, defines, file_path=None, include_paths=None):
    macros = _normalized_preprocessor_defines(defines)
    normalized_include_paths = _normalize_include_paths(include_paths)
    process_preprocessor_blocks = defines is not None

    def preprocess_source(source_code, current_file=None, include_stack=()):
        output = []
        stack = []
        preserved_conditional_depth = 0

        for line_number, line in enumerate(source_code.splitlines(), start=1):
            directive = PREPROCESSOR_DIRECTIVE_RE.match(line)
            active = _active_preprocessor_block(stack)
            if directive is None:
                if active:
                    output.append(_expand_object_macros(line, macros))
                continue

            name = directive.group("name")
            body = directive.group("body").strip()

            if not process_preprocessor_blocks:
                if name in {"if", "ifdef", "ifndef"}:
                    preserved_conditional_depth += 1
                    output.append(line)
                    continue
                if name == "endif":
                    if preserved_conditional_depth:
                        preserved_conditional_depth -= 1
                    output.append(line)
                    continue
                if name in {"elif", "else"} or preserved_conditional_depth:
                    output.append(line)
                    continue
                if name != "include":
                    output.append(line)
                    continue

            if name in {"if", "ifdef", "ifndef"}:
                parent_active = active
                if name == "ifdef":
                    condition = body in macros
                elif name == "ifndef":
                    condition = body not in macros
                else:
                    condition = (
                        _evaluate_preprocessor_condition(body, macros)
                        if parent_active
                        else False
                    )
                branch_active = parent_active and condition
                stack.append(
                    {
                        "parent_active": parent_active,
                        "active": branch_active,
                        "branch_taken": branch_active,
                        "else_seen": False,
                    }
                )
                continue

            if name == "elif":
                if not stack:
                    raise SyntaxError(f"#elif without #if at line {line_number}")
                frame = stack[-1]
                if frame["else_seen"]:
                    raise SyntaxError(f"#elif after #else at line {line_number}")
                if not frame["parent_active"] or frame["branch_taken"]:
                    frame["active"] = False
                else:
                    branch_active = _evaluate_preprocessor_condition(body, macros)
                    frame["active"] = branch_active
                    frame["branch_taken"] = branch_active
                continue

            if name == "else":
                if not stack:
                    raise SyntaxError(f"#else without #if at line {line_number}")
                frame = stack[-1]
                if frame["else_seen"]:
                    raise SyntaxError(f"duplicate #else at line {line_number}")
                frame["active"] = frame["parent_active"] and not frame["branch_taken"]
                frame["branch_taken"] = frame["branch_taken"] or frame["active"]
                frame["else_seen"] = True
                continue

            if name == "endif":
                if not stack:
                    raise SyntaxError(f"#endif without #if at line {line_number}")
                stack.pop()
                continue

            if not active:
                continue

            if name == "define":
                define = PREPROCESSOR_DEFINE_RE.match(body)
                if define is not None:
                    macro_name = define.group("name")
                    macro_body = define.group("body").strip()
                    if macro_body.startswith("("):
                        output.append(line)
                    else:
                        macros[macro_name] = macro_body or "1"
                continue

            if name == "undef":
                macros.pop(body, None)
                continue

            if name == "include":
                target = _parse_include_target(body, macros)
                if target is None:
                    output.append(line)
                    continue
                include_name, system_include = target
                resolved = _resolve_include_target(
                    include_name,
                    system_include,
                    current_file,
                    normalized_include_paths,
                )
                if resolved is None:
                    output.append(line)
                    continue
                if resolved in include_stack:
                    include_chain = " -> ".join(str(path) for path in include_stack)
                    raise SyntaxError(
                        f"Cyclic #include at line {line_number}: "
                        f"{include_chain} -> {resolved}"
                    )
                included_code = resolved.read_text(encoding="utf-8")
                output.append(
                    preprocess_source(
                        included_code,
                        current_file=resolved,
                        include_stack=(*include_stack, resolved),
                    )
                )
                continue

            output.append(line)

        if stack:
            raise SyntaxError("Unterminated preprocessor conditional")
        return "\n".join(output)

    current_file = None
    if file_path is not None:
        current_file = Path(file_path).expanduser().resolve()
    include_stack = (current_file,) if current_file is not None else ()
    return preprocess_source(
        code, current_file=current_file, include_stack=include_stack
    )


KEYWORDS = {
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
    "fn": "FUNCTION",
    "void": "VOID",
    "return": "RETURN",
    "yield": "YIELD",
    "async": "ASYNC",
    "await": "AWAIT",
    "if": "IF",
    "else": "ELSE",
    "elif": "ELIF",
    "match": "MATCH",
    "switch": "SWITCH",
    "case": "CASE",
    "default": "DEFAULT",
    "for": "FOR",
    "do": "DO",
    "while": "WHILE",
    "loop": "LOOP",
    "in": "IN",
    "break": "BREAK",
    "continue": "CONTINUE",
    "let": "LET",
    "var": "VAR",
    "mut": "MUT",
    "const": "CONST",
    "static": "STATIC",
    "extern": "EXTERN",
    "uniform": "UNIFORM",
    "cbuffer": "CBUFFER",
    "buffer": "BUFFER",
    "precision": "PRECISION",
    "pub": "PUBLIC",
    "priv": "PRIVATE",
    "protected": "PROTECTED",
    "internal": "INTERNAL",
    "unsafe": "UNSAFE",
    "safe": "SAFE",
    "ref": "REF",
    "box": "BOX",
    "move": "MOVE",
    "vertex": "VERTEX",
    "fragment": "FRAGMENT",
    "compute": "COMPUTE",
    "geometry": "GEOMETRY",
    "tessellation": "TESSELLATION",
    "tessellation_control": "TESSELLATION_CONTROL",
    "tessellation_evaluation": "TESSELLATION_EVALUATION",
    "hull": "TESSELLATION_CONTROL",
    "domain": "TESSELLATION_EVALUATION",
    "task": "TASK",
    "amplification": "AMPLIFICATION",
    "object": "OBJECT",
    "mesh": "MESH",
    "ray_generation": "RAY_GENERATION",
    "ray_intersection": "RAY_INTERSECTION",
    "ray_closest_hit": "RAY_CLOSEST_HIT",
    "ray_miss": "RAY_MISS",
    "ray_any_hit": "RAY_ANY_HIT",
    "ray_callable": "RAY_CALLABLE",
    "intersection": "RAY_INTERSECTION",
    "anyhit": "RAY_ANY_HIT",
    "closesthit": "RAY_CLOSEST_HIT",
    "miss": "RAY_MISS",
    "callable": "RAY_CALLABLE",
    "kernel": "KERNEL",
    "global": "GLOBAL",
    "local": "LOCAL",
    "shared": "SHARED",
    "threadgroup_imageblock": "THREADGROUP_IMAGEBLOCK",
    "threadgroup": "THREADGROUP",
    "workgroup": "WORKGROUP",
    "layout": "LAYOUT",
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
    "vec2": "VEC2",
    "vec3": "VEC3",
    "vec4": "VEC4",
    "ivec2": "IVEC2",
    "ivec3": "IVEC3",
    "ivec4": "IVEC4",
    "uvec2": "UVEC2",
    "uvec3": "UVEC3",
    "uvec4": "UVEC4",
    "dvec2": "DVEC2",
    "dvec3": "DVEC3",
    "dvec4": "DVEC4",
    "bvec2": "BVEC2",
    "bvec3": "BVEC3",
    "bvec4": "BVEC4",
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
    "dmat2": "DMAT2",
    "dmat3": "DMAT3",
    "dmat4": "DMAT4",
    "dmat2x2": "DMAT2X2",
    "dmat2x3": "DMAT2X3",
    "dmat2x4": "DMAT2X4",
    "dmat3x2": "DMAT3X2",
    "dmat3x3": "DMAT3X3",
    "dmat3x4": "DMAT3X4",
    "dmat4x2": "DMAT4X2",
    "dmat4x3": "DMAT4X3",
    "dmat4x4": "DMAT4X4",
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
    "sampler2dshadow": "SAMPLER2DSHADOW",
    "sampler2darrayshadow": "SAMPLER2DARRAYSHADOW",
    "samplercubeshadow": "SAMPLERCUBESHADOW",
    "samplercubearray": "SAMPLERCUBEARRAY",
    "samplercubearrayshadow": "SAMPLERCUBEARRAYSHADOW",
    "sampler2dms": "SAMPLER2DMS",
    "sampler2dmsarray": "SAMPLER2DMSARRAY",
    "iimage2d": "IIMAGE2D",
    "iimage3d": "IIMAGE3D",
    "iimagecube": "IIMAGECUBE",
    "iimagecubearray": "IIMAGECUBEARRAY",
    "iimage2darray": "IIMAGE2DARRAY",
    "iimage2dms": "IIMAGE2DMS",
    "iimage2dmsarray": "IIMAGE2DMSARRAY",
    "uimage2d": "UIMAGE2D",
    "uimage3d": "UIMAGE3D",
    "uimagecube": "UIMAGECUBE",
    "uimagecubearray": "UIMAGECUBEARRAY",
    "uimage2darray": "UIMAGE2DARRAY",
    "uimage2dms": "UIMAGE2DMS",
    "uimage2dmsarray": "UIMAGE2DMSARRAY",
    "image2d": "IMAGE2D",
    "image3d": "IMAGE3D",
    "imagecube": "IMAGECUBE",
    "imagecubearray": "IMAGECUBEARRAY",
    "image2darray": "IMAGE2DARRAY",
    "image2dms": "IMAGE2DMS",
    "image2dmsarray": "IMAGE2DMSARRAY",
    "where": "WHERE",
    "true": "BOOLEAN_LITERAL",
    "false": "BOOLEAN_LITERAL",
}


class Lexer:
    """Tokenizer for CrossGL Universal IR."""

    def __init__(self, code, file_path=None, include_paths=None, defines=None):
        self.code = self._splice_line_continuations(code.lstrip("\ufeff"))
        if defines is not None or file_path is not None or include_paths:
            self.code = _preprocess_code_with_defines(
                self.code,
                defines,
                file_path=file_path,
                include_paths=include_paths,
            )
        self.tokens = []
        self.token_cache = {}
        self.regex_cache = self._compile_patterns()
        self.tokenize()

    @staticmethod
    def _splice_line_continuations(code):
        return re.sub(r"\\(?:\r\n|\n|\r)", "", code)

    def _compile_patterns(self):
        combined_pattern = "|".join(
            f"(?P<{name}>{pattern})" for name, pattern in TOKENS.items()
        )
        return re.compile(combined_pattern)

    def _get_cached_token(self, text, token_type):
        """Return a stable tuple object for repeated token text/type pairs."""
        cache_key = (text, token_type)
        if cache_key not in self.token_cache:
            self.token_cache[cache_key] = (token_type, text)
        return self.token_cache[cache_key]

    def tokenize(self):
        pos = 0
        length = len(self.code)

        while pos < length:
            match = self.regex_cache.match(self.code, pos)
            if match:
                token_type = match.lastgroup
                text = match.group(token_type)

                if token_type == "IDENTIFIER" and text in KEYWORDS:
                    token_type = KEYWORDS[text]

                if token_type not in SKIP_TOKENS:
                    token = self._get_cached_token(text, token_type)
                    self.tokens.append(token)

                pos = match.end(0)
            else:
                bad_char = self.code[pos]
                line_num = self.code[:pos].count("\n") + 1
                col_num = pos - self.code.rfind("\n", 0, pos)

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

        self.tokens.append(self._get_cached_token(None, "EOF"))

    def get_tokens(self):
        return self.tokens

    def debug_print(self):
        """Print token indexes, types, and text for grammar debugging."""
        for i, (token_type, text) in enumerate(self.tokens):
            print(f"{i:3d}: {token_type:20s} '{text}'")
