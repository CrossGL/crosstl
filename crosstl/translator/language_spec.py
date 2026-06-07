"""Machine-readable CrossTL frontend language spec extraction."""

from __future__ import annotations

import argparse
import ast as py_ast
import enum
import hashlib
import inspect
import json
import sys
import textwrap
from pathlib import Path
from types import ModuleType
from typing import Any

from . import ast as ast_module
from . import lexer as lexer_module
from . import parser as parser_module
from . import stage_utils
from . import validation as validation_module

SPEC_SCHEMA_VERSION = 0
SPEC_KIND = "crosstl-frontend-language-spec-snapshot"

PRIMITIVE_TYPE_TOKENS = (
    "BOOL",
    "I8",
    "I16",
    "I32",
    "I64",
    "U8",
    "U16",
    "U32",
    "U64",
    "F16",
    "F32",
    "F64",
    "INT",
    "UINT",
    "FLOAT",
    "DOUBLE",
    "HALF",
    "CHAR",
    "STRING",
    "VOID",
)

VECTOR_TYPE_TOKENS = (
    "VEC2",
    "VEC3",
    "VEC4",
    "IVEC2",
    "IVEC3",
    "IVEC4",
    "UVEC2",
    "UVEC3",
    "UVEC4",
    "DVEC2",
    "DVEC3",
    "DVEC4",
    "BVEC2",
    "BVEC3",
    "BVEC4",
)

MATRIX_TYPE_PREFIXES = ("MAT", "DMAT")

PARSER_QUALIFIER_CONSTANTS = (
    "VARIABLE_QUALIFIER_TOKEN_TYPES",
    "PARAMETER_QUALIFIER_TOKEN_TYPES",
    "VARIABLE_QUALIFIER_NAMES",
    "PARAMETER_PRIMITIVE_QUALIFIER_NAMES",
    "SHADER_STAGE_TOKEN_TYPES",
)

PARSER_INTRINSIC_CONSTANTS = (
    "WAVE_INTRINSICS",
    "RAYTRACING_INTRINSICS",
    "MESH_INTRINSICS",
    "RAYQUERY_METHODS",
)

PARSER_RESOURCE_CONSTANTS = ("TEXTURE_TYPE_NAMES",)

VALIDATION_METADATA_CONSTANTS = (
    "SINGLE_VALUE_METADATA_NAMES",
    "SINGLE_VALUE_METADATA_ALIASES",
    "MULTI_VALUE_METADATA_NAMES",
    "HLSL_SEMANTIC_METADATA_BASE_NAMES",
    "RESOURCE_ACCESS_METADATA_NAMES",
    "DESCRIPTOR_INDEX_METADATA_NAMES",
    "IMAGE_FORMAT_METADATA_NAMES",
    "ADDRESS_SPACE_METADATA_NAMES",
    "MEMORY_LAYOUT_METADATA_NAMES",
    "DECLARATION_ROLE_METADATA_NAMES",
    "STRUCT_DECLARATION_ROLE_NAMES",
    "PARAMETER_DECLARATION_ROLE_NAMES",
    "BUILTIN_SEMANTIC_METADATA_NAMES",
    "FUNCTION_STAGE_ATTRIBUTE_NAMES",
)

VALIDATION_RULE_CONSTANTS = (
    "TEXTURE_INTRINSIC_MIN_ARGUMENTS",
    "TEXTURE_INTRINSIC_MAX_ARGUMENTS",
    "TEXTURE_INTRINSIC_ALLOWED_ARGUMENT_COUNTS",
    "TEXTURE_INTRINSICS_WITH_EXPLICIT_SAMPLERS",
    "METADATA_CONFLICT_GROUPS",
    "INTERPOLATION_MODE_METADATA_NAMES",
    "INTERPOLATION_SAMPLING_METADATA_NAMES",
    "STAGE_LAYOUT_DIRECTION_REQUIREMENTS",
    "STAGE_LAYOUT_EXCLUSIVE_ENTRY_GROUPS",
    "FUNCTION_STAGE_LAYOUT_FLAG_ALIASES",
    "FUNCTION_STAGE_OUTPUT_TOPOLOGY_ALIASES",
    "FUNCTION_STAGE_LAYOUT_VALUE_ENTRIES",
    "TESSELLATION_CONTROL_STAGE_LAYOUT_ENTRIES",
    "TESSELLATION_EVALUATION_STAGE_LAYOUT_ENTRIES",
    "TESSELLATION_CONTROL_FUNCTION_ATTRIBUTE_NAMES",
    "TESSELLATION_STAGE_FUNCTION_ATTRIBUTE_NAMES",
    "TESSELLATION_EVALUATION_FUNCTION_FLAG_NAMES",
    "IMAGE_RESOURCE_INTRINSIC_NAMES",
    "INTEGER_COORDINATE_INTRINSIC_NAMES",
    "OFFSET_DIMENSION_INTRINSIC_NAMES",
    "OFFSET_ARGUMENT_INDEX_OFFSETS",
    "GRADIENT_DIMENSION_INTRINSIC_NAMES",
    "GRADIENT_ARGUMENT_INDEX_OFFSETS",
    "COMPARE_INTRINSIC_NAMES",
    "LOD_ARGUMENT_INDEX_OFFSETS",
    "BIAS_ARGUMENT_INDEX_OFFSETS",
    "MIP_LEVEL_ARGUMENT_INDICES",
    "GATHER_COMPONENT_INTRINSIC_NAMES",
)

VALIDATION_RESOURCE_CONSTANTS = (
    "STORAGE_IMAGE_TYPE_NAMES",
    "RESOURCE_BUFFER_TYPE_NAMES",
    "UAV_RESOURCE_BUFFER_TYPE_NAMES",
    "SAMPLER_STATE_TYPE_NAMES",
)

AST_CATEGORY_ROOTS = (
    ("type", "TypeNode"),
    ("statement", "StatementNode"),
    ("expression", "ExpressionNode"),
    ("pattern", "PatternNode"),
)


class LanguageSpecExtractionError(RuntimeError):
    """Raised when bounded source introspection cannot extract a known shape."""


def repository_root() -> Path:
    """Return the repository root containing this translator package."""
    return Path(__file__).resolve().parents[2]


def _relative_module_path(module: ModuleType) -> str:
    path = Path(module.__file__).resolve()
    try:
        return path.relative_to(repository_root()).as_posix()
    except ValueError:
        return path.as_posix()


def _source_hash(module: ModuleType) -> str:
    path = Path(module.__file__).resolve()
    text = path.read_text(encoding="utf-8").replace("\r\n", "\n")
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _source_inventory() -> list[dict[str, str]]:
    modules = (lexer_module, parser_module, ast_module, validation_module)
    return [
        {
            "path": _relative_module_path(module),
            "sha256": _source_hash(module),
        }
        for module in modules
    ]


def _json_sort_key(value: Any) -> str:
    return json.dumps(_jsonable(value), sort_keys=True, separators=(",", ":"))


def _jsonable(value: Any) -> Any:
    if isinstance(value, enum.Enum):
        return value.value
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, dict):
        return {
            str(key): _jsonable(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (set, frozenset)):
        return [_jsonable(item) for item in sorted(value, key=_json_sort_key)]
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    return str(value)


def _extract_constants(module: ModuleType, names: tuple[str, ...]) -> dict[str, Any]:
    return {name: _jsonable(getattr(module, name)) for name in names}


def _lexer_spec() -> dict[str, Any]:
    keyword_tokens: dict[str, list[str]] = {}
    for spelling, token_name in lexer_module.KEYWORDS.items():
        keyword_tokens.setdefault(token_name, []).append(spelling)

    return {
        "tokens": [
            {"name": name, "pattern": pattern}
            for name, pattern in lexer_module.TOKENS.items()
        ],
        "keywords": _jsonable(lexer_module.KEYWORDS),
        "keyword_tokens": {
            token_name: sorted(spellings)
            for token_name, spellings in sorted(keyword_tokens.items())
        },
        "skip_tokens": _jsonable(lexer_module.SKIP_TOKENS),
    }


def _token_spellings_by_name(keywords: dict[str, str]) -> dict[str, list[str]]:
    spellings: dict[str, list[str]] = {}
    for spelling, token in keywords.items():
        spellings.setdefault(token, []).append(spelling)
    return {token: sorted(values) for token, values in spellings.items()}


def _preferred_spelling(token: str, spellings: dict[str, list[str]]) -> str:
    values = spellings.get(token)
    return values[0] if values else token.lower()


def _canonical_stage_from_token(token: str, canonical_stages: set[str]) -> str | None:
    manual_aliases = {"KERNEL": "compute"}
    if token in manual_aliases:
        return manual_aliases[token]
    candidate = token.lower()
    return candidate if candidate in canonical_stages else None


def _vector_element_type(token: str) -> str:
    if token.startswith("IVEC"):
        return "int"
    if token.startswith("UVEC"):
        return "uint"
    if token.startswith("DVEC"):
        return "double"
    if token.startswith("BVEC"):
        return "bool"
    return "float"


def _vector_width(spelling: str) -> int:
    return int(spelling[-1])


def _matrix_shape(spelling: str) -> tuple[int, int]:
    is_double = spelling.startswith("dmat")
    dimensions = spelling[4:] if is_double else spelling[3:]
    if "x" in dimensions:
        rows, columns = dimensions.split("x", 1)
        return int(rows), int(columns)
    size = int(dimensions)
    return size, size


def _is_matrix_token(token: str, spellings: dict[str, list[str]]) -> bool:
    spelling = _preferred_spelling(token, spellings)
    return spelling.startswith(("mat", "dmat")) and spelling[-1].isdigit()


def _class_descendants(classes: list[dict[str, Any]], root_name: str) -> list[str]:
    bases_by_name = {item["name"]: set(item["bases"]) for item in classes}

    def inherits(name: str, seen: set[str] | None = None) -> bool:
        if name == root_name:
            return False
        if seen is None:
            seen = set()
        if name in seen:
            return False
        seen.add(name)
        bases = bases_by_name.get(name, set())
        return root_name in bases or any(inherits(base, seen) for base in bases)

    return sorted(name for name in bases_by_name if inherits(name))


def _sorted_mapping(
    mapping: dict[str, str], key_name: str = "spelling", value_name: str = "canonical"
) -> list[dict[str, str]]:
    return [
        {key_name: str(key), value_name: str(value)}
        for key, value in sorted(mapping.items(), key=lambda item: str(item[0]))
    ]


def _ast_snapshot(ast_spec: dict[str, Any]) -> dict[str, Any]:
    classes = [
        {"name": node["name"], "bases": node["bases"]} for node in ast_spec["nodes"]
    ]
    class_fields = []
    for node in ast_spec["nodes"]:
        class_fields.append(
            {
                "class": node["name"],
                "constructorDefined": node["constructor_defined"],
                "constructorParameters": node["constructor_parameters"],
                "fields": node["fields"],
            }
        )

    return {
        "classes": classes,
        "classFields": class_fields,
        "typeNodes": _class_descendants(classes, "TypeNode"),
        "statementNodes": _class_descendants(classes, "StatementNode"),
        "expressionNodes": _class_descendants(classes, "ExpressionNode"),
        "enums": ast_spec["enums"],
    }


def _classify_sampler_image(canonical_name: str) -> str:
    lowered = canonical_name.lower()
    if lowered.startswith(("image", "iimage", "uimage")):
        return "storage-image"
    if lowered.startswith("sampler") or lowered == "comparison_sampler":
        return "sampler"
    return "resource"


def _texture_intrinsic_entries(
    validation_rules: dict[str, Any],
) -> list[dict[str, Any]]:
    minimums = validation_rules["TEXTURE_INTRINSIC_MIN_ARGUMENTS"]
    maximums = validation_rules["TEXTURE_INTRINSIC_MAX_ARGUMENTS"]
    allowed = validation_rules["TEXTURE_INTRINSIC_ALLOWED_ARGUMENT_COUNTS"]
    explicit = set(validation_rules["TEXTURE_INTRINSICS_WITH_EXPLICIT_SAMPLERS"])
    names = sorted(set(minimums) | set(maximums) | set(allowed) | explicit)
    entries = []
    for name in names:
        entry = {"name": name, "explicitSampler": name in explicit}
        if name in minimums:
            entry["minArguments"] = minimums[name]
        if name in maximums:
            entry["maxArguments"] = maximums[name]
        if name in allowed:
            entry["allowedArgumentCounts"] = allowed[name]
        entries.append(entry)
    return entries


def _frontend_snapshot_from_raw(raw: dict[str, Any]) -> dict[str, Any]:
    lexer = raw["lexer"]
    parser = raw["parser"]
    validation = raw["validation"]
    spellings = _token_spellings_by_name(lexer["keywords"])
    stage_tokens = set(parser["qualifiers"]["SHADER_STAGE_TOKEN_TYPES"])
    canonical_stages = set(raw["stages"]["canonical"])
    texture_type_names = parser["resources"]["TEXTURE_TYPE_NAMES"]
    sampler_image_names = parser["resources"]["SAMPLER_IMAGE_TYPE_NAMES"]

    matrix_types = []
    for token in sorted(
        token for token in spellings if token.startswith(MATRIX_TYPE_PREFIXES)
    ):
        if not _is_matrix_token(token, spellings):
            continue
        spelling = _preferred_spelling(token, spellings)
        rows, columns = _matrix_shape(spelling)
        matrix_types.append(
            {
                "spelling": spelling,
                "token": token,
                "elementType": "double" if spelling.startswith("dmat") else "float",
                "rows": rows,
                "columns": columns,
            }
        )

    stage_related_tokens = stage_tokens | {"KERNEL", "TESSELLATION"}
    return {
        "schemaVersion": SPEC_SCHEMA_VERSION,
        "kind": SPEC_KIND,
        "description": (
            "CrossTL frontend language surface snapshot for CrossGL-Compiler "
            "shared language spec integration."
        ),
        "source": {
            "repository": "CrossGL-Translator",
            "frontend": "crosstl.translator",
            "files": raw["source"]["files"],
            "extraction": {
                "tool": "python -m crosstl.translator.language_spec",
                "method": raw["source"]["method"],
            },
        },
        "lexical": {
            "tokens": lexer["tokens"],
            "keywords": [
                {"spelling": spelling, "token": token}
                for spelling, token in sorted(lexer["keywords"].items())
            ],
            "skipTokens": lexer["skip_tokens"],
            "literalTokens": [
                item["name"]
                for item in lexer["tokens"]
                if item["name"].endswith("_LITERAL")
                or item["name"].endswith("_NUMBER")
                or item["name"] == "NUMBER"
            ],
        },
        "language": {
            "stages": {
                "canonical": raw["stages"]["canonical"],
                "keywordSpellings": [
                    {
                        "spelling": spelling,
                        "token": token,
                        "canonical": _canonical_stage_from_token(
                            token, canonical_stages
                        ),
                        "acceptedAsStageBlock": token in stage_tokens,
                    }
                    for spelling, token in sorted(lexer["keywords"].items())
                    if token in stage_related_tokens
                ],
                "parserStageTokens": sorted(stage_tokens),
            },
            "types": {
                "primitive": [
                    {"spelling": _preferred_spelling(token, spellings), "token": token}
                    for token in PRIMITIVE_TYPE_TOKENS
                ],
                "vectors": [
                    {
                        "spelling": _preferred_spelling(token, spellings),
                        "token": token,
                        "elementType": _vector_element_type(token),
                        "width": _vector_width(_preferred_spelling(token, spellings)),
                    }
                    for token in VECTOR_TYPE_TOKENS
                ],
                "matrices": matrix_types,
                "textures": [
                    {
                        "token": token,
                        "spelling": _preferred_spelling(token, spellings),
                        "canonical": canonical,
                    }
                    for token, canonical in sorted(texture_type_names.items())
                ],
                "samplersAndImages": [
                    {
                        "token": token,
                        "canonical": canonical,
                        "kind": _classify_sampler_image(canonical),
                        "keywordSpellings": spellings.get(token, []),
                    }
                    for token, canonical in sorted(sampler_image_names.items())
                ],
                "namedTypeFallback": True,
                "arrayForms": [
                    "[element_type optional_size]",
                    "base_type[optional_size]",
                ],
                "postfixTypeOperators": [
                    {"operator": "*", "node": "PointerType"},
                    {"operator": "&", "node": "ReferenceType"},
                    {"operator": "& mut", "node": "ReferenceType", "mutable": True},
                ],
            },
            "qualifiers": {
                "variableQualifierTokens": parser["qualifiers"][
                    "VARIABLE_QUALIFIER_TOKEN_TYPES"
                ],
                "parameterQualifierTokens": parser["qualifiers"][
                    "PARAMETER_QUALIFIER_TOKEN_TYPES"
                ],
                "variableQualifierNames": parser["qualifiers"][
                    "VARIABLE_QUALIFIER_NAMES"
                ],
                "parameterPrimitiveQualifierNames": parser["qualifiers"][
                    "PARAMETER_PRIMITIVE_QUALIFIER_NAMES"
                ],
            },
            "resources": {
                "storageImageTypeNames": validation["resources"][
                    "STORAGE_IMAGE_TYPE_NAMES"
                ],
                "resourceBufferTypeNames": validation["resources"][
                    "RESOURCE_BUFFER_TYPE_NAMES"
                ],
                "uavResourceBufferTypeNames": validation["resources"][
                    "UAV_RESOURCE_BUFFER_TYPE_NAMES"
                ],
                "samplerStateTypeNames": validation["resources"][
                    "SAMPLER_STATE_TYPE_NAMES"
                ],
                "resourceAccessMetadata": _sorted_mapping(
                    validation["metadata"]["RESOURCE_ACCESS_METADATA_NAMES"]
                ),
                "descriptorIndexMetadata": _sorted_mapping(
                    validation["metadata"]["DESCRIPTOR_INDEX_METADATA_NAMES"],
                    value_name="role",
                ),
                "imageFormatMetadataNames": validation["metadata"][
                    "IMAGE_FORMAT_METADATA_NAMES"
                ],
                "addressSpaceMetadata": _sorted_mapping(
                    validation["metadata"]["ADDRESS_SPACE_METADATA_NAMES"]
                ),
                "memoryLayoutMetadata": _sorted_mapping(
                    validation["metadata"]["MEMORY_LAYOUT_METADATA_NAMES"]
                ),
                "builtinSemanticMetadata": _sorted_mapping(
                    validation["metadata"]["BUILTIN_SEMANTIC_METADATA_NAMES"]
                ),
            },
            "intrinsics": {
                "textureAndImage": _texture_intrinsic_entries(validation["rules"]),
                "imageResource": validation["rules"]["IMAGE_RESOURCE_INTRINSIC_NAMES"],
                "integerCoordinate": validation["rules"][
                    "INTEGER_COORDINATE_INTRINSIC_NAMES"
                ],
                "wave": parser["intrinsics"]["WAVE_INTRINSICS"],
                "rayTracing": parser["intrinsics"]["RAYTRACING_INTRINSICS"],
                "rayQueryMethods": parser["intrinsics"]["RAYQUERY_METHODS"],
                "mesh": parser["intrinsics"]["MESH_INTRINSICS"],
            },
        },
        "ast": _ast_snapshot(raw["ast"]),
        "validation": {
            "metadata": {
                "singleValueNames": validation["metadata"][
                    "SINGLE_VALUE_METADATA_NAMES"
                ],
                "singleValueAliases": _sorted_mapping(
                    validation["metadata"]["SINGLE_VALUE_METADATA_ALIASES"]
                ),
                "multiValueNames": validation["metadata"]["MULTI_VALUE_METADATA_NAMES"],
                "interpolationModes": _sorted_mapping(
                    validation["rules"]["INTERPOLATION_MODE_METADATA_NAMES"]
                ),
                "interpolationSampling": _sorted_mapping(
                    validation["rules"]["INTERPOLATION_SAMPLING_METADATA_NAMES"]
                ),
                "hlslSemanticBaseNames": validation["metadata"][
                    "HLSL_SEMANTIC_METADATA_BASE_NAMES"
                ],
            },
            "stageLayout": {
                "directionRequirements": _sorted_mapping(
                    validation["rules"]["STAGE_LAYOUT_DIRECTION_REQUIREMENTS"],
                    key_name="entry",
                    value_name="requiredDirection",
                ),
                "exclusiveEntryGroups": validation["rules"][
                    "STAGE_LAYOUT_EXCLUSIVE_ENTRY_GROUPS"
                ],
                "tessellationControlEntries": validation["rules"][
                    "TESSELLATION_CONTROL_STAGE_LAYOUT_ENTRIES"
                ],
                "tessellationEvaluationEntries": validation["rules"][
                    "TESSELLATION_EVALUATION_STAGE_LAYOUT_ENTRIES"
                ],
            },
        },
        "notes": [
            "This v0 artifact snapshots CrossTL lexical/type/resource/stage facts.",
            "It is not yet a full grammar or semantic spec.",
            "Compiler-only strictness must be represented as later shared spec deltas.",
        ],
    }


def _stage_spec() -> dict[str, Any]:
    return {
        "canonical": [stage.value for stage in ast_module.ShaderStage],
        "aliases": _jsonable(stage_utils.STAGE_NAME_ALIASES),
        "stage_names": _jsonable(stage_utils.SHADER_STAGE_NAMES),
        "qualifier_names": _jsonable(stage_utils.STAGE_QUALIFIER_NAMES),
    }


def _literal_value(node: py_ast.AST) -> Any:
    try:
        return py_ast.literal_eval(node)
    except (TypeError, ValueError) as exc:
        raise LanguageSpecExtractionError(
            f"unsupported parser literal: {py_ast.dump(node)}"
        ) from exc


def _extract_parse_type_sampler_map() -> dict[str, str]:
    """Extract Parser.parse_type's local sampler/image resource table."""
    source = textwrap.dedent(inspect.getsource(parser_module.Parser.parse_type))
    parsed = py_ast.parse(source)
    assignments = [
        node
        for node in py_ast.walk(parsed)
        if isinstance(node, py_ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], py_ast.Name)
        and node.targets[0].id == "sampler_types"
    ]
    if len(assignments) != 1:
        raise LanguageSpecExtractionError(
            "Parser.parse_type must contain exactly one sampler_types assignment"
        )

    sampler_types = _literal_value(assignments[0].value)
    if not isinstance(sampler_types, dict) or not sampler_types:
        raise LanguageSpecExtractionError("Parser.parse_type sampler_types is empty")
    return {str(key): str(value) for key, value in sorted(sampler_types.items())}


def _parser_spec() -> dict[str, Any]:
    resources = _extract_constants(parser_module, PARSER_RESOURCE_CONSTANTS)
    resources["SAMPLER_IMAGE_TYPE_NAMES"] = _jsonable(_extract_parse_type_sampler_map())

    return {
        "qualifiers": _extract_constants(parser_module, PARSER_QUALIFIER_CONSTANTS),
        "intrinsics": _extract_constants(parser_module, PARSER_INTRINSIC_CONSTANTS),
        "resources": resources,
    }


def _source_segment(source_text: str, node: py_ast.AST | None) -> str | None:
    if node is None:
        return None
    return py_ast.get_source_segment(source_text, node)


def _annotation_is_optional(text: str | None) -> bool:
    if text is None:
        return False
    return (
        "Optional[" in text
        or "None" in text
        or text.endswith(" | None")
        or " | None | " in text
    )


def _parameter_entry(
    arg: py_ast.arg, kind: str, default_node: py_ast.AST | None, source_text: str
) -> dict[str, Any]:
    annotation = _source_segment(source_text, arg.annotation)
    return {
        "name": arg.arg,
        "kind": kind,
        "annotation": annotation,
        "required": default_node is None,
        "default": _source_segment(source_text, default_node),
        "optional": default_node is not None or _annotation_is_optional(annotation),
    }


def _constructor_parameters(
    function: py_ast.FunctionDef, source_text: str
) -> list[dict[str, Any]]:
    parameters = []
    positional = list(function.args.posonlyargs) + list(function.args.args)
    defaults: list[py_ast.AST | None] = [None] * (
        len(positional) - len(function.args.defaults)
    )
    defaults.extend(function.args.defaults)
    for arg, default_node in zip(positional, defaults):
        if arg.arg == "self":
            continue
        kind = (
            "positional-only"
            if arg in function.args.posonlyargs
            else "positional-or-keyword"
        )
        parameters.append(_parameter_entry(arg, kind, default_node, source_text))

    for arg, default_node in zip(function.args.kwonlyargs, function.args.kw_defaults):
        parameters.append(
            _parameter_entry(arg, "keyword-only", default_node, source_text)
        )

    if function.args.vararg is not None:
        entry = _parameter_entry(
            function.args.vararg, "var-positional", None, source_text
        )
        entry["required"] = False
        entry["optional"] = True
        parameters.append(entry)

    if function.args.kwarg is not None:
        entry = _parameter_entry(function.args.kwarg, "var-keyword", None, source_text)
        entry["required"] = False
        entry["optional"] = True
        parameters.append(entry)

    return parameters


def _own_constructor_source(cls: type) -> tuple[py_ast.FunctionDef, str] | None:
    if "__init__" not in cls.__dict__:
        return None

    try:
        source = textwrap.dedent(inspect.getsource(cls.__init__))
    except (OSError, TypeError):
        return None

    try:
        parsed = py_ast.parse(source)
    except SyntaxError as exc:
        raise LanguageSpecExtractionError(
            f"could not parse {cls.__name__}.__init__ source"
        ) from exc

    for item in parsed.body:
        if isinstance(item, py_ast.FunctionDef) and item.name == "__init__":
            return item, source
    return None


def _self_attribute_name(node: py_ast.AST) -> str | None:
    if not isinstance(node, py_ast.Attribute):
        return None
    value = node.value
    if isinstance(value, py_ast.Name) and value.id == "self":
        return node.attr
    return None


def _referenced_parameter(
    value: py_ast.AST | None, parameter_names: set[str]
) -> str | None:
    if isinstance(value, py_ast.Name) and value.id in parameter_names:
        return value.id
    if isinstance(value, py_ast.BoolOp):
        for item in value.values:
            name = _referenced_parameter(item, parameter_names)
            if name is not None:
                return name
    return None


def _field_source(value: py_ast.AST | None, parameter_name: str | None) -> str:
    if parameter_name is not None:
        if isinstance(value, py_ast.Name):
            return "parameter"
        return "parameter-derived"
    if isinstance(value, py_ast.Constant):
        return "constant"
    return "derived"


def _field_annotation(
    parameter_name: str | None,
    parameter_by_name: dict[str, dict[str, Any]],
    annotation: py_ast.AST | None,
    source_text: str,
) -> str | None:
    explicit = _source_segment(source_text, annotation)
    if explicit is not None:
        return explicit
    if parameter_name is not None:
        return parameter_by_name[parameter_name]["annotation"]
    return None


def _field_default(
    parameter_name: str | None,
    parameter_by_name: dict[str, dict[str, Any]],
    value: py_ast.AST | None,
    source_text: str,
) -> str | None:
    if parameter_name is not None:
        return parameter_by_name[parameter_name]["default"]
    if isinstance(value, py_ast.Constant):
        return _source_segment(source_text, value)
    return None


def _assignment_field_entries(
    statement: py_ast.AST,
    parameter_by_name: dict[str, dict[str, Any]],
    source_text: str,
) -> list[dict[str, Any]]:
    entries = []
    parameter_names = set(parameter_by_name)
    if isinstance(statement, py_ast.Assign):
        targets = statement.targets
        value = statement.value
        annotation = None
    elif isinstance(statement, py_ast.AnnAssign):
        targets = [statement.target]
        value = statement.value
        annotation = statement.annotation
    else:
        return entries

    for target in targets:
        name = _self_attribute_name(target)
        if name is None:
            continue
        parameter_name = _referenced_parameter(value, parameter_names)
        entries.append(
            {
                "name": name,
                "source": _field_source(value, parameter_name),
                "parameter": parameter_name,
                "annotation": _field_annotation(
                    parameter_name, parameter_by_name, annotation, source_text
                ),
                "required": (
                    bool(parameter_by_name[parameter_name]["required"])
                    if parameter_name is not None
                    else False
                ),
                "default": _field_default(
                    parameter_name, parameter_by_name, value, source_text
                ),
                "optional": (
                    bool(parameter_by_name[parameter_name]["optional"])
                    if parameter_name is not None
                    else True
                ),
                "initializer": _source_segment(source_text, value),
            }
        )
    return entries


def _class_field_inventory(
    cls: type,
) -> tuple[bool, list[dict[str, Any]], list[dict[str, Any]]]:
    constructor = _own_constructor_source(cls)
    if constructor is None:
        return False, [], []

    init, source_text = constructor
    parameters = _constructor_parameters(init, source_text)
    parameter_by_name = {parameter["name"]: parameter for parameter in parameters}
    fields_by_name = {}
    for statement in py_ast.walk(init):
        for entry in _assignment_field_entries(
            statement, parameter_by_name, source_text
        ):
            fields_by_name[entry["name"]] = entry

    return (
        True,
        parameters,
        [fields_by_name[name] for name in sorted(fields_by_name.keys())],
    )


def _ast_node_categories(cls: type) -> list[str]:
    categories = []
    for category, root_name in AST_CATEGORY_ROOTS:
        root_cls = getattr(ast_module, root_name, None)
        if root_cls is not None and issubclass(cls, root_cls):
            categories.append(category)
    return categories


def _ast_spec() -> dict[str, Any]:
    nodes = []
    aliases = {}
    enums = {}

    for public_name, cls in inspect.getmembers(ast_module, inspect.isclass):
        if getattr(cls, "__module__", None) != ast_module.__name__:
            continue

        if public_name != cls.__name__:
            aliases[public_name] = cls.__name__
            continue

        bases = [
            base.__name__
            for base in cls.__bases__
            if getattr(base, "__module__", None) == ast_module.__name__
            or base is enum.Enum
        ]

        if issubclass(cls, enum.Enum):
            enums[public_name] = [
                {"name": item.name, "value": item.value} for item in cls
            ]
            nodes.append(
                {
                    "name": public_name,
                    "bases": bases,
                    "categories": [],
                    "constructor_defined": False,
                    "constructor_parameters": [],
                    "fields": [],
                }
            )
            continue

        if not issubclass(cls, ast_module.ASTNode):
            continue

        constructor_defined, constructor_parameters, fields = _class_field_inventory(
            cls
        )
        nodes.append(
            {
                "name": public_name,
                "bases": bases,
                "categories": _ast_node_categories(cls),
                "constructor_defined": constructor_defined,
                "constructor_parameters": constructor_parameters,
                "fields": fields,
            }
        )

    return {
        "nodes": sorted(nodes, key=lambda item: item["name"]),
        "aliases": _jsonable(aliases),
        "enums": _jsonable(enums),
    }


def _validation_spec() -> dict[str, Any]:
    return {
        "metadata": _extract_constants(
            validation_module, VALIDATION_METADATA_CONSTANTS
        ),
        "rules": _extract_constants(validation_module, VALIDATION_RULE_CONSTANTS),
        "resources": _extract_constants(
            validation_module, VALIDATION_RESOURCE_CONSTANTS
        ),
    }


def extract_raw_language_facts() -> dict[str, Any]:
    """Return the translator-local extracted fact model before schema bridging."""
    return {
        "source": {
            "package": "crosstl.translator",
            "files": _source_inventory(),
            "method": (
                "live module constants/classes plus bounded static extraction "
                "of Parser.parse_type resource literals and AST constructor "
                "public field assignments"
            ),
        },
        "lexer": _lexer_spec(),
        "stages": _stage_spec(),
        "parser": _parser_spec(),
        "ast": _ast_spec(),
        "validation": _validation_spec(),
    }


def extract_language_spec() -> dict[str, Any]:
    """Return the compiler-compatible CrossTL frontend snapshot contract."""
    return _frontend_snapshot_from_raw(extract_raw_language_facts())


def render_language_spec_json() -> str:
    """Return the deterministic JSON representation of the language spec."""
    return json.dumps(extract_language_spec(), indent=2) + "\n"


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the language spec extraction CLI parser."""
    parser = argparse.ArgumentParser(
        description="Extract the CrossTL frontend language spec snapshot as JSON."
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Write JSON to this path instead of stdout.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for compiler-side tooling and local inspection."""
    args = build_arg_parser().parse_args(argv)
    text = render_language_spec_json()

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
        return 0

    sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
