"""Derive the GLSL inventory (desktop 4.60 core and GLSL ES 3.20) from the single-source specification.

The Khronos GLSL repository builds both specifications from the same AsciiDoc chapters, selecting text with
``ifdef::GLSL[]`` and ``ifdef::ESSL[]``. This module resolves those conditionals once per profile, numbers the
sections the way the published specifications do (chapters 1-8, numbered to three levels), and extracts:

- types from the basic-type tables (section "Basic Types"), structures and arrays;
- storage, auxiliary, interpolation, parameter, precision, memory and layout qualifiers from their tables;
- built-in variables and constants from the declarations in "Built-In Variables", with the stages that declare
  them (the compatibility profile is out of scope and skipped);
- built-in function prototypes from "Built-In Functions", one item per prototype;
- preprocessor directives, operators and predefined macros from "Preprocessor";
- stages from "Overview of Shading"; and every GLSL extension specification in the repository's
  ``extensions/`` directory.

Operators and statements are listed below by hand and cited to the sections this module numbers; a citation to a
section that does not exist in the pinned text is an error.

    python -m tests.conformance.inventory.glsl --fetch --check
    python -m tests.conformance.inventory.glsl --inputs-dir DIR --write
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import tempfile
from pathlib import Path

from . import DATA, InventoryError, curated, download, dumps, validate

REPOSITORY = "KhronosGroup/GLSL"
COMMIT = "0099bf83b0288469917fc2e2972cb0a2b74d6711"
REVISION = 9  # core.adoc ":revision:"
# Numbered chapters in core.adoc order; the preamble is not numbered.
CHAPTERS = (
    "introduction",
    "overview",
    "basics",
    "variables",
    "operators",
    "statements",
    "builtins",
    "builtinfunctions",
)
SHA256 = {
    "introduction.adoc": (
        "c0257c6a89e98bc4b1e3dabf49b1aaca072f480c9227072e58b42ad2c7328db4"
    ),
    "overview.adoc": "1a825c79824cc99a61b687721616e6898e374f418cd154b2d5b86d2b1033ad3b",
    "basics.adoc": "893b49819fab670730569caa8edb72695dd23f93d7008484da263a1c6cb24a64",
    "variables.adoc": (
        "10b33a2b195c6eb79f99a9777324f33fac73ab61d543e0fe10e950da6d660fe9"
    ),
    "operators.adoc": (
        "68a0be9be0cef3a176f1f7da39b71b57a393441b3fd75e3d94da64d49ee85479"
    ),
    "statements.adoc": (
        "9dca68089dbb09ed27ead9436beb466a06eec9cf4255d56f54909d7b943b852e"
    ),
    "builtins.adoc": "fb3d69086396b61f1bfd29a6335710796488339269289d18c2e459293c86a139",
    "builtinfunctions.adoc": (
        "95c602702118f5aabba3a5e8569abfca9865d4c86e02a210021eed16b5dd6bc5"
    ),
    # Sorted paths under extensions/ at COMMIT, from the git tree API.
    "extensions.txt": (
        "130d4a1de943ac607ec160595adc55b2f40b3e1ec5c8661ce47b01b713e46194"
    ),
}
PROFILES = {"glsl": "GLSL", "essl": "ESSL"}
SOURCES = {
    "glsl": ("glsl-4.60", "The OpenGL Shading Language, Version 4.60", "4.60", 460),
    "essl": ("essl-3.20", "The OpenGL ES Shading Language, Version 3.20", "3.20", 320),
}
ATTRIBUTES = {
    "GLSL": {
        "slname": "OpenGL Shading Language",
        "slabbrev": "GLSL",
        "apiversion": "4.6",
        "apiname": "OpenGL",
    },
    "ESSL": {
        "slname": "OpenGL ES Shading Language",
        "slabbrev": "GLSL ES",
        "apiversion": "3.2",
        "apiname": "OpenGL ES",
    },
}
PRECISION = {"highp", "mediump", "lowp"}
# Typographical errors in prototypes at COMMIT, corrected before parsing. Each must still occur in the text, so a
# fixed upstream typo forces this table to be revisited.
ERRATA = {
    "samplerCubeShadow sampler, in lod": "samplerCubeShadow sampler, int lod",
    "gsamler2DRect sampler": "gsampler2DRect sampler",
}
# Placeholder types the function chapter defines for families of prototypes.
GENERIC_TYPES = {
    "genFType",
    "genDType",
    "genIType",
    "genUType",
    "genBType",
    "vec",
    "ivec",
    "uvec",
    "bvec",
    "mat",
    "dvec",
    "dmat",
    "gvec4",
    "IMAGE_PARAMS",
}
PARAMETER_QUALIFIERS = {
    "in",
    "out",
    "inout",
    "const",
    "coherent",
    "volatile",
    "restrict",
    "readonly",
    "writeonly",
}
HEADING = re.compile(r"^(=+) ([A-Z*{_`\[].*)$")
DELIMITER = re.compile(r"^(-{4,}|\.{4,}|\+{4,}|/{4,}|\*{4,}|_{4,}|={4,})$")
PROTOTYPE = re.compile(
    r"^(?P<ret>(?:(?:highp|mediump|lowp)\s+)?[A-Za-z_]\w*)\s+(?P<name>[A-Za-z_]\w*)\s*\((?P<args>[^()]*)\)\s*;?$"
)
DECLARATOR = re.compile(r"\b(gl_\w+)\s*(?:\[[^\]]*\])?\s*(?:=[^;]*)?;")
BLOCK = re.compile(r"\b(gl_\w+)\s*\{")
BOLD = re.compile(r"\*([A-Za-z_][\w]*)\*")

# Built-in variable sections (level-3 titles) -> stage; None skips the section.
VARIABLE_SECTIONS = {
    "Vertex Shader Special Variables": "vertex",
    "Tessellation Control Shader Special Variables": "tessellation-control",
    "Tessellation Evaluation Shader Special Variables": "tessellation-evaluation",
    "Geometry Shader Special Variables": "geometry",
    "Fragment Shader Special Variables": "fragment",
    "Compute Shader Special Variables": "compute",
    "Compatibility Profile Built-In Language Variables": None,
    "Compatibility Profile Built-In Constants": None,
    "Compatibility Profile State": None,
}
VARIABLE_CHAPTER_SECTIONS = {
    "Built-In Constants": "constant",
    "Built-In Uniform State": "uniform",
    "Compatibility Profile Vertex Shader Built-In Inputs": None,
    "Redeclaring Built-In Blocks": None,
}
# Qualifier tables: section title -> (id kind, category).
QUALIFIER_TABLES = {
    "Storage Qualifiers": ("storage", "qualifier"),
    "Interpolation Qualifiers": ("interpolation", "qualifier"),
    "Parameter Qualifiers": ("parameter", "qualifier"),
    "Precision Qualifiers": ("precision", "qualifier"),
    "Memory Qualifiers": ("memory", "qualifier"),
    "Layout Qualifiers": ("layout", "qualifier"),
}
RESOURCE_STORAGE = {"uniform", "buffer", "shared"}
RESOURCE_LAYOUTS = {
    "binding",
    "set",
    "push_constant",
    "input_attachment_index",
    "std140",
    "std430",
    "shared",
    "packed",
    "offset",
    "align",
    "row_major",
    "column_major",
}
COMPATIBILITY = {"attribute", "varying"}
STAGES = {
    "Vertex Processor": "vertex",
    "Tessellation Control Processor": "tessellation-control",
    "Tessellation Evaluation Processor": "tessellation-evaluation",
    "Geometry Processor": "geometry",
    "Fragment Processor": "fragment",
    "Compute Processor": "compute",
}


def slug(text):
    return re.sub(r"[^A-Za-z0-9_]+", "-", text).strip("-")


def preprocess(text, profile):
    """Resolve ifdef/ifndef conditionals for one profile (``GLSL`` or ``ESSL``)."""
    lines, stack = [], []
    for line in text.splitlines():
        match = re.match(r"^(ifdef|ifndef)::([\w,+\-]+)\[(.*)\]\s*$", line)
        if match:
            kind, names, content = match.groups()
            defined = any(name == profile for name in re.split(r"[,+]", names))
            active = defined if kind == "ifdef" else not defined
            if content:
                if all(stack) and active:
                    lines.append(content)
            else:
                stack.append(active)
            continue
        if re.match(r"^endif::[\w,+\-]*\[\]\s*$", line):
            if not stack:
                raise InventoryError(f"unbalanced endif in {profile} text: {line!r}")
            stack.pop()
            continue
        if all(stack):
            lines.append(line)
    if stack:
        raise InventoryError(f"unterminated conditional in {profile} text")
    return lines


def title_text(title, profile):
    for name, value in ATTRIBUTES[profile].items():
        title = title.replace("{" + name + "}", value)
    return title


class Chapter:
    """One chapter after preprocessing: each line tagged with its numbered section and enclosing titles."""

    def __init__(self, number, lines, profile):
        self.lines = (
            []
        )  # (text, section number, level-3 title, nearest title, in_block)
        self.sections = {}
        self.titles = {}
        counters = [number, 0, 0]
        current, level3, nearest = str(number), None, None
        block = None
        for text in lines:
            stripped = text.strip()
            if block:
                if stripped == block:
                    block = None
                self.lines.append((text, current, level3, nearest, True))
                continue
            if DELIMITER.match(stripped) or stripped.startswith("|==="):
                block = stripped
                self.lines.append((text, current, level3, nearest, False))
                continue
            heading = HEADING.match(text)
            if heading:
                level, title = len(heading.group(1)), title_text(
                    heading.group(2).strip(), profile
                )
                if level == 1:
                    current, level3 = str(number), None
                    self.sections[current] = title
                elif level == 2:
                    counters[1], counters[2] = counters[1] + 1, 0
                    current, level3 = f"{number}.{counters[1]}", None
                    self.sections[current] = title
                elif level == 3:
                    counters[2] += 1
                    current, level3 = f"{number}.{counters[1]}.{counters[2]}", title
                    self.sections[current] = title
                nearest = title
                self.titles.setdefault(
                    title, current
                )  # unnumbered headings cite their numbered parent
            self.lines.append((text, current, level3, nearest, False))

    def section(self, title):
        return self.titles.get(title)

    def tables(self):
        """Rows of every table: (first-cell bold tokens, section number, nearest title, caption)."""
        rows, inside, continuing, caption, last = [], False, False, None, None
        for text, section, _, nearest, _ in self.lines:
            stripped = text.strip()
            if stripped.startswith("|==="):
                inside, continuing = not inside, False
                if inside:
                    caption = last
                continue
            if not inside:
                if stripped.startswith("*") and stripped.endswith("*"):
                    last = stripped.strip("*")
                elif stripped:
                    last = last if stripped.startswith("[") else None
                continue
            if stripped.startswith("|"):
                cell = stripped[1:].split("|", 1)[0]
                continuing = cell.rstrip().endswith("+") and "|" not in stripped[1:]
                rows.append((BOLD.findall(cell), section, nearest, caption))
            elif continuing:
                cell = stripped.split("|", 1)[0]
                continuing = cell.rstrip().endswith("+") and "|" not in stripped
                rows[-1][0].extend(BOLD.findall(cell))
        return rows

    def listings(self):
        """Source listings: (lines, section number, level-3 title, nearest title)."""
        result, current, armed = [], None, False
        for text, section, level3, nearest, in_block in self.lines:
            stripped = text.strip()
            if not in_block and stripped.startswith("[source"):
                armed = True
                continue
            if current is None and armed and stripped == "----" and not in_block:
                current, armed = ([], section, level3, nearest), False
                continue
            if current is not None:
                if stripped == "----" and in_block:
                    result.append(current)
                    current = None
                else:
                    current[0].append(text)
                continue
            if stripped:
                armed = armed and stripped.startswith("[")
        return result


def parse_parameters(text):
    """Split a parameter list; parameters inside ``[ ... ]`` are optional, as in ``vec2 P [, float bias]``."""
    text = re.sub(
        r"(\w)\[(\d*)\]", r"\1<\2>", text
    )  # protect array suffixes such as offsets[4]
    parameters, depth = [], 0
    for piece in text.split(","):
        stripped = piece.strip()
        optional = depth > 0 or stripped.startswith("[")
        depth += piece.count("[") - piece.count("]")
        part = re.sub(
            r"<(\d*)>", r"[\1]", piece.replace("[", "").replace("]", "")
        ).strip()
        if part:
            parameters.append((part, optional))
    if depth:
        raise InventoryError(f"unbalanced optional parameters in {text!r}")
    return parameters


def parameter_type(text):
    tokens = text.split()
    if all(token in PARAMETER_QUALIFIERS for token in tokens[:-1]):
        # A lone type, possibly qualified: an unnamed parameter such as "readonly IMAGE_PARAMS".
        return tokens[-1], [*tokens[:-1]]
    name, words = tokens[-1], tokens[:-1]
    array = re.search(r"\[(\d*)\]$", name)
    qualifiers = [word for word in words if word in PARAMETER_QUALIFIERS]
    types = [
        word
        for word in words
        if word not in PARAMETER_QUALIFIERS and word not in PRECISION
    ]
    base = " ".join(types)
    if array:
        base += f"[{array.group(1)}]"
    return base, qualifiers


def prototypes(chapter, applied):
    found = []
    for lines, section, _, nearest in chapter.listings():
        joined, buffer = [], ""
        for text in lines:
            for wrong, right in ERRATA.items():
                if wrong in text:
                    text = text.replace(wrong, right)
                    applied.add(wrong)
            text = re.sub(r"//.*", "", text).strip()
            if not text:
                continue
            buffer = f"{buffer} {text}".strip()
            if buffer.count("(") == buffer.count(")"):
                joined.append(buffer)
                buffer = ""
        for text in joined:
            match = PROTOTYPE.match(re.sub(r"\s+", " ", text))
            if not match or "=" in text:
                continue
            ret = " ".join(
                word for word in match["ret"].split() if word not in PRECISION
            )
            params = (
                []
                if match["args"].strip() in {"", "void"}
                else parse_parameters(match["args"])
            )
            shown, keys = [], []
            for part, optional in params:
                base, qualifiers = parameter_type(part)
                key = re.sub(
                    r"\[(\d*)\]", lambda m: f"_array{m.group(1)}", base
                ).replace(" ", "_")
                keys.append(("opt_" if optional else "") + key)
                shown.append(
                    ("[" if optional else "")
                    + " ".join([*qualifiers, base])
                    + ("]" if optional else "")
                )
            name = match["name"]
            signature = f"{ret} {name}({', '.join(shown)})"
            found.append(
                (
                    f"glsl.function.{name}.{'-'.join(keys) or 'void'}",
                    name,
                    signature,
                    section,
                    nearest,
                )
            )
    return found


def variables(chapter):
    found = []
    level2 = {}
    for number, title in chapter.sections.items():
        if number.count(".") == 1:
            level2[number] = title
    for lines, section, level3, nearest in chapter.listings():
        top = level2.get(".".join(section.split(".")[:2]))
        if level3 in VARIABLE_SECTIONS:
            stage = VARIABLE_SECTIONS[level3]
        elif top in VARIABLE_CHAPTER_SECTIONS:
            stage = VARIABLE_CHAPTER_SECTIONS[top]
        else:
            continue
        if stage is None:
            continue
        code = " ".join(re.sub(r"//.*", "", text) for text in lines)
        names = [(name, "block") for name in BLOCK.findall(code)]
        kind = (
            "constant"
            if stage == "constant"
            else "uniform" if stage == "uniform" else "variable"
        )
        names += [(name, kind) for name in DECLARATOR.findall(code)]
        for name, item_kind in names:
            found.append((name, item_kind, stage, section))
    return found


class Profile:
    """One profile's chapters, the sections its items cite, and the items found so far (first finding wins)."""

    def __init__(self, profile, texts):
        self.profile = profile
        self.chapters = {
            name: Chapter(number, preprocess(texts[f"{name}.adoc"], profile), profile)
            for number, name in enumerate(CHAPTERS, start=1)
        }
        self.cited = {}
        self.errata = set()
        self.items = {}  # id -> (item without cite, section number)

    def find(self, chapter, title):
        return self.chapters[chapter].section(title)

    def add(self, item_id, name, category, kind, section, chapter, **extra):
        if section is None or item_id in self.items:
            return
        self.cited[section] = self.chapters[chapter].sections[section]
        item = {"id": item_id, "name": name, "category": category, "kind": kind}
        item.update({key: value for key, value in extra.items() if value})
        self.items[item_id] = (item, section)


def _table_items(p):
    """Basic types and qualifiers from the tables of "Variables and Types"."""
    chapter = p.chapters["variables"]
    basic = chapter.section("Basic Types")
    for tokens, section, nearest, caption in chapter.tables():
        if section == basic:
            opaque = bool(caption and "Opaque" in caption)
            for token in tokens:
                category, kind = ("resource", "opaque") if opaque else ("type", "basic")
                p.add(f"glsl.type.{token}", token, category, kind, section, "variables")
            continue
        top = chapter.sections.get(".".join(section.split(".")[:2]))
        title = (
            nearest
            if nearest in QUALIFIER_TABLES
            else top if top in QUALIFIER_TABLES else None
        )
        if title is None:
            continue
        kind, category = QUALIFIER_TABLES[title]
        for token in tokens:
            if token in COMPATIBILITY:
                continue
            item_kind, item_category = kind, category
            if title == "Storage Qualifiers" and token in {
                "centroid",
                "sample",
                "patch",
            }:
                item_kind = "auxiliary"
            if (item_kind, token in RESOURCE_STORAGE) == ("storage", True) or (
                item_kind == "layout" and token in RESOURCE_LAYOUTS
            ):
                item_category = "resource"
            if item_kind == "layout" and re.match(r"^r(g(ba?)?)?\d", token):
                item_kind, item_category = "format", "resource"
            p.add(
                f"glsl.{item_kind}.{token}",
                token,
                item_category,
                item_kind,
                section,
                "variables",
            )


def _curated_items(p):
    """Constructs listed in curated/glsl.tsv, cited by section title."""
    for row in curated("glsl"):
        for chapter in ("variables", "operators", "statements"):
            section = p.find(chapter, row["reference"])
            if section:
                p.add(
                    row["id"],
                    row["name"],
                    row["category"],
                    row["kind"],
                    section,
                    chapter,
                    note=row["note"],
                )
                break


def _function_items(p):
    types = {
        item_id.split(".", 2)[2]
        for item_id in p.items
        if item_id.startswith("glsl.type.")
    }
    for item_id, name, signature, section, _ in prototypes(
        p.chapters["builtinfunctions"], p.errata
    ):
        unknown = [
            kind for kind in _signature_types(signature) if not _known_type(kind, types)
        ]
        if unknown:
            raise InventoryError(
                f"{signature}: unknown types {unknown} in the {p.profile} prototypes"
            )
        p.add(
            item_id,
            name,
            "builtin-function",
            "function",
            section,
            "builtinfunctions",
            signature=signature,
        )


def _variable_items(p):
    stages = {}
    for name, kind, stage, section in variables(p.chapters["builtins"]):
        stages.setdefault(name, (kind, set(), section))[1].add(stage)
    for name, (kind, used, section) in stages.items():
        used = sorted(used - {"constant", "uniform"})
        requires = {"stages": used} if used else None
        p.add(
            f"glsl.variable.{name}",
            name,
            "builtin-variable",
            kind,
            section,
            "builtins",
            requires=requires,
        )


def _preprocessor_items(p):
    preprocessor = p.find("basics", "Preprocessor")
    names = {"#": "null-directive", "##": "token-paste", "defined": "defined"}
    for text, section, _, _, in_block in p.chapters["basics"].lines:
        if section != preprocessor:
            continue
        line = text.strip()
        directive = re.match(r"^(?:\{empty\}::\s*)?(#\w*|defined|##)\s*\+?$", line)
        if directive:
            token = directive.group(1)
            kind = "operator" if token in {"##", "defined"} else "directive"
            name = names.get(token, token.lstrip("#"))
            p.add(
                f"glsl.preprocessor.{name}",
                token,
                "preprocessor",
                kind,
                section,
                "basics",
            )
        for macro in re.findall(r"\+(__\w+__)\+", line):
            p.add(
                f"glsl.macro.{macro}", macro, "preprocessor", "macro", section, "basics"
            )
        defined = re.match(r"^#define (\w+) 1$", line)
        if defined and in_block:
            macro = defined.group(1)
            p.add(
                f"glsl.macro.{macro}", macro, "preprocessor", "macro", section, "basics"
            )


def _profile_items(p):
    """Items of one profile: id -> (item without cite, section number)."""
    _table_items(p)
    _curated_items(p)
    _function_items(p)
    _variable_items(p)
    _preprocessor_items(p)
    for title, stage in STAGES.items():
        p.add(
            f"glsl.stage.{stage}",
            stage,
            "stage",
            "stage",
            p.find("overview", title),
            "overview",
        )
    return p.items


def _signature_types(signature):
    returned, rest = signature.split(" ", 1)
    arguments = rest[rest.index("(") + 1 : -1]
    kinds = [returned]
    for argument in filter(None, arguments.split(", ")):
        if argument.startswith("[") and argument.endswith("]"):
            argument = argument[1:-1]
        kinds.append(re.sub(r"\[\d*\]$", "", argument.split()[-1]))
    return kinds


def _known_type(kind, types):
    if kind in types or kind in GENERIC_TYPES or kind == "void":
        return True
    # gsampler2D, gimage2D, gtexture2D, gsubpassInput: the float, int and uint opaque families.
    return kind.startswith("g") and kind[1:] in types and ("i" + kind[1:]) in types


def _extension_name(path):
    stem = Path(path).stem
    return "GL_" + stem[len("GLSL_") :] if stem.startswith("GLSL_") else stem


def derive(texts, cells=None):
    profiles = {key: Profile(name, texts) for key, name in PROFILES.items()}
    per_profile = {key: _profile_items(profile) for key, profile in profiles.items()}
    missing = [
        row["reference"]
        for row in curated("glsl")
        if not any(row["id"] in found for found in per_profile.values())
    ]
    if missing:
        raise InventoryError(
            f"curated sections do not exist in either profile: {missing}"
        )
    unused = set(ERRATA) - set().union(
        *(profile.errata for profile in profiles.values())
    )
    if unused:
        raise InventoryError(
            f"errata no longer occur in the specification: {sorted(unused)}"
        )
    items = {}
    for key in ("glsl", "essl"):
        source = SOURCES[key][0]
        for item_id, (item, section) in per_profile[key].items():
            if item_id not in items:
                items[item_id] = {**item, "cite": f"{source} §{section}"}
                if not items[item_id].get("requires"):
                    items[item_id].pop("requires", None)
            entry = items[item_id]
            entry.setdefault("requires", {}).setdefault("profiles", []).append(key)
    extensions = sorted(
        {
            _extension_name(line.strip())
            for line in texts["extensions.txt"].splitlines()
            if re.search(r"/(GL|GLSL)_\w+\.(txt|adoc|asciidoc)$", line.strip())
        }
    )
    documents = {}
    for line in sorted(texts["extensions.txt"].splitlines()):
        path = line.strip()
        if re.search(r"/(GL|GLSL)_\w+\.(txt|adoc|asciidoc)$", path):
            documents.setdefault(
                _extension_name(path),
                f"https://github.com/{REPOSITORY}/blob/{COMMIT}/{path}",
            )
    for name in extensions:
        items[f"glsl.extension.{name}"] = {
            "id": f"glsl.extension.{name}",
            "name": name,
            "category": "extension",
            "kind": "extension",
            "cite": f"glsl-extensions {name}",
        }
    ordered = []
    for item_id in sorted(items):
        item = items[item_id]
        if "requires" in item:
            requires = dict(item["requires"])
            requires["profiles"] = sorted(set(requires.get("profiles", [])))
            item["requires"] = {key: value for key, value in requires.items() if value}
            if not item["requires"]:
                del item["requires"]
        ordered.append(item)
    cells = dict(cells or {})
    orphaned = sorted(set(cells) - set(items))
    if orphaned:
        raise InventoryError(
            f"classified items vanished from the specification: {', '.join(orphaned)}"
        )
    sources = {}
    for key, (source, title, version, _) in SOURCES.items():
        cited = profiles[key].cited
        sources[source] = {
            "title": f"{title}.{REVISION} (single-source AsciiDoc)",
            "url": f"https://github.com/{REPOSITORY}/tree/{COMMIT}",
            "version": f"{version}.{REVISION}",
            "sections": dict(
                sorted(
                    cited.items(),
                    key=lambda kv: [int(part) for part in kv[0].split(".")],
                )
            ),
        }
    sources["glsl-extensions"] = {
        "title": "GLSL extension specifications in the Khronos GLSL repository",
        "url": f"https://github.com/{REPOSITORY}/tree/{COMMIT}/extensions",
        "documents": documents,
    }
    return {
        "schemaVersion": 1,
        "language": "glsl",
        "title": "GLSL 4.60 core and GLSL ES 3.20",
        "sources": sources,
        "generated": {
            "by": "tests/conformance/inventory/glsl.py",
            "repository": f"https://github.com/{REPOSITORY}",
            "commit": COMMIT,
            "files": dict(sorted(SHA256.items())),
            "scope": "core profile only: compatibility-profile constructs are omitted",
        },
        "items": ordered,
        "cells": cells,
    }


def read_inputs(directory):
    texts = {}
    for name, expected in SHA256.items():
        data = (Path(directory) / name).read_bytes()
        digest = hashlib.sha256(data).hexdigest()
        if digest != expected:
            raise InventoryError(
                f"{name} has sha256 {digest}, not the pinned {expected}"
            )
        texts[name] = data.decode()
    return texts


def fetch(directory):
    base = f"https://raw.githubusercontent.com/{REPOSITORY}/{COMMIT}/chapters/"
    for name in SHA256:
        if name.endswith(".adoc"):
            (Path(directory) / name).write_bytes(download(base + name))
    listing = json.loads(
        download(
            f"https://api.github.com/repos/{REPOSITORY}/git/trees/{COMMIT}?recursive=1"
        )
    )
    paths = sorted(
        entry["path"]
        for entry in listing["tree"]
        if entry["path"].startswith("extensions/")
    )
    (Path(directory) / "extensions.txt").write_text("\n".join(paths) + "\n")


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    where = parser.add_mutually_exclusive_group(required=True)
    where.add_argument(
        "--inputs-dir", type=Path, help=f"Directory holding {', '.join(SHA256)}"
    )
    where.add_argument(
        "--fetch", action="store_true", help="Download the pinned inputs"
    )
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--write", action="store_true")
    action.add_argument("--check", action="store_true")
    parser.add_argument("--output", type=Path, default=DATA / "glsl.json")
    args = parser.parse_args(argv)
    with tempfile.TemporaryDirectory() as scratch:
        directory = args.inputs_dir
        if args.fetch:
            directory = Path(scratch)
            fetch(directory)
        texts = read_inputs(directory)
    existing = json.loads(args.output.read_text()) if args.output.is_file() else {}
    data = derive(texts, existing.get("cells"))
    errors = validate(data, language="glsl")
    if errors:
        raise InventoryError("\n".join(errors))
    text = dumps(data)
    if args.write:
        args.output.write_text(text)
        print(f"Wrote {args.output}: {len(data['items'])} items")
        return 0
    if args.output.is_file() and args.output.read_text() == text:
        print(
            f"{args.output} matches the pinned specification ({len(data['items'])} items)"
        )
        return 0
    print(f"{args.output} is out of date; regenerate with --write", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
