"""Machine-readable feature inventories of MSL, HLSL, GLSL (desktop and ES) and SPIR-V.

Each language has one JSON file under ``data/``. An inventory lists the language's constructs as *items*, each
cited to a section of a pinned specification, and records for every (item, target) *cell* the state the translator
gives the construct on that target:

- ``preserved``: the construct reaches the target with its source semantics;
- ``polyfilled``: the target has no direct equivalent and the translator emits an exact emulation;
- ``refused``: the target cannot express the construct and translation stops with a typed diagnostic, named in
  the cell's ``diagnostic`` field.

A cell that is absent is *unknown*. A classified cell must name at least one test that demonstrates its state:
``case:<id>`` for a conformance case, or ``pytest:<path>::<test>`` for a test function in the repository. Refusals
must name their diagnostic code. Completion of the inventory work means no cell is unknown
(``python -m tests.conformance.inventory --require-complete``).

File format (``schemaVersion`` 1)::

    {
      "schemaVersion": 1,
      "language": "glsl",
      "title": "...",
      "sources": {"<source id>": {"title": ..., "url": ..., "version": ..., "sections": {"<n>": "<title>"}}},
      "generated": {...},                        # optional: how the items were derived
      "items": [{"id": ..., "name": ..., "category": ..., "kind": ..., "cite": "<source id> §<section>",
                 "requires": {...}, "aliases": [...], "signature": ..., "note": ...}],
      "cells": {"<item id>": {"<target>": {"status": ..., "tests": [...], "diagnostic": ..., "note": ...}}}
    }

A citation names one of the file's sources and either a numbered section listed in that source's ``sections``
table (``glsl-4.60 §8.3``), a stable section label listed in its ``labels`` table (``hlsl-spec [Decl.cbuffer]``)
or, for collections of documents such as extension registries, a listed document name
(``spirv-registry SPV_KHR_ray_query``).
"""

from __future__ import annotations

import json
import os
import re
import urllib.request
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

SCHEMA = 1
DATA = Path(__file__).resolve().parent / "data"
CURATED = Path(__file__).resolve().parent / "curated"
REPOSITORY = Path(__file__).resolve().parents[3]

LANGUAGES = ("metal", "hlsl", "glsl", "spirv")
TARGETS = ("metal", "hlsl", "glsl", "essl", "spirv")
STATUSES = ("preserved", "polyfilled", "refused")
CATEGORIES = (
    "type",
    "qualifier",
    "declaration",
    "statement",
    "operator",
    "builtin-function",
    "builtin-variable",
    "resource",
    "stage",
    "extension",
    "preprocessor",
)
# The contract's inventory scope, as groups of categories every language must populate.
REQUIRED_GROUPS = {
    "types": {"type"},
    "qualifiers and attributes": {"qualifier"},
    "declarations, statements and operators": {"declaration", "statement", "operator"},
    "built-in functions and variables": {"builtin-function", "builtin-variable"},
    "resources and binding models": {"resource"},
    "stages": {"stage"},
    "extensions": {"extension"},
}
# Targets a source language translates to: every target outside its own language family.
FAMILY = {
    "metal": {"metal"},
    "hlsl": {"hlsl"},
    "glsl": {"glsl", "essl"},
    "spirv": {"spirv"},
}
REQUIRES = {
    "metal": {"msl", "platforms", "stages"},
    "hlsl": {"sm", "hlsl", "stages"},
    "glsl": {"profiles", "glsl", "essl", "extensions", "stages"},
    "spirv": {"version", "capabilities", "extensions"},
}
ID = re.compile(r"^[a-z]+(\.[A-Za-z0-9_\-]+)+$")
VERSION = re.compile(r"^\d+\.\d+$")
CITE = re.compile(
    r"^(?P<source>[a-z0-9.\-]+) "
    r"(?:§(?P<section>[0-9A-Z]+(?:\.[0-9]+)*)|\[(?P<label>[A-Za-z][\w.\-]*)\]|(?P<document>[^\s\[]\S*))$"
)
TEST = re.compile(
    r"^(?:case:(?P<case>[A-Za-z0-9_]+)|pytest:(?P<path>[\w./\-]+\.py)::(?P<test>\w+))$"
)


class InventoryError(ValueError):
    """An inventory file violates the schema; the message lists every violation."""


@dataclass(frozen=True)
class Inventory:
    language: str
    path: Path | None
    data: dict

    @property
    def items(self):
        return self.data["items"]

    @property
    def cells(self):
        return self.data.get("cells", {})

    def targets(self):
        return targets_for(self.language)

    def counts(self):
        """Cell states per target, with absent cells counted as ``unknown``."""
        result = {}
        for target in self.targets():
            counter = Counter(
                self.cells.get(item["id"], {}).get(target, {}).get("status", "unknown")
                for item in self.items
            )
            result[target] = dict(sorted(counter.items()))
        return result

    def unknown(self):
        return sum(counts.get("unknown", 0) for counts in self.counts().values())


def download(url):
    """GET a pinned upstream input; GitHub API calls use GITHUB_TOKEN when set, to avoid anonymous rate limits."""
    request = urllib.request.Request(
        url, headers={"User-Agent": "crosstl-conformance-inventory"}
    )
    token = os.environ.get("GITHUB_TOKEN")
    if token and url.startswith("https://api.github.com/"):
        request.add_header("Authorization", f"Bearer {token}")
    with urllib.request.urlopen(
        request, timeout=120
    ) as response:  # noqa: S310 - fixed https URLs
        return response.read()


def curated(language):
    """Hand-listed constructs of a language: rows of curated/<language>.tsv as dictionaries.

    The ``reference`` column is a citation (``hlsl-spec [Decl.cbuffer]``) or, for sources numbered from a table of
    contents, the exact section title the derivation resolves to a number.
    """
    lines = (CURATED / f"{language}.tsv").read_text().splitlines()
    header = lines[0].split("\t")
    rows = []
    for line in filter(None, lines[1:]):
        fields = line.split("\t")
        fields += [""] * (
            len(header) - len(fields)
        )  # the trailing note column may be empty
        row = dict(zip(header, fields))
        if len(fields) != len(header) or not all(
            row[key] for key in header if key != "note"
        ):
            raise InventoryError(f"curated/{language}.tsv: malformed row {line!r}")
        rows.append(row)
    return rows


def targets_for(language):
    return tuple(target for target in TARGETS if target not in FAMILY[language])


def _known_cases():
    from ..cases import BY_ID

    return set(BY_ID)


def _test_exists(reference, cases):
    match = TEST.match(reference)
    if not match:
        return False
    if match["case"]:
        return match["case"] in cases
    path = (REPOSITORY / match["path"]).resolve()
    if REPOSITORY not in path.parents or not path.is_file():
        return False
    return (
        re.search(rf"^\s*(?:async\s+)?def {match['test']}\(", path.read_text(), re.M)
        is not None
    )


def _check_requires(language, item, errors):
    requires = item.get("requires", {})
    if not isinstance(requires, dict):
        errors.append(f"{item['id']}: requires must be an object")
        return
    for key, value in requires.items():
        if key not in REQUIRES[language]:
            errors.append(
                f"{item['id']}: requirement {key!r} is not defined for {language}"
            )
        elif key in {"capabilities", "extensions", "stages", "platforms", "profiles"}:
            if not (
                isinstance(value, list)
                and value
                and all(isinstance(v, str) for v in value)
            ):
                errors.append(
                    f"{item['id']}: requirement {key!r} must be a non-empty list of names"
                )
            elif value != sorted(set(value)):
                errors.append(
                    f"{item['id']}: requirement {key!r} must be sorted and unique"
                )
        elif not (isinstance(value, str) and (VERSION.match(value) or value == "none")):
            errors.append(
                f"{item['id']}: requirement {key!r} must be a version like 4.60 or 'none'"
            )


def _check_cite(item, sources, errors):
    match = CITE.match(item.get("cite", ""))
    if not match:
        errors.append(f"{item['id']}: citation {item.get('cite')!r} is malformed")
        return
    source = sources.get(match["source"])
    if source is None:
        errors.append(
            f"{item['id']}: citation names unknown source {match['source']!r}"
        )
        return
    if match["section"] and match["section"] not in source.get("sections", {}):
        errors.append(
            f"{item['id']}: section {match['section']} is not listed for {match['source']}"
        )
    if match["label"] and match["label"] not in source.get("labels", {}):
        errors.append(
            f"{item['id']}: label [{match['label']}] is not listed for {match['source']}"
        )
    if match["document"] and match["document"] not in source.get("documents", {}):
        errors.append(
            f"{item['id']}: document {match['document']} is not listed for {match['source']}"
        )


def _check_cells(language, data, ids, cases, errors):
    cells = data.get("cells", {})
    if not isinstance(cells, dict):
        errors.append("cells must be an object")
        return
    allowed = set(targets_for(language))
    for item_id, by_target in cells.items():
        if item_id not in ids:
            errors.append(f"cells name unknown item {item_id!r}")
            continue
        for target, cell in by_target.items():
            where = f"{item_id} -> {target}"
            if target not in allowed:
                errors.append(f"{where}: {target} is not a target of {language}")
                continue
            if cell.get("status") not in STATUSES:
                errors.append(f"{where}: status must be one of {', '.join(STATUSES)}")
            tests = cell.get("tests")
            if not (isinstance(tests, list) and tests):
                errors.append(f"{where}: a classified cell needs at least one test")
            else:
                errors.extend(
                    f"{where}: test {test!r} does not exist"
                    for test in tests
                    if not _test_exists(test, cases)
                )
            if cell.get("status") == "refused" and not cell.get("diagnostic"):
                errors.append(f"{where}: a refusal must name its diagnostic code")
            if cell.get("status") != "refused" and "diagnostic" in cell:
                errors.append(f"{where}: only refusals carry a diagnostic code")
            unknown = set(cell) - {"status", "tests", "diagnostic", "note"}
            if unknown:
                errors.append(f"{where}: unknown fields {sorted(unknown)}")


def validate(data, *, language=None, cases=None):
    """Return the list of schema violations of an inventory's parsed JSON (empty when valid)."""
    errors = []
    if data.get("schemaVersion") != SCHEMA:
        errors.append(f"schemaVersion must be {SCHEMA}")
    language = language or data.get("language")
    if data.get("language") != language or language not in LANGUAGES:
        return errors + [
            f"language must be one of {', '.join(LANGUAGES)} and match the file"
        ]
    sources = data.get("sources")
    if not (isinstance(sources, dict) and sources):
        return errors + ["sources must name at least one specification"]
    for name, source in sources.items():
        if not (source.get("title") and source.get("url")):
            errors.append(f"source {name}: title and url are required")
        if not (
            source.get("sections") or source.get("labels") or source.get("documents")
        ):
            errors.append(
                f"source {name}: list the sections, labels or documents that items cite"
            )
    items = data.get("items")
    if not (isinstance(items, list) and items):
        return errors + ["items must be a non-empty list"]
    ids = set()
    for item in items:
        item_id = item.get("id", "")
        if not ID.match(item_id) or not item_id.startswith(language + "."):
            errors.append(
                f"item id {item_id!r} must look like {language}.<kind>.<name>"
            )
        if item_id in ids:
            errors.append(f"duplicate item id {item_id!r}")
        ids.add(item_id)
        if not item.get("name"):
            errors.append(f"{item_id}: name is required")
        if item.get("category") not in CATEGORIES:
            errors.append(
                f"{item_id}: category {item.get('category')!r} is not one of {', '.join(CATEGORIES)}"
            )
        if not item.get("kind"):
            errors.append(f"{item_id}: kind is required")
        unknown = set(item) - {
            "id",
            "name",
            "category",
            "kind",
            "cite",
            "requires",
            "aliases",
            "signature",
            "note",
        }
        if unknown:
            errors.append(f"{item_id}: unknown fields {sorted(unknown)}")
        _check_cite(item, sources, errors)
        _check_requires(language, item, errors)
    if [item.get("id") for item in items] != sorted(ids):
        errors.append("items must be sorted by id")
    present = {item.get("category") for item in items}
    for group, categories in REQUIRED_GROUPS.items():
        if not present & categories:
            errors.append(f"no item covers the required group {group!r}")
    _check_cells(
        language, data, ids, _known_cases() if cases is None else cases, errors
    )
    return errors


def dumps(data):
    """Canonical serialization: sorted keys, one item per line, so diffs stay reviewable."""
    head = {key: value for key, value in data.items() if key not in {"items", "cells"}}
    lines = ["{"]
    for key in sorted(head):
        lines.append(f"  {json.dumps(key)}: {json.dumps(head[key], sort_keys=True)},")
    lines.append('  "items": [')
    items = [json.dumps(item, sort_keys=True) for item in data["items"]]
    lines.extend(f"    {item}," for item in items[:-1])
    lines.append(f"    {items[-1]}")
    lines.append("  ],")
    cells = data.get("cells", {})
    if cells:
        lines.append('  "cells": {')
        rows = [
            f"    {json.dumps(key)}: {json.dumps(cells[key], sort_keys=True)}"
            for key in sorted(cells)
        ]
        lines.append(",\n".join(rows))
        lines.append("  }")
    else:
        lines.append('  "cells": {}')
    lines.append("}")
    return "\n".join(lines) + "\n"


def load(language, path=None):
    """Load and validate one language's inventory; raises InventoryError listing every violation."""
    path = Path(path) if path else DATA / f"{language}.json"
    data = json.loads(path.read_text())
    errors = validate(data, language=language)
    if errors:
        raise InventoryError(f"{path}:\n  " + "\n  ".join(errors))
    return Inventory(language, path, data)


def load_all():
    return {language: load(language) for language in LANGUAGES}
