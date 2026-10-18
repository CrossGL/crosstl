"""Derive the HLSL inventory (HLSL 2021, Shader Model 6.0 and later) from pinned upstream sources.

HLSL has no single complete specification, so the inventory combines three pinned sources and cites each item to
the most authoritative one that defines it:

- ``hlsl-spec``: the draft HLSL language specification (microsoft/hlsl-specs), cited by its stable section labels
  such as ``[Decl.cbuffer]``; the label table is read from the pinned LaTeX sources;
- ``hlsl-docs``: the HLSL reference in MicrosoftDocs/win32, cited by page; the page list is the pinned
  ``desktop-src/direct3dhlsl`` directory, and the intrinsic-function table supplies minimum shader models;
- ``directx-specs``: the Shader Model feature specifications in microsoft/DirectX-Specs, cited by document;
- ``dxc``: the reference compiler's declaration tables at the pinned DXC release (``gen_intrin_main.txt`` for
  every intrinsic and object method, ``DxilConstants.h`` for system-value semantics and shader kinds), cited where
  the reference does not document a construct.

Intrinsics, object methods, resource objects, system-value semantics and stages are derived from those tables.
Types, qualifiers, attributes, declarations, statements, operators and preprocessor directives are listed by hand
in ``curated/hlsl.tsv``; every citation they make must resolve to a label or page that exists in the pinned
sources.

    python -m tests.conformance.inventory.hlsl --fetch --check
    python -m tests.conformance.inventory.hlsl --inputs-dir DIR --write
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

SPEC = (
    "microsoft/hlsl-specs",
    "5c1aae7ef249efba8a993ca50d69430c379ac1a1",
    "specs/language",
)
DOCS = (
    "MicrosoftDocs/win32",
    "ca8274decead49403d70ceeed4af8d4ca74eaf11",
    "desktop-src/direct3dhlsl",
)
DXC = (
    "microsoft/DirectXShaderCompiler",
    "d355aa8364d34df3f0822ba0de8d1dfc75ae6f48",
    "",
)  # v1.9.2602.24
SPECS = ("microsoft/DirectX-Specs", "71e431fa0f713818c6a3a66f49ab0e6732712d15", "d3d")
SPEC_CHAPTERS = (
    "introduction",
    "lex",
    "basic",
    "conversions",
    "expressions",
    "statements",
    "declarations",
    "overloading",
    "resources",
)
INPUTS = {
    **{f"{name}.tex": (SPEC, f"{SPEC[2]}/{name}.tex") for name in SPEC_CHAPTERS},
    "dx-graphics-hlsl-intrinsic-functions.md": (
        DOCS,
        f"{DOCS[2]}/dx-graphics-hlsl-intrinsic-functions.md",
    ),
    "dx-graphics-hlsl-semantics.md": (DOCS, f"{DOCS[2]}/dx-graphics-hlsl-semantics.md"),
    "docs-listing.txt": (
        DOCS,
        None,
    ),  # sorted file names in DOCS[2], from the git tree API
    "directx-specs-listing.txt": (SPECS, None),  # sorted file names in SPECS[2]
    "gen_intrin_main.txt": (DXC, "utils/hct/gen_intrin_main.txt"),
    "DxilConstants.h": (DXC, "include/dxc/DXIL/DxilConstants.h"),
}
SHA256 = {
    "introduction.tex": (
        "038fb7fcc3ddafe316ff2a8e3e79245f88fcc25a960497bf12a52826a8f1130b"
    ),
    "lex.tex": "a126374dfcf44e2193ec3f4f5f6980f0ad8731ea6c22bf7be2cc59c19beda029",
    "basic.tex": "e7f4e44c4cdb9219163a8d0675f882ddf0e937a11abc7f91cc0553815850f11a",
    "conversions.tex": (
        "f0175d665b978854e9bbea12ba6b600d31798e783e6d8c8a52fb84979dc4dfd7"
    ),
    "expressions.tex": (
        "30471e31c8b4bb340e750207ffe5aa3ae8663c2cb14382d91d82e3af5ae8eee0"
    ),
    "statements.tex": (
        "fe7a7fa0a52ed04e90c66aa7d7a7b7aeee0a458acc2a17b90a1690585d3fb3d9"
    ),
    "declarations.tex": (
        "8fcb3b228472a321bb4cd90e0c2b09583187e9a3ec24362f84aaf9e35d86c23a"
    ),
    "overloading.tex": (
        "ac871c68c7c91b1a2e0549bda7ba7873d59c1a1d26f9894f0f7f183570c0ecef"
    ),
    "resources.tex": "dbe5844510ee0da82f0087d74565c005ae3466a0d332130ab73459d2b12db26d",
    "dx-graphics-hlsl-intrinsic-functions.md": (
        "7c492e956192abc5436d710238e14cdcaa3cc7ca442395780a8fa30237014464"
    ),
    "dx-graphics-hlsl-semantics.md": (
        "994564fd2864e8c34531960a33b0bd4286cab2bf1cb7b61e14d6421acca883cd"
    ),
    "docs-listing.txt": (
        "3ea267500972ff2d4159c5fb8fb9abeb21119e27956b30642f6fec8ae683b80c"
    ),
    "directx-specs-listing.txt": (
        "7515cb83560e975e639937824f1c57daaa1b7a9566f232100eefc03214738741"
    ),
    "gen_intrin_main.txt": (
        "24bc62cd69825c062f6527b783d2089c669c643b2df9336a6b5c7895a0ad7eec"
    ),
    "DxilConstants.h": (
        "ffe701558a5e65dcefc2693db1b3ea1f418798890d5d6e7f709d9284f6212599"
    ),
}
DOCS_URL = "https://learn.microsoft.com/windows/win32/direct3dhlsl/"
# DXC's internal object names -> HLSL spelling.
OBJECTS = {
    "TextureCUBE": "TextureCube",
    "TextureCUBEArray": "TextureCubeArray",
    "Stream": "StreamOutputObject",
    "DxHitObject": "dx::HitObject",
    "VkSubpassInput": "vk::SubpassInput",
    "VkSubpassInputMS": "vk::SubpassInputMS",
    "BufferPointer": "vk::BufferPointer",
}
SHADER_KINDS_SKIPPED = {"Library", "Invalid"}
SEMANTICS_SKIPPED = {"Arbitrary", "Invalid"}


def spec_labels(texts):
    """Stable labels of the draft specification, numbered the way its LaTeX build numbers them."""
    labels = {}
    for chapter, name in enumerate(SPEC_CHAPTERS, start=1):
        counters = [chapter, 0, 0, 0]
        for line in texts[f"{name}.tex"].splitlines():
            match = re.match(r"^\\(Ch|Sec|Sub|SubSub)\{(.*)\}\{([^{}]+)\}\s*$", line)
            if not match:
                continue
            kind, title, label = match.groups()
            title = re.sub(
                r"\\acr(?:long|short)\{(\w+)\}", lambda m: m.group(1).upper(), title
            )
            depth = {"Ch": 0, "Sec": 1, "Sub": 2, "SubSub": 3}[kind]
            if depth:
                counters[depth] += 1
                counters[depth + 1 :] = [0] * (3 - depth)
            number = ".".join(str(part) for part in counters[: depth + 1])
            labels[label] = f"{number} {title}"
    return labels


def intrinsic_table(text):
    """Intrinsic names, their documentation page and minimum shader model, from the reference table."""
    rows = {}
    for name, page, model in re.findall(
        r"^\| \[\*\*(\w+)\*\*\]\(([^)]+)\)\s*\|[^|]*\|\s*([^|]*?)\s*\|", text, re.M
    ):
        digits = re.match(r"\d+(?:\.\d+)?", model)
        version = None
        if digits:
            version = (
                digits.group(0) if "." in digits.group(0) else f"{digits.group(0)}.0"
            )
        rows[name] = (Path(page).stem, version)
    return rows


def dxc_declarations(text):
    """(namespace, name) -> attribute strings of every overload in gen_intrin_main.txt."""
    namespace, found = None, {}
    for line in text.splitlines():
        match = re.match(r"^namespace (\w+) \{", line)
        if match:
            namespace = match.group(1)
            continue
        if line.startswith("} namespace"):
            namespace = None
            continue
        code = re.sub(r"//.*", "", line).strip()
        if not code or namespace is None:
            continue
        match = re.match(r"^(.*?)\s+\[\[([^\]]*)\]\]\s+(\w+)\s*\(", code)
        if not match:
            raise InventoryError(
                f"unparsed declaration in namespace {namespace}: {code!r}"
            )
        found.setdefault((namespace, match.group(3)), []).append(match.group(2))
    return found


def enum_members(text, name):
    match = re.search(rf"enum class {name}\b[^{{]*\{{(.*?)\}};", text, re.S)
    if not match:
        raise InventoryError(f"DxilConstants.h no longer defines enum {name}")
    members = []
    for line in match.group(1).splitlines():
        line = re.sub(r"//.*", "", line).strip().rstrip(",")
        member = re.match(r"^(\w+)(?:\s*=.*)?$", line)
        if member and not member.group(1).startswith("Last"):
            members.append(member.group(1))
    return members


def documented_semantics(text):
    return {
        name.replace("\\", "").lower(): name.replace("\\", "")
        for name in re.findall(r"^\| \[?(SV\\?_\w+)", text, re.M)
    }


def _min_shader_model(attributes):
    models = [re.search(r"min_sm=(\d+\.\d+)", attribute) for attribute in attributes]
    found = sorted(
        {match.group(1) for match in models if match},
        key=lambda v: [int(p) for p in v.split(".")],
    )
    return found[0] if found else None


class _Builder:
    """Collects items and checks every citation against the pinned sources."""

    def __init__(self, texts):
        self.labels = spec_labels(texts)
        self.pages = {
            Path(line.strip()).stem
            for line in texts["docs-listing.txt"].splitlines()
            if line.strip().endswith(".md")
        }
        self.specs = {
            Path(line.strip()).stem
            for line in texts["directx-specs-listing.txt"].splitlines()
            if line.strip()
        }
        self.items = {}
        self.cited = {
            "hlsl-spec": set(),
            "hlsl-docs": set(),
            "directx-specs": set(),
            "dxc": set(),
        }

    def check(self, citation):
        source, reference = citation.split(" ", 1)
        known = {
            "hlsl-spec": lambda: reference.strip("[]") in self.labels,
            "hlsl-docs": lambda: reference in self.pages,
            "directx-specs": lambda: reference in self.specs,
            "dxc": lambda: reference.split("#", 1)[0] in DXC_DOCUMENTS,
        }
        if source not in known or not known[source]():
            raise InventoryError(
                f"citation {citation!r} does not resolve in the pinned sources"
            )
        self.cited[source].add(
            reference.strip("[]") if source == "hlsl-spec" else reference
        )
        return citation

    def add(self, item_id, name, category, kind, citation, requires=None, note=None):
        if item_id in self.items:
            raise InventoryError(f"duplicate item {item_id}")
        item = {
            "id": item_id,
            "name": name,
            "category": category,
            "kind": kind,
            "cite": self.check(citation),
        }
        if requires:
            item["requires"] = requires
        if note:
            item["note"] = note
        self.items[item_id] = item

    def page(self, *candidates):
        return next(
            (candidate for candidate in candidates if candidate in self.pages), None
        )


DXC_DOCUMENTS = {
    "gen_intrin_main.txt": "utils/hct/gen_intrin_main.txt",
    "DxilConstants.h": "include/dxc/DXIL/DxilConstants.h",
    "docs/SPIR-V.rst": "docs/SPIR-V.rst",
}


def _intrinsics(builder, declarations, documented):
    omitted = 0
    names = {name for (namespace, name) in declarations if namespace == "Intrinsics"}
    for name in sorted(names | set(documented)):
        if name.startswith("__builtin_"):
            omitted += 1  # DXC-internal builtins behind library headers, not part of the language
            continue
        attributes = declarations.get(("Intrinsics", name), [])
        page, docs_model = documented.get(name, (None, None))
        page = (
            page
            if page in builder.pages
            else builder.page(name.lower(), f"dx-graphics-hlsl-{name.lower()}")
        )
        model = _min_shader_model(attributes) or docs_model
        builder.add(
            f"hlsl.intrinsic.{name}",
            name,
            "builtin-function",
            "intrinsic",
            f"hlsl-docs {page}" if page else "dxc gen_intrin_main.txt#Intrinsics",
            {"sm": model} if model else None,
            None if attributes else "documented but not declared by DXC",
        )
    return omitted


def _namespaced(builder, declarations):
    """vk:: and dx:: intrinsics, resource objects and their methods."""
    for (namespace, name), attributes in sorted(declarations.items()):
        if namespace == "Intrinsics":
            continue
        model = _min_shader_model(attributes)
        requires = {"sm": model} if model else None
        if namespace in {"VkIntrinsics", "DxIntrinsics"}:
            vulkan = namespace == "VkIntrinsics"
            builder.add(
                f"hlsl.vk.{name}" if vulkan else f"hlsl.intrinsic.dx-{name}",
                f"{'vk' if vulkan else 'dx'}::{name}",
                "extension" if vulkan else "builtin-function",
                "vk-intrinsic" if vulkan else "intrinsic",
                f"dxc gen_intrin_main.txt#{namespace}",
                requires,
            )
            continue
        if not namespace.endswith("Methods"):
            raise InventoryError(f"unexpected DXC namespace {namespace}")
        raw = namespace[: -len("Methods")]
        obj = OBJECTS.get(raw, raw)
        slug = re.sub(r"\W+", "-", obj).strip("-")
        vulkan = obj.startswith("vk::")
        if f"hlsl.object.{slug}" not in builder.items:
            page = builder.page(f"sm5-object-{raw.lower()}")
            builder.add(
                f"hlsl.object.{slug}",
                obj,
                "extension" if vulkan else "resource",
                "object",
                f"hlsl-docs {page}" if page else f"dxc gen_intrin_main.txt#{namespace}",
            )
        page = builder.page(
            f"sm5-object-{raw.lower()}-{name.lower()}",
            f"dx-graphics-hlsl-to-{name.lower()}",
        )
        builder.add(
            f"hlsl.method.{slug}.{name}",
            f"{obj}::{name}",
            "extension" if vulkan else "builtin-function",
            "method",
            f"hlsl-docs {page}" if page else f"dxc gen_intrin_main.txt#{namespace}",
            requires,
        )


def _semantics_and_stages(builder, texts):
    constants = texts["DxilConstants.h"]
    documented = documented_semantics(texts["dx-graphics-hlsl-semantics.md"])
    for kind in enum_members(constants, "SemanticKind"):
        if kind in SEMANTICS_SKIPPED:
            continue
        key = f"sv_{kind.lower()}"
        page = builder.page(f"sv-{kind.lower()}")
        if page:
            citation = f"hlsl-docs {page}"
        elif key in documented:
            citation = "hlsl-docs dx-graphics-hlsl-semantics"
        else:
            citation = "dxc DxilConstants.h#SemanticKind"
        name = documented.get(key, f"SV_{kind}")
        builder.add(
            f"hlsl.semantic.{name}", name, "builtin-variable", "system-value", citation
        )
    for kind in enum_members(constants, "ShaderKind"):
        if kind not in SHADER_KINDS_SKIPPED:
            builder.add(
                f"hlsl.stage.{kind.lower()}",
                kind.lower(),
                "stage",
                "shader-kind",
                "dxc DxilConstants.h#ShaderKind",
            )


def _sources(builder):
    spec_repo, spec_commit, _ = SPEC
    docs_repo, docs_commit, docs_dir = DOCS
    dxc_repo, dxc_commit, _ = DXC
    specs_repo, specs_commit, specs_dir = SPECS
    cited = {source: sorted(references) for source, references in builder.cited.items()}
    return {
        "hlsl-spec": {
            "title": "HLSL Specification (draft), microsoft/hlsl-specs",
            "url": f"https://github.com/{spec_repo}/tree/{spec_commit}/specs/language",
            "labels": {label: builder.labels[label] for label in cited["hlsl-spec"]},
        },
        "hlsl-docs": {
            "title": "HLSL reference, Microsoft Learn (MicrosoftDocs/win32)",
            "url": f"https://github.com/{docs_repo}/tree/{docs_commit}/{docs_dir}",
            "documents": {page: f"{DOCS_URL}{page}" for page in cited["hlsl-docs"]},
        },
        "directx-specs": {
            "title": "DirectX Specs: Shader Model feature specifications",
            "url": f"https://github.com/{specs_repo}/tree/{specs_commit}/{specs_dir}",
            "documents": {
                name: (
                    f"https://github.com/{specs_repo}/blob/{specs_commit}/{specs_dir}/{name}.md"
                )
                for name in cited["directx-specs"]
            },
        },
        "dxc": {
            "title": "DirectX Shader Compiler v1.9.2602.24 declaration tables",
            "url": f"https://github.com/{dxc_repo}/tree/{dxc_commit}",
            "documents": {
                reference: (
                    f"https://github.com/{dxc_repo}/blob/{dxc_commit}/"
                    + DXC_DOCUMENTS[reference.split("#", 1)[0]]
                )
                for reference in cited["dxc"]
            },
        },
    }


def derive(texts, cells=None):
    builder = _Builder(texts)
    declarations = dxc_declarations(texts["gen_intrin_main.txt"])
    omitted = _intrinsics(
        builder,
        declarations,
        intrinsic_table(texts["dx-graphics-hlsl-intrinsic-functions.md"]),
    )
    _namespaced(builder, declarations)
    _semantics_and_stages(builder, texts)
    for row in curated("hlsl"):
        builder.add(
            row["id"],
            row["name"],
            row["category"],
            row["kind"],
            row["reference"],
            note=row["note"] or None,
        )
    cells = dict(cells or {})
    orphaned = sorted(set(cells) - set(builder.items))
    if orphaned:
        raise InventoryError(
            f"classified items vanished from the sources: {', '.join(orphaned)}"
        )
    return {
        "schemaVersion": 1,
        "language": "hlsl",
        "title": "HLSL 2021 for Shader Model 6.0 and later",
        "sources": _sources(builder),
        "generated": {
            "by": "tests/conformance/inventory/hlsl.py",
            "inputs": {
                name: {
                    "repository": repo,
                    "commit": commit,
                    "path": path,
                    "sha256": SHA256[name],
                }
                for name, ((repo, commit, _), path) in sorted(INPUTS.items())
            },
            "omitted": {"internal DXC builtins": omitted},
        },
        "items": [builder.items[item_id] for item_id in sorted(builder.items)],
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
    for name, ((repository, commit, _), path) in INPUTS.items():
        if path:
            (Path(directory) / name).write_bytes(
                download(
                    f"https://raw.githubusercontent.com/{repository}/{commit}/{path}"
                )
            )
    repository, commit, subdirectory = DOCS
    tree = json.loads(
        download(f"https://api.github.com/repos/{repository}/git/trees/{commit}")
    )
    for part in subdirectory.split("/"):
        sha = next(entry["sha"] for entry in tree["tree"] if entry["path"] == part)
        tree = json.loads(
            download(f"https://api.github.com/repos/{repository}/git/trees/{sha}")
        )
    names = sorted(entry["path"] for entry in tree["tree"] if entry["type"] == "blob")
    (Path(directory) / "docs-listing.txt").write_text("\n".join(names) + "\n")
    repository, commit, subdirectory = SPECS
    listing = json.loads(
        download(
            f"https://api.github.com/repos/{repository}/contents/{subdirectory}?ref={commit}"
        )
    )
    names = sorted(entry["name"] for entry in listing if entry["type"] == "file")
    (Path(directory) / "directx-specs-listing.txt").write_text("\n".join(names) + "\n")


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
    parser.add_argument("--output", type=Path, default=DATA / "hlsl.json")
    args = parser.parse_args(argv)
    with tempfile.TemporaryDirectory() as scratch:
        directory = args.inputs_dir
        if args.fetch:
            directory = Path(scratch)
            fetch(directory)
        texts = read_inputs(directory)
    existing = json.loads(args.output.read_text()) if args.output.is_file() else {}
    data = derive(texts, existing.get("cells"))
    errors = validate(data, language="hlsl")
    if errors:
        raise InventoryError("\n".join(errors))
    text = dumps(data)
    if args.write:
        args.output.write_text(text)
        print(f"Wrote {args.output}: {len(data['items'])} items")
        return 0
    if args.output.is_file() and args.output.read_text() == text:
        print(f"{args.output} matches the pinned sources ({len(data['items'])} items)")
        return 0
    print(f"{args.output} is out of date; regenerate with --write", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
