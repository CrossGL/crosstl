"""Build the MSL inventory from the Metal Shading Language Specification 4.1 and the Metal toolchain's headers.

Two pinned inputs:

- the specification, which Apple publishes only as a PDF at an unversioned URL, so it is pinned by hash: the PDF
  (sha256 below) converted with ``pdftotext -layout`` (poppler 26.10.0), whose text output is pinned as well. Its
  numbered section table is read from the table of contents;
- the Metal standard library headers of the pinned Metal toolchain (``msl_stdlib``), pinned by a digest of every
  header's sha256. Every public function, member function, class, alias, enumeration, enumerator and library
  constant of namespace ``metal`` is derived from the compiler's AST of ``<metal_stdlib>`` for each language
  version, with the first version that declares it.

Library items cite the specification section that documents their header or class, and the header itself
(``metal-stdlib <header>``) where the specification has no such section. Types, qualifiers, attributes, address
spaces, statements, operators, stages and compiler options are listed by hand in ``curated/metal.tsv``, cited to a
section by its exact title; a title that is absent from the pinned text is an error.

Because neither input is downloadable at a versioned URL, CI validates the committed inventory but does not
re-derive it. On a Mac with the pinned toolchain:

    pdftotext -layout Metal-Shading-Language-Specification.pdf msl.txt
    python -m tests.conformance.inventory.msl --text msl.txt --check
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path

from . import DATA, InventoryError, curated, dumps, msl_stdlib, validate

PDF_URL = "https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf"
PDF_SHA256 = "41538b30d2f1140a5b2a0c84ce0a9f7b67bf0c707e224cfea0bfe5a44aa26cf5"
TEXT_SHA256 = "6b5e4c211e45ca42b1412fa4cb54029f1e414b15ec0b8dc35d12beef268ff8ee"
TOOLCHAIN_VERSION = "Apple metal version 32023.918 (metalfe-32023.918.1)"
HEADERS_SHA256 = "22be4c9e30e4a4feeab27b1872f82be59dde978cd583054c942fe7ecfecfb56e"
TOOLCHAIN_URL = "https://developer.apple.com/metal/"
COMPILER = ("xcrun", "-sdk", "macosx", "metal")
TOC_END = "Tables and Figures"
TOC_LINE = re.compile(r"^(\d+(?:\.\d+)*)\s+(.+?)(?:\s*\.{2,}\s*\d+)?\s*$")

# Specification sections that document the functions of a header.
FUNCTION_SECTIONS = {
    "metal_common": "Common Functions",
    "metal_integer": "Integer Functions",
    "metal_relational": "Relational Functions",
    "metal_math": "Math Functions",
    "metal_matrix": "Matrix Functions",
    "metal_simdgroup_matrix": "SIMD-Group Matrix Functions",
    "metal_geometric": "Geometric Functions",
    "metal_compute": "Threadgroup and SIMD-Group Synchronization Functions",
    "metal_simdgroup": "SIMD-Group Functions",
    "metal_quadgroup": "Quad-Group Functions",
    "metal_graphics": "Graphics Functions",
    "metal_interpolate": "Pull-Model Interpolation",
    "metal_texture": "Texture Functions",
    "metal_imageblocks": "Imageblock Functions",
    "metal_pack": "Pack and Unpack Functions",
    "metal_atomic": "Atomic Functions",
    "metal_command_buffer": "Encoding Commands for Indirect Command Buffers",
    "metal_raytracing": "Ray-Tracing Functions",
    "metal_curves": "Curve Utility Functions",
    "metal_logging": "Logging Functions",
    "metal_new": "Placement New",
    "metal_tensor": "Tensor Types",
    "metal_cooperative_tensor": "Cooperative Tensor Type",
    "metal_mesh": "Mesh Shader Types",
    "metal_visible_function_table": "Visible Function Table",
}
# Specification sections that document the types of a header.
TYPE_SECTIONS = {
    "metal_types": "Vector Data Types",
    "metal_extended_vector": "Vector Data Types",
    "metal_packed_vector": "Packed Vector Types",
    "metal_matrix": "Matrix Data Types",
    "metal_simdgroup_matrix": "SIMD-group Matrix Data Types",
    "metal_atomic": "Atomic Data Types",
    "metal_pixel": "Pixel Data Types",
    "metal_texture": "Textures",
    "metal_imageblocks": "Imageblocks",
    "metal_uniform": "Uniform Type",
    "metal_visible_function_table": "Visible Function Table",
    "metal_raytracing": "Ray-Tracing Types",
    "metal_curves": "Ray-Tracing Types",
    "metal_interpolate": "Interpolant Type",
    "metal_vertex_value": "Per-Vertex Values",
    "metal_mesh": "Mesh Shader Types",
    "metal_packed_numeric": "Packed Numeric Type",
    "metal_tensor": "Tensor Types",
    "metal_cooperative_tensor": "Cooperative Tensor Type",
}
# Classes with a section of their own; their member functions cite it too.
CLASS_SECTIONS = {
    "texture1d": "1D Texture",
    "texture1d_array": "1D Texture Array",
    "texture2d": "2D Texture",
    "texture2d_array": "2D Texture Array",
    "texture3d": "3D Texture",
    "texturecube": "Cube Texture",
    "texturecube_array": "Cube Texture Array",
    "texture2d_ms": "2D Multisampled Texture",
    "texture2d_ms_array": "2D Multisampled Texture Array",
    "depth2d": "2D Depth Texture",
    "depth2d_array": "2D Depth Texture Array",
    "depth2d_ms": "2D Multisampled Depth Texture",
    "depth2d_ms_array": "2D Multisampled Depth Texture Array",
    "depthcube": "Cube Depth Texture",
    "depthcube_array": "Cube Depth Texture Array",
    "texture_buffer": "Texture Buffer Functions",
    "sampler": "Samplers",
    "array": "Arrays of Textures, Texture Buffers, and Samplers",
    "array_ref": "Arrays of Textures, Texture Buffers, and Samplers",
    "raytracing.ray": "Ray Type",
    "raytracing.intersection_function_table": "Intersection Function Table",
    "raytracing.intersection_result": "Intersection Result Type",
    "raytracing.intersection_result_ref": "Intersection Result Reference Type",
    "raytracing.intersector": "Intersector Type",
    "raytracing.acceleration_structure": "Acceleration Structure Type",
    "raytracing.intersection_query": "Intersection Query Type",
    "mesh_grid_properties": "Mesh Grid Property Type",
    "mesh": "Mesh Type",
    "tensor": "Tensor Type",
    "cooperative_tensor": "Cooperative Tensor Type",
    "render_command": "Encoding Render Commands in Indirect Command Buffers",
    "compute_command": "Encoding Compute Commands in Indirect Command Buffers",
    # Enumerations and their enumerators cite the section that defines their values.
    "access": "Textures",
    "memory_order": "Memory Order",
    "thread_scope": "Thread Scope",
    "memory_coherence": "Memory Coherency",
    **{
        name: "Samplers"
        for name in (
            "address",
            "border_color",
            "compare_func",
            "coord",
            "filter",
            "mag_filter",
            "min_filter",
            "mip_filter",
            "r_address",
            "reduction",
            "s_address",
            "t_address",
        )
    },
}
# (id prefix, category, kind) per declaration kind; classes that are resources take the resource category.
SHAPES = {
    "function": ("function", "builtin-function", "library-function"),
    "member-function": ("method", "builtin-function", "member-function"),
    "class": ("class", "type", "library-class"),
    "alias": ("alias", "type", "library-alias"),
    "enum": ("enum", "type", "library-enum"),
    "enumerator": ("enumerator", "type", "enumerator"),
    "constant": ("constant", "builtin-variable", "library-constant"),
}


def table_of_contents(text):
    sections = {}
    for line in text.splitlines():
        if line.strip() == TOC_END:
            break
        match = TOC_LINE.match(line)
        if match and not re.match(r"^\d+ of \d+$", match.group(2)):
            sections.setdefault(
                match.group(1), re.sub(r"\s+", " ", match.group(2)).strip()
            )
    return sections


def _header_key(header):
    """Texture classes live in ``__bits/metal_<texture>``; they share ``metal_texture``'s sections."""
    return "metal_texture" if header.startswith("__bits/metal_") else header


def library_citation(kind, scope, name, headers):
    """The specification section title that documents a library declaration, or its header document."""
    owner = ".".join((*scope, name)) if kind in {"class", "enum"} else ".".join(scope)
    if kind != "function" and owner in CLASS_SECTIONS:
        return "section", CLASS_SECTIONS[owner]
    tables = (
        (FUNCTION_SECTIONS,)
        if kind in {"function", "member-function"}
        else (TYPE_SECTIONS, FUNCTION_SECTIONS)
    )
    for table in tables:
        for header in headers:
            title = table.get(_header_key(header))
            if title:
                return "section", title
    return "document", headers[0]


def _library_items(stdlib, by_title, cited, documents):
    items = []
    for (kind, scope, name), (version, headers) in sorted(stdlib.items()):
        prefix, category, item_kind = SHAPES[kind]
        if msl_stdlib.is_resource(kind, name):
            category = "resource"
        how, reference = library_citation(kind, scope, name, headers)
        if how == "section":
            number = by_title.get(reference)
            if number is None:
                raise InventoryError(
                    f"section {reference!r} does not exist in the pinned specification"
                )
            cited[number] = reference
            cite = f"msl-4.1 §{number}"
        else:
            documents[reference] = None
            cite = f"metal-stdlib {reference}"
        items.append(
            {
                "id": ".".join(("metal", prefix, *scope, name)),
                "name": "::".join(("metal", *scope, name)),
                "category": category,
                "kind": item_kind,
                "cite": cite,
                "requires": {"msl": version},
            }
        )
    return items


def derive(text, stdlib, toolchain, cells=None):
    toc = table_of_contents(text)
    by_title = {}
    for number, title in toc.items():
        by_title.setdefault(title, number)
    items, cited, documents = [], {}, {}
    for row in curated("metal"):
        number = by_title.get(row["reference"])
        if number is None:
            raise InventoryError(
                f"section {row['reference']!r} does not exist in the pinned specification"
            )
        cited[number] = toc[number]
        item = {key: row[key] for key in ("id", "name", "category", "kind")}
        item["cite"] = f"msl-4.1 §{number}"
        if row["note"]:
            item["note"] = row["note"]
        items.append(item)
    items += _library_items(stdlib, by_title, cited, documents)
    items.sort(key=lambda item: item["id"])
    duplicates = sorted(
        name
        for name, count in Counter(item["id"] for item in items).items()
        if count > 1
    )
    if duplicates:
        raise InventoryError(f"duplicate items: {', '.join(duplicates)}")
    cells = dict(cells or {})
    orphaned = sorted(set(cells) - {item["id"] for item in items})
    if orphaned:
        raise InventoryError(f"classified items vanished: {', '.join(orphaned)}")
    return {
        "schemaVersion": 1,
        "language": "metal",
        "title": "Metal Shading Language 4.1",
        "sources": {
            "metal-stdlib": {
                "title": "Metal standard library headers of the pinned Metal toolchain",
                "url": TOOLCHAIN_URL,
                "version": toolchain["version"],
                "documents": {
                    name: toolchain["headers"][name] for name in sorted(documents)
                },
            },
            "msl-4.1": {
                "title": "Metal Shading Language Specification, Version 4.1",
                "url": PDF_URL,
                "version": "4.1",
                "sections": dict(
                    sorted(
                        cited.items(), key=lambda kv: [int(p) for p in kv[0].split(".")]
                    )
                ),
            },
        },
        "generated": {
            "by": "tests/conformance/inventory/msl.py",
            "pdfSha256": PDF_SHA256,
            "textSha256": TEXT_SHA256,
            "converter": "pdftotext -layout (poppler 26.10.0)",
            "toolchain": toolchain["version"],
            "headersSha256": toolchain["digest"],
            "standards": [standard for standard, _ in msl_stdlib.STANDARDS],
            "scope": (
                "public declarations of namespace metal per language version, by name (overloads and template "
                "specializations are one item); minimum versions are those of the macOS compiler"
            ),
        },
        "items": items,
        "cells": cells,
    }


def toolchain(compiler=COMPILER):
    """Version, header digests and the derived library of the Metal compiler, refusing an unpinned toolchain."""
    version = subprocess.run(
        [*compiler, "--version"], capture_output=True, text=True, check=False
    ).stdout.splitlines()
    version = version[0].strip() if version else ""
    root = msl_stdlib.include_root(compiler)
    headers = msl_stdlib.header_digests(root)
    digest = msl_stdlib.tree_digest(headers)
    if (version, digest) != (TOOLCHAIN_VERSION, HEADERS_SHA256):
        raise InventoryError(
            f"Metal toolchain {version!r} with headers {digest} is not the pinned {TOOLCHAIN_VERSION!r} "
            f"({HEADERS_SHA256}); after reviewing the change, update TOOLCHAIN_VERSION and HEADERS_SHA256"
        )
    return (
        {"version": version, "digest": digest, "headers": headers},
        msl_stdlib.library(compiler, root),
    )


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--text",
        type=Path,
        required=True,
        help="pdftotext -layout output of the pinned PDF",
    )
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--write", action="store_true")
    action.add_argument("--check", action="store_true")
    parser.add_argument("--output", type=Path, default=DATA / "metal.json")
    args = parser.parse_args(argv)
    data = args.text.read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    if digest != TEXT_SHA256:
        raise InventoryError(
            f"{args.text} has sha256 {digest}, not the pinned {TEXT_SHA256}"
        )
    existing = json.loads(args.output.read_text()) if args.output.is_file() else {}
    pinned, stdlib = toolchain()
    inventory = derive(data.decode(), stdlib, pinned, existing.get("cells"))
    errors = validate(inventory, language="metal")
    if errors:
        raise InventoryError("\n".join(errors))
    text = dumps(inventory)
    if args.write:
        args.output.write_text(text)
        print(f"Wrote {args.output}: {len(inventory['items'])} items")
        return 0
    if args.output.is_file() and args.output.read_text() == text:
        print(
            f"{args.output} matches the pinned specification and toolchain ({len(inventory['items'])} items)"
        )
        return 0
    print(f"{args.output} is out of date; regenerate with --write", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
