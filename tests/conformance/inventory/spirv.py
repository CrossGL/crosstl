"""Derive the SPIR-V inventory from the machine-readable grammar at a pinned SPIRV-Headers commit.

Every core instruction, every enumerant of the value and bit enumerations, every GLSL.std.450 extended
instruction and every extension named by the grammar becomes one item, provided a Vulkan environment can use it.
Instructions and enumerants are cited to their section of the unified SPIR-V specification at the grammar's
revision; instructions the specification excludes (``@exclude``) and extensions are cited to their extension
document in the SPIR-V registry.

The scope is SPIR-V as Vulkan consumes it (the Vulkan specification's appendix "Vulkan Environment for SPIR-V").
The pinned ``vk.xml`` lists every SPIR-V capability and extension a Vulkan implementation may accept. An item is
kept when it needs no capability or one of its enabling capabilities is available (listed, or implicitly declared
by a listed one), and when it is either part of a core SPIR-V version or enabled by a listed extension. Enumerations
are kept only when an admitted instruction or enumerant can take them. Kernel (OpenCL) and other non-Vulkan
constructs are therefore omitted; their number is recorded under ``generated``.

    python -m tests.conformance.inventory.spirv --fetch --check   # CI: the committed file matches the inputs
    python -m tests.conformance.inventory.spirv --inputs-dir DIR --write

Regenerating keeps every classified cell; an item that disappears while it still has cells is an error. A grammar
kind or instruction class without a category below is an error too, so a grammar update cannot add items silently.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tempfile
import xml.etree.ElementTree as ElementTree
from pathlib import Path

from . import DATA, InventoryError, download, dumps, validate

CORE = "spirv.core.grammar.json"
GLSL450 = "extinst.glsl.std.450.grammar.json"
VKXML = "vk.xml"
# Input -> (repository, commit, path, sha256). SPIRV-Headers is the commit SPIRV-Tools v2026.4 pins; vk.xml is
# Vulkan-Headers v1.4.363, the registry of the Vulkan SDK in TOOLCHAINS.md.
INPUTS = {
    CORE: (
        "KhronosGroup/SPIRV-Headers",
        "496543121ce6419f23d6fa5d7194ba66c36212d2",
        "include/spirv/unified1/spirv.core.grammar.json",
        "c125a0fbf6730fde4625fef2618c4f7611ab42edf9b2ad5bb9291002b10bfad3",
    ),
    GLSL450: (
        "KhronosGroup/SPIRV-Headers",
        "496543121ce6419f23d6fa5d7194ba66c36212d2",
        "include/spirv/unified1/extinst.glsl.std.450.grammar.json",
        "5d72e48247569f9b77a17a849db624350abc9d26f7f72b153bbcf15c37238239",
    ),
    VKXML: (
        "KhronosGroup/Vulkan-Headers",
        "6802bb4733b63ed5efd3adb308a6c885ef180ea1",
        "registry/vk.xml",
        "55ec60950cfb18c3575dcf5fd52741b2bb70eb1e466408f803a91049973ee6fb",
    ),
}
SPEC_URL = "https://registry.khronos.org/SPIR-V/specs/unified1/SPIRV.html"
GLSL450_URL = "https://registry.khronos.org/SPIR-V/specs/unified1/GLSL.std.450.html"
REGISTRY_URL = "https://github.khronos.org/SPIRV-Registry/"

# Instruction class -> (section of the unified specification, title, category).
CLASSES = {
    "Miscellaneous": ("3.3.1", "Miscellaneous Instructions", "operator"),
    "Debug": ("3.3.2", "Debug Instructions", "declaration"),
    "Annotation": ("3.3.3", "Annotation Instructions", "qualifier"),
    "Extension": ("3.3.4", "Extension Instructions", "extension"),
    "Mode-Setting": ("3.3.5", "Mode-Setting Instructions", "declaration"),
    "Type-Declaration": ("3.3.6", "Type-Declaration Instructions", "type"),
    "Constant-Creation": ("3.3.7", "Constant-Creation Instructions", "declaration"),
    "Memory": ("3.3.8", "Memory Instructions", "operator"),
    "Function": ("3.3.9", "Function Instructions", "declaration"),
    "Image": ("3.3.10", "Image Instructions", "builtin-function"),
    "Conversion": ("3.3.11", "Conversion Instructions", "operator"),
    "Composite": ("3.3.12", "Composite Instructions", "operator"),
    "Arithmetic": ("3.3.13", "Arithmetic Instructions", "operator"),
    "Bit": ("3.3.14", "Bit Instructions", "operator"),
    "Relational_and_Logical": (
        "3.3.15",
        "Relational and Logical Instructions",
        "operator",
    ),
    "Derivative": ("3.3.16", "Derivative Instructions", "builtin-function"),
    "Control-Flow": ("3.3.17", "Control-Flow Instructions", "statement"),
    "Atomic": ("3.3.18", "Atomic Instructions", "builtin-function"),
    "Primitive": ("3.3.19", "Primitive Instructions", "builtin-function"),
    "Barrier": ("3.3.20", "Barrier Instructions", "builtin-function"),
    "Group": ("3.3.21", "Group and Subgroup Instructions", "builtin-function"),
    "Device-Side_Enqueue": (
        "3.3.22",
        "Device-Side Enqueue Instructions",
        "builtin-function",
    ),
    "Pipe": ("3.3.23", "Pipe Instructions", "builtin-function"),
    "Non-Uniform": ("3.3.24", "Non-Uniform Instructions", "builtin-function"),
    "Tensor": ("3.3.25", "Tensor Instructions", "builtin-function"),
    "Graph": ("3.3.26", "Graph Instructions", "builtin-function"),
    "Reserved": ("3.3.27", "Reserved Instructions", "builtin-function"),
    "@exclude": (None, None, "builtin-function"),
}
# Instructions whose role differs from their class.
INSTRUCTION_CATEGORY = {
    "OpEntryPoint": "stage",
    "OpExecutionMode": "qualifier",
    "OpExecutionModeId": "qualifier",
    "OpCapability": "extension",
    "OpVariable": "declaration",
    "OpFunctionCall": "operator",
}
# Prefixes that identify the role of extension instructions filed under Reserved or @exclude.
PREFIX_CATEGORY = (
    ("OpType", "type"),
    ("OpConstant", "declaration"),
    ("OpSpecConstant", "declaration"),
    ("OpDecorate", "qualifier"),
    ("OpMemberDecorate", "qualifier"),
    ("OpExecutionMode", "qualifier"),
)
# Operand kind -> (section, title, category). Kinds set to None are not language constructs.
KINDS = {
    "SourceLanguage": None,  # OpSource debug metadata
    "KernelProfilingInfo": (
        None
    ),  # OpenCL profiling queries, not a construct of any shading language
    "ExecutionModel": ("3.2.2", "Execution Model", "stage"),
    "AddressingModel": ("3.2.3", "Addressing Model", "declaration"),
    "MemoryModel": ("3.2.4", "Memory Model", "declaration"),
    "ExecutionMode": ("3.2.5", "Execution Mode", "qualifier"),
    "StorageClass": ("3.2.6", "Storage Class", "resource"),
    "Dim": ("3.2.7", "Dim", "type"),
    "SamplerAddressingMode": ("3.2.8", "Sampler Addressing Mode", "resource"),
    "SamplerFilterMode": ("3.2.9", "Sampler Filter Mode", "resource"),
    "ImageFormat": ("3.2.10", "Image Format", "resource"),
    # Values only OpImageQueryOrder and OpImageQueryFormat return; both require the Kernel capability.
    "ImageChannelOrder": None,
    "ImageChannelDataType": None,
    "ImageOperands": ("3.2.13", "Image Operands", "builtin-function"),
    "FPFastMathMode": ("3.2.14", "FP Fast Math Mode", "qualifier"),
    "FPRoundingMode": ("3.2.15", "FP Rounding Mode", "qualifier"),
    "LinkageType": ("3.2.16", "Linkage Type", "qualifier"),
    "AccessQualifier": ("3.2.17", "Access Qualifier", "qualifier"),
    "FunctionParameterAttribute": (
        "3.2.18",
        "Function Parameter Attribute",
        "qualifier",
    ),
    "Decoration": ("3.2.19", "Decoration", "qualifier"),
    "BuiltIn": ("3.2.20", "BuiltIn", "builtin-variable"),
    "SelectionControl": ("3.2.21", "Selection Control", "statement"),
    "LoopControl": ("3.2.22", "Loop Control", "statement"),
    "FunctionControl": ("3.2.23", "Function Control", "qualifier"),
    "MemorySemantics": ("3.2.24", "Memory Semantics <id>", "qualifier"),
    "MemoryAccess": ("3.2.25", "Memory Operands", "qualifier"),
    "Scope": ("3.2.26", "Scope <id>", "qualifier"),
    "GroupOperation": ("3.2.27", "Group Operation", "builtin-function"),
    "KernelEnqueueFlags": ("3.2.28", "Kernel Enqueue Flags", "builtin-function"),
    "Capability": ("3.2.30", "Capability", "extension"),
    "RayFlags": ("3.2.31", "Ray Flags", "builtin-function"),
    "RayQueryIntersection": ("3.2.32", "Ray Query Intersection", "builtin-function"),
    "RayQueryCommittedIntersectionType": (
        "3.2.33",
        "Ray Query Committed Type",
        "builtin-function",
    ),
    "RayQueryCandidateIntersectionType": (
        "3.2.34",
        "Ray Query Candidate Type",
        "builtin-function",
    ),
    "FragmentShadingRate": ("3.2.35", "Fragment Shading Rate", "builtin-variable"),
    "FPDenormMode": ("3.2.36", "FP Denorm Mode", "qualifier"),
    "FPOperationMode": ("3.2.37", "FP Operation Mode", "qualifier"),
    "QuantizationModes": ("3.2.38", "Quantization Mode", "qualifier"),
    "OverflowModes": ("3.2.39", "Overflow Mode", "qualifier"),
    "PackedVectorFormat": ("3.2.40", "Packed Vector Format", "operator"),
    "CooperativeMatrixOperands": (
        "3.2.41",
        "Cooperative Matrix Operands",
        "builtin-function",
    ),
    "CooperativeMatrixLayout": (
        "3.2.42",
        "Cooperative Matrix Layout",
        "builtin-function",
    ),
    "CooperativeMatrixUse": ("3.2.43", "Cooperative Matrix Use", "builtin-function"),
    "CooperativeMatrixReduce": (
        "3.2.44",
        "Cooperative Matrix Reduce Mode",
        "builtin-function",
    ),
    "TensorClampMode": ("3.2.45", "Tensor Clamp Mode", "builtin-function"),
    "TensorAddressingOperands": (
        "3.2.46",
        "Tensor Addressing Operands",
        "builtin-function",
    ),
    "TensorOperands": ("3.2.47", "Tensor Operands", "builtin-function"),
    "InitializationModeQualifier": (
        "3.2.48",
        "Initialization Mode Qualifier",
        "qualifier",
    ),
    "HostAccessQualifier": ("3.2.49", "Host Access Qualifier", "qualifier"),
    "LoadCacheControl": ("3.2.50", "Load Cache Control", "qualifier"),
    "StoreCacheControl": ("3.2.51", "Store Cache Control", "qualifier"),
    "NamedMaximumNumberOfRegisters": (
        "3.2.52",
        "Named Maximum Number of Registers",
        "qualifier",
    ),
    "MatrixMultiplyAccumulateOperands": (
        "3.2.53",
        "Matrix Multiply Accumulate Operands",
        "builtin-function",
    ),
    "RawAccessChainOperands": ("3.2.54", "Raw Access Chain Operands", "operator"),
    "FPEncoding": ("3.2.55", "FP Encoding", "type"),
    "CooperativeVectorMatrixLayout": (
        "3.2.56",
        "Cooperative Vector Matrix Layout",
        "builtin-function",
    ),
    "ComponentType": (
        "3.2.57",
        "Cooperative Vector Matrix Component Type",
        "builtin-function",
    ),
    "GatherModes": ("3.2.58", "Gather Modes", "builtin-function"),
}
ENUMERATIONS = ("ValueEnum", "BitEnum")


def _requires(entry):
    requires = {}
    version = entry.get("version")
    if version:
        requires["version"] = "none" if version == "None" else version
    for key in ("capabilities", "extensions"):
        if entry.get(key):
            requires[key] = sorted(set(entry[key]))
    return requires


def _item(item_id, name, category, kind, cite, entry):
    item = {
        "id": item_id,
        "name": name,
        "category": category,
        "kind": kind,
        "cite": cite,
    }
    requires = _requires(entry)
    if requires:
        item["requires"] = requires
    if entry.get("aliases"):
        item["aliases"] = sorted(set(entry["aliases"]))
    if entry.get("provisional"):
        item["note"] = "provisional in the grammar"
    return item


def _instruction_category(instruction):
    name = instruction["opname"]
    if name in INSTRUCTION_CATEGORY:
        return INSTRUCTION_CATEGORY[name]
    if instruction["class"] in {"Reserved", "@exclude"}:
        for prefix, category in PREFIX_CATEGORY:
            if name.startswith(prefix):
                return category
    return CLASSES[instruction["class"]][2]


def _excluded_cite(entry, capability_extensions, environment, problems):
    """Extension document for an instruction the unified specification excludes, preferring Vulkan ones."""
    extensions = sorted(set(entry.get("extensions", [])))
    if not extensions:
        extensions = sorted(
            {
                ext
                for cap in entry.get("capabilities", [])
                for ext in capability_extensions.get(cap, ())
            }
        )
    if not extensions:
        problems.append(
            f"{entry['opname']}: excluded from the specification and no extension defines it"
        )
        return None
    usable = [name for name in extensions if name in environment.extensions]
    return f"spirv-registry {(usable or extensions)[0]}"


class Environment:
    """The SPIR-V capabilities and extensions a Vulkan implementation may accept, from vk.xml."""

    def __init__(self, vk_xml, core):
        root = ElementTree.fromstring(vk_xml)
        self.capabilities = {node.get("name") for node in root.iter("spirvcapability")}
        self.extensions = {node.get("name") for node in root.iter("spirvextension")}
        implied = {}
        aliases = {}
        for kind in core["operand_kinds"]:
            if kind["kind"] != "Capability":
                continue
            for enumerant in kind["enumerants"]:
                implied[enumerant["enumerant"]] = enumerant.get("capabilities", [])
                for alias in enumerant.get("aliases", []):
                    aliases[alias] = enumerant["enumerant"]
        listed = {aliases.get(name, name) for name in self.capabilities}
        available, pending = set(), list(listed)
        while (
            pending
        ):  # declaring a capability implicitly declares the ones it depends on
            name = pending.pop()
            if name not in available:
                available.add(name)
                pending.extend(implied.get(name, []))
        self.available = available
        self.aliases = aliases

    def capability(self, name):
        return self.aliases.get(name, name) in self.available

    def admits(self, entry):
        capabilities = entry.get("capabilities", [])
        if capabilities and not any(self.capability(name) for name in capabilities):
            return False
        extensions = entry.get("extensions", [])
        core_version = entry.get("version", "1.0") != "None"
        return (
            core_version
            or not extensions
            or any(name in self.extensions for name in extensions)
        )


# Operand kinds that carry an enumeration's value as a constant <id>.
ID_KINDS = {"IdMemorySemantics": "MemorySemantics", "IdScope": "Scope"}


class _Everything:
    @staticmethod
    def admits(entry):
        return True


def _used_kinds(core, environment):
    """Operand kinds that an admitted instruction, or a parameter of an admitted enumerant, can take."""
    by_name = {kind["kind"]: kind for kind in core["operand_kinds"]}
    pending = [
        ID_KINDS.get(operand["kind"], operand["kind"])
        for instruction in core["instructions"]
        if environment.admits(instruction)
        for operand in instruction.get("operands", [])
    ]
    pending.append(
        "Capability"
    )  # OpCapability's operand; listed explicitly for clarity
    used = set()
    while pending:
        name = pending.pop()
        if name in used:
            continue
        used.add(name)
        kind = by_name.get(name, {})
        for enumerant in kind.get("enumerants", []):
            if environment.admits(enumerant):
                pending.extend(
                    parameter["kind"] for parameter in enumerant.get("parameters", [])
                )
        pending.extend(kind.get("bases", []))
    return used


class _Derivation:
    """Accumulates items, cited sections, extensions and problems while walking the grammar."""

    def __init__(self, core, environment):
        self.core, self.environment = core, environment
        self.problems, self.sections, self.items = [], {}, []
        self.extensions, self.omitted = set(), 0
        self.capability_extensions = {
            enumerant["enumerant"]: enumerant.get("extensions", [])
            for kind in core["operand_kinds"]
            if kind["kind"] == "Capability"
            for enumerant in kind["enumerants"]
        }

    def instructions(self):
        for instruction in self.core["instructions"]:
            if instruction["class"] not in CLASSES:
                self.problems.append(
                    f"instruction class {instruction['class']!r} has no category"
                )
                continue
            if not self.environment.admits(instruction):
                self.omitted += 1
                continue
            section, title, _ = CLASSES[instruction["class"]]
            self.extensions.update(instruction.get("extensions", []))
            if section:
                self.sections[section] = title
                cite = f"spirv §{section}"
            else:
                cite = _excluded_cite(
                    instruction,
                    self.capability_extensions,
                    self.environment,
                    self.problems,
                )
            if cite:
                name = instruction["opname"]
                category = _instruction_category(instruction)
                self.items.append(
                    _item(
                        f"spirv.op.{name}",
                        name,
                        category,
                        "instruction",
                        cite,
                        instruction,
                    )
                )

    def enumerants(self):
        # An enumeration some instruction names directly is kept only if an admitted one does; enumerations only
        # ever passed through IdRef operands (ray flags, cooperative matrix use) are filtered per enumerant.
        used = _used_kinds(self.core, self.environment)
        referenced = _used_kinds(self.core, _Everything)
        for kind in self.core["operand_kinds"]:
            if kind["category"] not in ENUMERATIONS:
                continue
            if kind["kind"] in referenced and kind["kind"] not in used:
                self.omitted += len(kind["enumerants"])
                continue
            if kind["kind"] not in KINDS:
                self.problems.append(f"operand kind {kind['kind']!r} has no category")
                continue
            if KINDS[kind["kind"]] is not None:
                self.enumeration(kind, *KINDS[kind["kind"]])

    def enumeration(self, kind, section, title, category):
        for enumerant in kind["enumerants"]:
            if kind["category"] == "BitEnum" and enumerant["value"] in {0, "0x0000"}:
                continue  # the empty mask
            name = enumerant["enumerant"]
            if kind["kind"] == "Capability":
                admitted = self.environment.capability(name)
            else:
                admitted = self.environment.admits(enumerant)
            if not admitted:
                self.omitted += 1
                continue
            self.sections[section] = title
            self.extensions.update(enumerant.get("extensions", []))
            item_id = f"spirv.{kind['kind']}.{name}"
            self.items.append(
                _item(
                    item_id,
                    name,
                    category,
                    kind["kind"],
                    f"spirv §{section}",
                    enumerant,
                )
            )

    def extended(self, glsl450):
        for instruction in glsl450["instructions"]:
            if not self.environment.admits(instruction):
                self.omitted += 1
                continue
            name = instruction["opname"]
            item_id = f"spirv.glsl-std-450.{name}"
            self.items.append(
                _item(
                    item_id,
                    name,
                    "builtin-function",
                    "GLSL.std.450",
                    "glsl-std-450 §2",
                    instruction,
                )
            )

    def extension_items(self):
        self.omitted += len(self.extensions - self.environment.extensions)
        self.extensions &= self.environment.extensions
        self.extensions |= {
            item["cite"].split(" ", 1)[1]
            for item in self.items
            if item["cite"].startswith("spirv-registry ")
        }
        for extension in sorted(self.extensions):
            self.items.append(
                {
                    "id": f"spirv.extension.{extension}",
                    "name": extension,
                    "category": "extension",
                    "kind": "extension",
                    "cite": f"spirv-registry {extension}",
                }
            )


def derive(core, glsl450, environment, cells=None):
    """Build the inventory from parsed inputs, keeping the given cells; raises InventoryError on gaps."""
    walk = _Derivation(core, environment)
    walk.instructions()
    walk.enumerants()
    walk.extended(glsl450)
    walk.extension_items()
    items, problems, sections, extensions, omitted = (
        sorted(walk.items, key=lambda item: item["id"]),
        walk.problems,
        walk.sections,
        walk.extensions,
        walk.omitted,
    )
    seen = set()
    for item in items:
        if item["id"] in seen:
            problems.append(f"duplicate item {item['id']}")
        seen.add(item["id"])
    cells = dict(cells or {})
    orphaned = sorted(set(cells) - seen)
    if orphaned:
        problems.append(
            f"classified items vanished from the inputs: {', '.join(orphaned)}"
        )
    if problems:
        raise InventoryError("\n".join(problems))
    documents = {
        extension: (
            f"{REGISTRY_URL}extensions/{extension.split('_')[1]}/{extension}.html"
        )
        for extension in sorted(extensions)
    }
    version = f"{core['major_version']}.{core['minor_version']}"
    glsl450_version = f"{glsl450['version'] / 100:.2f}"
    return {
        "schemaVersion": 1,
        "language": "spirv",
        "title": "SPIR-V in the Vulkan environment",
        "sources": {
            "spirv": {
                "title": (
                    f"SPIR-V Specification (unified), Version {version}, Revision {core['revision']}"
                ),
                "url": SPEC_URL,
                "version": f"{version} revision {core['revision']}",
                "sections": dict(
                    sorted(
                        sections.items(),
                        key=lambda kv: [int(p) for p in kv[0].split(".")],
                    )
                ),
            },
            "glsl-std-450": {
                "title": (
                    f"GLSL.std.450 Extended Instruction Set, Version {glsl450_version}, "
                    f"Revision {glsl450['revision']}"
                ),
                "url": GLSL450_URL,
                "version": f"{glsl450_version} revision {glsl450['revision']}",
                "sections": {"2": "Binary Form"},
            },
            "spirv-registry": {
                "title": "SPIR-V Registry extension specifications",
                "url": REGISTRY_URL,
                "documents": documents,
            },
        },
        "generated": {
            "by": "tests/conformance/inventory/spirv.py",
            "inputs": {
                name: {
                    "repository": repo,
                    "commit": commit,
                    "path": path,
                    "sha256": digest,
                }
                for name, (repo, commit, path, digest) in sorted(INPUTS.items())
            },
            "omittedKinds": sorted(
                kind for kind, entry in KINDS.items() if entry is None
            ),
            "omittedOutsideVulkan": omitted,
        },
        "items": items,
        "cells": cells,
    }


def read_inputs(directory):
    """Read the pinned inputs from a directory, verifying their hashes."""
    loaded = {}
    for name, (_, _, _, expected) in INPUTS.items():
        file = Path(directory) / name
        data = file.read_bytes()
        digest = hashlib.sha256(data).hexdigest()
        if digest != expected:
            raise InventoryError(
                f"{file} has sha256 {digest}, not the pinned {expected}"
            )
        loaded[name] = data
    core = json.loads(loaded[CORE])
    return core, json.loads(loaded[GLSL450]), Environment(loaded[VKXML], core)


def fetch(directory):
    for name, (repository, commit, path, _) in INPUTS.items():
        url = f"https://raw.githubusercontent.com/{repository}/{commit}/{path}"
        (Path(directory) / name).write_bytes(download(url))


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    where = parser.add_mutually_exclusive_group(required=True)
    where.add_argument(
        "--inputs-dir", type=Path, help=f"Directory holding {', '.join(INPUTS)}"
    )
    where.add_argument(
        "--fetch", action="store_true", help="Download the pinned inputs"
    )
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--write", action="store_true")
    action.add_argument("--check", action="store_true")
    parser.add_argument("--output", type=Path, default=DATA / "spirv.json")
    args = parser.parse_args(argv)
    with tempfile.TemporaryDirectory() as scratch:
        directory = args.inputs_dir
        if args.fetch:
            directory = Path(scratch)
            fetch(directory)
        core, glsl450, environment = read_inputs(directory)
    existing = json.loads(args.output.read_text()) if args.output.is_file() else {}
    data = derive(core, glsl450, environment, existing.get("cells"))
    errors = validate(data, language="spirv")
    if errors:
        raise InventoryError("\n".join(errors))
    text = dumps(data)
    if args.write:
        args.output.write_text(text)
        print(f"Wrote {args.output}: {len(data['items'])} items")
        return 0
    if args.output.is_file() and args.output.read_text() == text:
        print(f"{args.output} matches the pinned inputs ({len(data['items'])} items)")
        return 0
    print(f"{args.output} is out of date; regenerate with --write", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
