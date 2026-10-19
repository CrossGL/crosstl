"""Derive the public Metal standard library from the Metal toolchain's own headers.

The Metal compiler parses ``#include <metal_stdlib>`` once per language version with modules disabled and dumps
its AST (``-Xclang -ast-dump``). Every public declaration of namespace ``metal`` and its public nested namespaces
(``metal::fast``, ``metal::raytracing``, ...) becomes an item: free functions, member functions of public classes,
class templates, structs, type aliases, enumerations and their enumerators, and library constants. Declarations
whose name or enclosing scope starts with an underscore are implementation details, and constructors,
destructors, conversions and operators are left to the curated language items. The first language version whose
headers declare a name is its minimum version (``requires.msl``).

The headers are Apple's and are not redistributed: only declaration names are recorded, and the header tree is
pinned by a digest of its file hashes, so a different toolchain is reported instead of silently changing the
inventory.
"""

from __future__ import annotations

import hashlib
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

from . import InventoryError

# Language versions in release order, as ``-std`` values of the macOS Metal compiler.
STANDARDS = (
    ("macos-metal1.0", "1.0"),
    ("macos-metal1.1", "1.1"),
    ("macos-metal1.2", "1.2"),
    ("macos-metal2.0", "2.0"),
    ("macos-metal2.1", "2.1"),
    ("macos-metal2.2", "2.2"),
    ("macos-metal2.3", "2.3"),
    ("macos-metal2.4", "2.4"),
    ("metal3.0", "3.0"),
    ("metal3.1", "3.1"),
    ("metal3.2", "3.2"),
    ("metal4.0", "4.0"),
    ("metal4.1", "4.1"),
)
NODE = re.compile(r"^(?P<indent>(?:[| ] )*)[|`]-(?P<kind>[A-Za-z]+)(?P<rest>.*)$")
FILE = re.compile(r"(/[^\s:<>,']+|<scratch space>|<built-in>|<command line>):\d+:\d+")
LOCATION = re.compile(
    r"^\s*(?:<invalid sloc>|\S+:\d+(?::\d+)?)(?:\s*<Spelling=[^>]*>)?"
)
FLAGS = {
    "implicit",
    "used",
    "referenced",
    "invalid",
    "constexpr",
    "consteval",
    "class",
    "struct",
    "union",
    "interface",
    "typename",
    "imported",
    "hidden",
    "inline",
    "static",
    "extern",
    "nested",
}
RECORDS = {
    "CXXRecordDecl",
    "ClassTemplateDecl",
    "ClassTemplatePartialSpecializationDecl",
    "ClassTemplateSpecializationDecl",
}
TYPES = RECORDS | {"TypedefDecl", "TypeAliasDecl", "TypeAliasTemplateDecl", "EnumDecl"}
FUNCTIONS = {"FunctionDecl", "FunctionTemplateDecl"}
METHODS = {"CXXMethodDecl", "FunctionTemplateDecl"}
CONSTANTS = {"VarDecl", "VarTemplateDecl"}
# Classes that are resources: textures, samplers, imageblocks, tensors, acceleration structures, function tables.
RESOURCE = re.compile(
    r"^(?:texture(?:1d|2d|3d|cube|_buffer)\w*|depth(?:2d|cube)\w*|sampler|imageblock|tensor"
    r"|\w*acceleration_structure\w*|\w*function_table)$"
)


@dataclass(frozen=True)
class Declaration:
    """One public library name: ``kind`` is function, member-function, class, alias, enum, enumerator, constant."""

    kind: str
    scope: tuple
    name: str
    header: str

    @property
    def qualified(self):
        return "::".join(("metal", *self.scope, self.name))


def _strip_range(rest):
    """Drop the node address, parent/prev links and the balanced source range that precede the location."""
    rest = re.sub(r"^\s*0x[0-9a-f]+", "", rest)
    rest = re.sub(r"^(?:\s+(?:parent|prev) 0x[0-9a-f]+)+", "", rest)
    rest = rest.lstrip()
    if rest.startswith("<"):
        depth = 0
        for index, character in enumerate(rest):
            depth += {"<": 1, ">": -1}.get(character, 0)
            if depth == 0:
                return rest[index + 1 :]
    return rest


def parse_node(rest):
    """``(implicit, name)`` of an AST node line; ``name`` is None for anonymous or unnamed nodes."""
    rest = LOCATION.sub("", _strip_range(rest), count=1)
    implicit = False
    for token in rest.split():
        if token in FLAGS:
            implicit = implicit or token == "implicit"
            continue
        if token.startswith(("'", '"', "<")) or not re.match(r"^[A-Za-z_]\w*$", token):
            return implicit, None
        return implicit, token
    return implicit, None


def header_of(path, root):
    """Header path relative to the include root, or None outside it."""
    try:
        return Path(path).resolve().relative_to(root).as_posix()
    except (ValueError, OSError):
        return None


@dataclass
class _Frame:
    kind: str
    name: str | None
    public: bool
    access: str = "public"


@dataclass
class _Record:
    """Methods and public bases of one class, public or internal, keyed by its scope."""

    header: str | None
    methods: set = field(default_factory=set)
    bases: list = field(default_factory=list)


def _base_name(rest):
    """Scope path of a base specifier such as ``public 'metal::_texture2d<T, a, c>'``."""
    match = re.search(r"'([^']+)'", rest)
    if match is None:
        return None
    text = re.sub(r"^(?:const |volatile )*", "", match.group(1))
    depth, name = 0, []
    for (
        character
    ) in text:  # drop template arguments, which may themselves contain ``::``
        depth += {"<": 1, ">": -1}.get(character, 0)
        if depth == 0 and character not in "<>":
            name.append(character)
    path = tuple(part for part in "".join(name).strip().split("::") if part)
    return path[1:] if path[:1] == ("metal",) else path


def _resolve(records, scope, path):
    """The record a base names, looked up from the derived class's scope outward, as C++ name lookup does."""
    for cut in range(len(scope), -1, -1):
        key = (*scope[:cut], *path)
        if key in records:
            return key
    return None


def _inherited(records, key, seen=()):
    record = records.get(key)
    if record is None or key in seen:
        return set()
    methods = set(record.methods)
    for base in record.bases:
        found = _resolve(records, key[:-1], base)
        if found is not None:
            methods |= _inherited(records, found, (*seen, key))
    return methods


def declarations(lines, root):
    """Every public declaration in one AST dump, as a set of :class:`Declaration`."""
    root = Path(root).resolve()
    found, records, classes = set(), {}, {}
    stack = []
    current = None
    resolved = {}
    for line in lines:
        for match in FILE.finditer(line):
            current = match.group(1)
        node = NODE.match(line)
        if node is None:
            continue
        depth = len(node["indent"]) // 2
        del stack[depth:]
        kind = node["kind"]
        parent = stack[-1] if stack else None
        if parent is not None and parent.kind in RECORDS:
            if kind == "AccessSpecDecl":
                parent.access = node["rest"].split()[-1]
                continue
            if kind in {"public", "protected", "private"}:
                base = _base_name(node["rest"])
                scope = _scope(stack)
                if kind == "public" and base and scope in records:
                    records[scope].bases.append(base)
                continue
        implicit, name = parse_node(node["rest"])
        if name == "operator" or (
            kind in METHODS
            and parent is not None
            and parent.kind in RECORDS
            and name == parent.name
        ):
            name = None  # conversion and constructor templates are not named members
        if kind == "NamespaceDecl" and name is None:
            name = node["rest"].split()[-1] if node["rest"].split() else None
        within = bool(stack) and (stack[0].kind, stack[0].name) == (
            "NamespaceDecl",
            "metal",
        )
        visible = (
            within
            and not implicit
            and name is not None
            and (parent.kind not in RECORDS or parent.access == "public")
        )
        public = (
            visible and not name.startswith("_") and all(f.public for f in stack)
        ) or (not stack and kind == "NamespaceDecl" and name == "metal")
        frame = _Frame(kind, name, public)
        if kind in RECORDS:
            frame.access = (
                "private" if re.search(r"\bclass\b", node["rest"]) else "public"
            )
        stack.append(frame)
        if current not in resolved:
            resolved[current] = header_of(current, root) if current else None
        header = resolved[current]
        if not within or header is None or kind == "NamespaceDecl":
            continue
        if kind in RECORDS and visible:
            records.setdefault(_scope(stack), _Record(header))
        if (
            parent.kind in RECORDS
            and kind in METHODS
            and visible
            and not name.startswith("_")
        ):
            records.setdefault(_scope(stack[:-1]), _Record(header)).methods.add(name)
        if not public:
            continue
        namespaces = tuple(f.name for f in stack[1:-1] if f.kind == "NamespaceDecl")
        if parent.kind == "NamespaceDecl":
            if kind in FUNCTIONS:
                found.add(Declaration("function", namespaces, name, header))
            elif kind in TYPES:
                kinds = (
                    "enum"
                    if kind == "EnumDecl"
                    else "class" if kind in RECORDS else "alias"
                )
                found.add(Declaration(kinds, namespaces, name, header))
                if kind in RECORDS:
                    classes.setdefault(_scope(stack), header)
            elif kind in CONSTANTS:
                found.add(Declaration("constant", namespaces, name, header))
        elif parent.kind == "EnumDecl" and kind == "EnumConstantDecl":
            found.add(Declaration("enumerator", _scope(stack[:-1]), name, header))
    # Member functions of every public class, including those it inherits from internal bases.
    for key, header in sorted(classes.items()):
        for method in _inherited(records, key):
            found.add(Declaration("member-function", key, method, header))
    return found


def _scope(stack):
    """Names of the namespaces and classes a frame stack denotes, below ``metal``; templates nest a class twice."""
    names = [
        f.name
        for f in stack[1:]
        if f.kind == "NamespaceDecl" or f.kind in RECORDS or f.kind == "EnumDecl"
    ]
    return _dedupe(names)


def _dedupe(names):
    """A class template and its pattern record nest the same name twice; keep one level per class."""
    result = []
    for name in names:
        if not result or result[-1] != name:
            result.append(name)
    return tuple(result)


def include_root(compiler):
    """The ``include/metal`` directory the compiler resolves ``<metal_stdlib>`` from."""
    result = subprocess.run(
        [*compiler, "-x", "metal", "-fno-modules", "-M", "-"],
        input="#include <metal_stdlib>\n",
        capture_output=True,
        text=True,
        check=False,
    )
    match = re.search(r"(\S+)/metal_stdlib\b", result.stdout.replace("\\\n", " "))
    if result.returncode != 0 or match is None:
        raise InventoryError(f"cannot locate <metal_stdlib>: {result.stderr[-500:]}")
    return Path(match.group(1)).resolve()


def header_digests(root):
    """sha256 of every header file under the include root, keyed by relative path."""
    root = Path(root)
    return {
        path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def tree_digest(digests):
    return hashlib.sha256(
        "".join(
            f"{name}\0{digest}\n" for name, digest in sorted(digests.items())
        ).encode()
    ).hexdigest()


def dump(compiler, standard):
    """The AST of ``#include <metal_stdlib>`` for one language version, streamed line by line."""
    command = [
        *compiler,
        "-x",
        "metal",
        f"-std={standard}",
        "-fno-modules",
        "-fno-color-diagnostics",
        "-fsyntax-only",
        "-Xclang",
        "-ast-dump",
        "-",
    ]
    process = subprocess.Popen(
        command,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    process.stdin.write("#include <metal_stdlib>\n")
    process.stdin.close()
    yield from process.stdout
    error = process.stderr.read()
    if process.wait() != 0:
        raise InventoryError(f"metal -std={standard} failed: {error[-500:]}")


def library(compiler, root):
    """``(kind, scope, name)`` of every public declaration -> (first language version, sorted headers)."""
    first, latest = {}, {}
    for standard, version in STANDARDS:
        latest = {}
        for item in declarations(dump(compiler, standard), root):
            latest.setdefault((item.kind, item.scope, item.name), set()).add(
                item.header
            )
        for key in latest:
            first.setdefault(key, version)
    missing = sorted(
        "::".join(("metal", *key[1], key[2])) for key in first if key not in latest
    )
    if missing:
        raise InventoryError(
            f"declarations removed by {STANDARDS[-1][0]}: {', '.join(missing[:20])}"
        )
    return {
        key: (version, tuple(sorted(latest[key]))) for key, version in first.items()
    }


def is_resource(kind, name):
    return kind in {"class", "alias"} and RESOURCE.match(name) is not None
