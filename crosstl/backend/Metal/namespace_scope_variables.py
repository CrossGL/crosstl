"""Namespace-scope variables of a Metal unit keep their namespace identity in flattened CrossGL.

CrossGL has a single global scope, so the importer emits every namespace member at program scope under its
unqualified name. References were not resolved against that flattening ([namespace.qual], [basic.lookup.unqual]):

- a qualified reference such as ``c10::metal::max_ndim`` was spelled as one sanitized identifier that nothing
  declares, so every target used an undeclared name;
- two variables of the same name in different namespaces (``a::N`` and ``b::N``, or ``a::N`` and a global ``N``)
  were both emitted as ``N``, which no target accepts.

:func:`resolve_namespace_scope_variables` runs on the parsed unit before code generation. It gives each variable
declared in more than one namespace a distinct program-scope name built from its namespace path (the one in the
global namespace keeps its name), and rewrites every reference to a namespace-scope variable to the name its
declaration is emitted under. A reference is resolved the way C++ looks it up: a qualified name relative to the
enclosing namespaces of its use (or from the global namespace for a leading ``::``), an unqualified name through
the enclosing block scopes, then the enclosing namespaces and the using-directives they contain. An unqualified
use of a variable name that several namespaces declare and that this lookup cannot resolve to one of them is
refused with :class:`MetalNamespaceLookupError`.
"""

from __future__ import annotations

from typing import Iterable, Sequence

__all__ = ["MetalNamespaceLookupError", "resolve_namespace_scope_variables"]

_SKIPPED_FIELDS = frozenset({"parent", "source_location", "annotations"})


class MetalNamespaceLookupError(ValueError):
    """An unqualified use of a namespace-scope variable that lookup cannot bind to one declaration."""

    project_diagnostic_code = "project.translate.metal-namespace-lookup"
    missing_capabilities = ("crossgl.namespaces",)

    def __init__(self, name: str, namespace: str, candidates: Sequence[str]):
        self.variable_name = name
        self.namespace = namespace
        self.candidates = tuple(candidates)
        where = f"namespace '{namespace}'" if namespace else "the global namespace"
        super().__init__(
            f"Metal name '{name}' used in {where} does not resolve to one of the namespace-scope variables "
            f"{', '.join(self.candidates)}; qualify the use so the importer can bind it"
        )


def _namespace_chain(namespace: str) -> list[str]:
    """``a::b`` -> ``['a::b', 'a', '']``: the namespaces unqualified lookup searches, innermost first."""
    components = [part for part in str(namespace or "").split("::") if part]
    return ["::".join(components[:count]) for count in range(len(components), -1, -1)]


def _join(namespace: str, name: str) -> str:
    return f"{namespace}::{name}" if namespace else name


def _normalized(name: str) -> str:
    return "::".join(part.strip() for part in str(name).split("::"))


def _declarations(ast) -> list[tuple[object, str, str]]:
    """``(VariableNode, namespace, name)`` of every namespace-scope variable the unit defines."""
    found = []
    seen: set[int] = set()
    for collection in ("global_variables", "global_vars"):
        for entry in getattr(ast, collection, None) or []:
            declaration = getattr(entry, "left", entry)
            if id(declaration) in seen or type(declaration).__name__ != "VariableNode":
                continue
            seen.add(id(declaration))
            name = getattr(declaration, "name", None)
            if not isinstance(name, str) or not name or "::" in name or "<" in name:
                continue
            found.append(
                (declaration, str(getattr(declaration, "namespace", "") or ""), name)
            )
    return found


def _identifier_names(roots: Iterable[object]) -> set[str]:
    names: set[str] = set()
    for node in _nodes(roots):
        name = getattr(node, "name", None)
        if isinstance(name, str):
            names.add(name)
    return names


def _nodes(roots: Iterable[object]):
    seen: set[int] = set()
    stack = list(roots)
    while stack:
        node = stack.pop()
        if id(node) in seen:
            continue
        seen.add(id(node))
        if isinstance(node, (list, tuple)):
            stack.extend(node)
            continue
        if isinstance(node, dict):
            stack.extend(node.values())
            continue
        if not hasattr(node, "__dict__") or isinstance(node, type):
            continue
        yield node
        for field, value in vars(node).items():
            if field not in _SKIPPED_FIELDS and (
                isinstance(value, (list, tuple, dict)) or hasattr(value, "__dict__")
            ):
                stack.append(value)


class _Resolver:
    """Rewrites references in one unit; ``emitted`` maps each qualified variable name to its CrossGL name."""

    def __init__(self, ast, emitted: dict[str, str], ambiguous: set[str]):
        self.emitted = emitted
        self.ambiguous = ambiguous
        self.aliases = {
            _normalized(alias): _normalized(target)
            for alias, target in (getattr(ast, "namespace_aliases", None) or {}).items()
        }
        self.directives = [
            (
                _normalized(directive.get("namespace") or ""),
                _normalized(directive.get("target") or ""),
                directive.get("declaration_context"),
            )
            for directive in getattr(ast, "using_namespace_directives", None) or []
            if isinstance(directive, dict) and directive.get("target")
        ]
        self.namespaces = {
            namespace
            for qualified in emitted
            for namespace in _namespace_chain(
                qualified.rsplit("::", 1)[0] if "::" in qualified else ""
            )
        }
        self.visited: set[int] = set()

    def expand_alias(self, name: str) -> str:
        head, _, rest = name.partition("::")
        target = self.aliases.get(head)
        return f"{target}::{rest}" if target and rest else name

    def qualified(self, reference: str, namespace: str) -> str | None:
        reference = self.expand_alias(_normalized(reference))
        if reference.startswith("::"):
            reference = reference[2:]
            return reference if reference in self.emitted else None
        for scope in _namespace_chain(namespace):
            candidate = _join(scope, reference)
            if candidate in self.emitted:
                return candidate
        return None

    def used_namespaces(self, namespace: str, context: str | None) -> list[str]:
        """Namespaces nominated by the using-directives of ``namespace`` (and of the function ``context``)."""
        nominated = []
        for directive_namespace, target, declaration_context in self.directives:
            if directive_namespace != namespace:
                continue
            if declaration_context and declaration_context != context:
                continue
            target = self.expand_alias(target)
            for scope in (
                [""] if target.startswith("::") else _namespace_chain(namespace)
            ):
                candidate = _join(scope, target.lstrip(":"))
                if candidate in self.namespaces:
                    nominated.append(candidate)
                    break
        return nominated

    def unqualified(self, name: str, namespace: str, context: str | None) -> str:
        for scope in _namespace_chain(namespace):
            candidate = _join(scope, name)
            if candidate in self.emitted:
                return candidate
            found = sorted(
                {
                    _join(used, name)
                    for used in self.used_namespaces(scope, context)
                    if _join(used, name) in self.emitted
                }
            )
            if len(found) == 1:
                return found[0]
            if found:
                raise MetalNamespaceLookupError(name, namespace, found)
        raise MetalNamespaceLookupError(
            name,
            namespace,
            sorted(q for q in self.emitted if q.rsplit("::", 1)[-1] == name),
        )

    def rewrite(
        self, node, namespace: str, context: str | None, scopes: list[set[str]]
    ) -> None:
        name = node.name
        if "::" in name:
            target = self.qualified(name, namespace)
        elif name in self.ambiguous and not any(name in scope for scope in scopes):
            target = self.unqualified(name, namespace, context)
        else:
            return
        if target is not None:
            node.name = self.emitted[target]

    def walk(
        self, node, namespace: str, context: str | None, scopes: list[set[str]]
    ) -> None:
        if isinstance(node, (list, tuple)):
            inner = scopes + [set()]
            for item in node:
                self.walk(item, namespace, context, inner)
            return
        if isinstance(node, dict):
            for value in node.values():
                self.walk(value, namespace, context, scopes)
            return
        if (
            not hasattr(node, "__dict__")
            or isinstance(node, type)
            or id(node) in self.visited
        ):
            return
        self.visited.add(id(node))
        kind = type(node).__name__
        if kind == "VariableNode":
            self.walk_variable(node, namespace, context, scopes)
            return
        if kind in {"FunctionNode", "LambdaNode"}:
            inner = scopes + [set()]
            if kind == "FunctionNode":
                namespace = str(getattr(node, "namespace", namespace) or "")
                context = f"function {node.name}"
            for param in getattr(node, "params", None) or []:
                self.walk(param, namespace, context, inner)
            self.walk(getattr(node, "body", None), namespace, context, inner)
            return
        if kind == "StructNode":
            namespace = str(getattr(node, "namespace", namespace) or "")
            members = {
                member.name
                for member in getattr(node, "members", None) or []
                if isinstance(getattr(member, "name", None), str)
            }
            inner = scopes + [members]
        elif kind in {"ForNode", "RangeForNode", "IfNode", "WhileNode", "SwitchNode"}:
            inner = scopes + [set()]
        else:
            inner = scopes
        if kind == "RangeForNode":
            self.walk(getattr(node, "iterable", None), namespace, context, inner)
            inner[-1].add(node.name)
        for field, value in vars(node).items():
            if field in _SKIPPED_FIELDS or (
                kind == "RangeForNode" and field == "iterable"
            ):
                continue
            self.walk(value, namespace, context, inner)

    def walk_variable(
        self, node, namespace: str, context: str | None, scopes: list[set[str]]
    ) -> None:
        name = getattr(node, "name", None)
        if not getattr(node, "vtype", None):
            if isinstance(name, str) and name:
                self.rewrite(node, namespace, context, scopes)
            return
        for field in ("array_sizes", "default_value"):
            self.walk(getattr(node, field, None), namespace, context, scopes)
        if scopes and isinstance(name, str):
            scopes[-1].add(name)
        self.walk(getattr(node, "value", None), namespace, context, scopes)


def _unique(base: str, taken: set[str]) -> str:
    candidate = base
    suffix = 2
    while candidate in taken:
        candidate = f"{base}_{suffix}"
        suffix += 1
    taken.add(candidate)
    return candidate


def resolve_namespace_scope_variables(ast) -> None:
    """Give same-named namespace-scope variables distinct names and bind every reference to its declaration."""
    declarations = _declarations(ast)
    if not declarations:
        return
    qualified_by_name: dict[str, set[str]] = {}
    for _node, namespace, name in declarations:
        qualified_by_name.setdefault(name, set()).add(_join(namespace, name))
    ambiguous = {
        name for name, qualified in qualified_by_name.items() if len(qualified) > 1
    }
    if not ambiguous and not any(namespace for _node, namespace, _name in declarations):
        return
    roots = [value for value in vars(ast).values() if isinstance(value, list)]
    taken = _identifier_names(roots) if ambiguous else set()
    emitted: dict[str, str] = {}
    for _node, namespace, name in sorted(
        declarations, key=lambda item: (item[2], item[1])
    ):
        qualified = _join(namespace, name)
        if qualified in emitted:
            continue
        emitted[qualified] = (
            _unique("_".join([*namespace.split("::"), name]), taken)
            if name in ambiguous and namespace
            else name
        )
    resolver = _Resolver(ast, emitted, ambiguous)
    for root in roots:
        for node in root:
            # A program-scope declaration is looked up from the namespace it is declared in.
            declared = (
                node if hasattr(node, "namespace") else getattr(node, "left", None)
            )
            resolver.walk(node, str(getattr(declared, "namespace", "") or ""), None, [])
    for node, namespace, name in declarations:
        node.name = emitted[_join(namespace, name)]
