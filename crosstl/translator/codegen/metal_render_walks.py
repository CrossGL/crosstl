"""Replayed AST walks for one Metal render.

The Metal generator learns most facts about a function by walking its body: the locals it declares, the user
functions it calls, and the resources, constant buffers and stage parameters it reads. Each fact is a separate walk,
and the dependency fixpoints and every ``generate_function`` ask for the same facts again, so one body is walked tens
of times per program. For MLX's ``scan.metal`` about half of the generation time went into stepping these walks
(#4479).

``render_program`` finishes rewriting the program before it collects facts. From then on generation only reads the
AST, and every lowering it applies works on a copy, so each walk of one value inside a render produces the same nodes
in the same order. :class:`MetalRenderWalks` records the nodes of the first walk of each value and replays them for
the later walks of that value. It also keeps, per value, the walked nodes a collector selects by type, so a collector
that needs only calls or only declarations visits only those.

Replay is guarded:

* It happens only inside :meth:`MetalRenderWalks.replaying`, which ``render_program`` enters after its rewrites.
  Outside it, and inside :meth:`MetalRenderWalks.suspended`, every walk is a fresh :func:`iter_ast_values` walk.
  Each window has its own records, so a render nested in another never replays the outer render's walks.
* A record keeps the walked value alive, so no other value can take over the identity it is keyed on.
* A walked list or tuple is compared item by item with the items it had when it was recorded, and is walked again
  when they differ, because callers build such roots for a single question and may reuse the object.
* A walk that fails, such as one that reaches a value inside itself, records nothing and fails again when it is
  repeated.

Replayed walks are complete: a consumer that stops early sees the same prefix as before, and the nodes are those the
first walk produced, so a consumer must not rely on seeing its own edits of the walked nodes. Code that edits the AST
runs before the replay window opens.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Callable, Iterator

from ..ast import FunctionCallNode, RayTracingOpNode
from .ast_value_walk import iter_ast_values

__all__ = [
    "MetalRenderWalks",
    "names_identifier_class",
    "names_resource_reference_class",
]

_SEQUENCE_TYPES = (list, tuple)


def names_identifier_class(node_class: type) -> bool:
    """Whether the generator treats instances of ``node_class`` as identifier references.

    The generator has always tested ``"Identifier" in str(node.__class__)``, which also accepts
    ``IdentifierPatternNode`` and any other class whose qualified name contains the word.
    """
    return "Identifier" in str(node_class)


def names_resource_reference_class(node_class: type) -> bool:
    """Whether a node can name a global resource: identifiers, calls and ray-tracing operations."""
    return names_identifier_class(node_class) or issubclass(
        node_class, (FunctionCallNode, RayTracingOpNode)
    )


class _Window:
    """The walk records of one replay window."""

    __slots__ = ("nodes", "selected")

    def __init__(self) -> None:
        # id(root) -> (root, the items of a list or tuple root when recorded, the walked nodes)
        self.nodes: dict[int, tuple[Any, tuple | None, tuple]] = {}
        # (id(root), selector) -> the walked nodes the selector accepts, in walk order
        self.selected: dict[tuple[int, Any], tuple] = {}


class MetalRenderWalks:
    """The walks of the values visited during one Metal render, recorded once and replayed."""

    __slots__ = ("_windows",)

    def __init__(self) -> None:
        # The innermost window is last. ``None`` marks a suspended stretch in which nothing is replayed.
        self._windows: list[_Window | None] = []

    @contextmanager
    def replaying(self) -> Iterator[None]:
        """Record and replay walks until the block exits."""
        self._windows.append(_Window())
        try:
            yield
        finally:
            self._windows.pop()

    @contextmanager
    def suspended(self) -> Iterator[None]:
        """Walk afresh until the block exits, as a render's rewrites need."""
        self._windows.append(None)
        try:
            yield
        finally:
            self._windows.pop()

    @property
    def active(self) -> bool:
        return bool(self._windows) and self._windows[-1] is not None

    def walk(self, root: Any) -> Iterator[Any]:
        """The values :func:`iter_ast_values` produces for ``root``, replayed while a window is open."""
        window = self._windows[-1] if self._windows else None
        if window is None:
            return iter_ast_values(root)
        return iter(self._recorded(window, root))

    def nodes_of_type(self, root: Any, types: type | tuple[type, ...]) -> tuple:
        """The walked values of ``root`` that are instances of ``types``, in walk order."""
        return self._select(
            root, types, lambda node_class: issubclass(node_class, types)
        )

    def nodes_with_class(self, root: Any, class_test: Callable[[type], bool]) -> tuple:
        """The walked values of ``root`` whose class passes ``class_test``, in walk order.

        ``class_test`` must depend only on the class it is given; its answer is reused for every value of that class.
        """
        return self._select(root, class_test, class_test)

    def body_nodes_of_type(
        self, function: Any, types: type | tuple[type, ...]
    ) -> tuple:
        """:meth:`nodes_of_type` for the body of ``function``; a function without a body has no nodes."""
        return self.nodes_of_type(getattr(function, "body", None), types)

    def body_nodes_with_class(
        self, function: Any, class_test: Callable[[type], bool]
    ) -> tuple:
        """:meth:`nodes_with_class` for the body of ``function``; a function without a body has no nodes."""
        return self.nodes_with_class(getattr(function, "body", None), class_test)

    def _select(
        self, root: Any, selector: Any, accepts: Callable[[type], bool]
    ) -> tuple:
        window = self._windows[-1] if self._windows else None
        if window is None:
            return _filtered(iter_ast_values(root), accepts)
        nodes = self._recorded(window, root)
        key = (id(root), selector)
        selected = window.selected.get(key)
        if selected is None or selected[0] is not nodes:
            selected = (nodes, _filtered(nodes, accepts))
            window.selected[key] = selected
        return selected[1]

    @staticmethod
    def _recorded(window: _Window, root: Any) -> tuple:
        record = window.nodes.get(id(root))
        items = tuple(root) if isinstance(root, _SEQUENCE_TYPES) else None
        if record is not None and record[0] is root and _same_items(record[1], items):
            return record[2]
        nodes = tuple(iter_ast_values(root))
        window.nodes[id(root)] = (root, items, nodes)
        return nodes


def _same_items(recorded: tuple | None, current: tuple | None) -> bool:
    if recorded is None or current is None:
        return recorded is current
    return len(recorded) == len(current) and all(
        left is right for left, right in zip(recorded, current)
    )


def _filtered(nodes: Any, accepts: Callable[[type], bool]) -> tuple:
    decisions: dict[type, bool] = {}
    selected = []
    for node in nodes:
        node_class = node.__class__
        accepted = decisions.get(node_class)
        if accepted is None:
            accepted = decisions[node_class] = bool(accepts(node_class))
        if accepted:
            selected.append(node)
    return tuple(selected)
