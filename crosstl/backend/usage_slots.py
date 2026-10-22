"""Resource slots whose type an importer decides from how they are used.

Some source languages leave a resource property to the uses: Vulkan GLSL makes a ``texture2D`` a depth texture only
where it is combined into a ``sampler2DShadow``, and MSL makes a ``sampler`` a comparison sampler only through the
``sample_compare`` calls that take it. CrossGL states both in the type. An importer records the uses of each *slot*
(a global, a function parameter) and joins the slots one call binds (an argument and its parameter), since they must
have one type; :meth:`UsageSlots.only` then tells whether every use in a slot's class is of one kind.
"""

from __future__ import annotations

__all__ = ["UsageSlots"]


class UsageSlots:
    """Union-find over hashable slot keys, with the set of use kinds recorded per joined class."""

    def __init__(self):
        self._parent = {}
        self._uses = {}

    def _find(self, slot):
        self._parent.setdefault(slot, slot)
        root = slot
        while self._parent[root] != root:
            root = self._parent[root]
        while self._parent[slot] != root:
            self._parent[slot], slot = root, self._parent[slot]
        return root

    def join(self, left, right):
        """Give ``left`` and ``right`` one type: their uses are pooled."""

        left_root, right_root = self._find(left), self._find(right)
        if left_root == right_root:
            return
        self._parent[right_root] = left_root
        self._uses.setdefault(left_root, set()).update(
            self._uses.pop(right_root, set())
        )

    def use(self, slot, kind):
        """Record one use of ``slot`` of the given kind."""

        self._uses.setdefault(self._find(slot), set()).add(kind)

    def only(self, slot, kind):
        """True when ``slot``'s class is used, and every use is of ``kind``."""

        return self._uses.get(self._find(slot), set()) == {kind}
