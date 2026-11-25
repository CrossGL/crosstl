"""Name-to-type views that resolve one name at a time.

The preprocessor types expressions at a source position through views such as ``{name: type}`` of every variable
declared before it (``_flatten_types_at``). Building such a view resolves every declared name, while typing one
expression looks up only the few names it reads. A translation unit with many ``auto`` locals and many declared
names therefore spent most of its preprocessing time building views it barely read.

A :class:`LazyTypeView` is a ``dict`` equal to the view built eagerly. ``view[key]``, ``view.get(key)`` and
``key in view`` resolve only ``key``. Every other operation (iteration, length, equality, copies, updates) first
resolves every name in declaration order, so it sees the same keys, values and key order as the eager view.
"""

from __future__ import annotations

from typing import Callable, Hashable, Iterable, Optional, Tuple

__all__ = ["LazyTypeView"]

_Entry = Optional[Tuple[Hashable, object]]


def _complete_first(method_name):
    method = getattr(dict, method_name)

    def wrapper(self, *args, **kwargs):
        self._complete()
        return method(self, *args, **kwargs)

    wrapper.__name__ = method_name
    wrapper.__doc__ = method.__doc__
    return wrapper


class LazyTypeView(dict):
    """The ``dict`` that resolving every name of ``names`` in order would build, resolved on demand.

    ``entry(name)`` returns the ``(key, value)`` pair the eager view records for ``name``, or ``None`` when it
    records nothing; it must not depend on when it is called. ``name_of(key)`` returns the name whose entry can
    have ``key``. ``names`` supports ``in`` and keeps its order and contents while the view is in use.
    """

    __slots__ = ("_names", "_entry", "_name_of", "_absent", "_resolved")

    def __init__(
        self,
        names: Iterable[Hashable],
        entry: Callable[[Hashable], _Entry],
        name_of: Callable[[Hashable], Hashable] = lambda key: key,
    ):
        super().__init__()
        self._names = names
        self._entry = entry
        self._name_of = name_of
        self._absent = set()
        self._resolved = False

    def _resolve(self, key) -> None:
        if self._resolved or dict.__contains__(self, key) or key in self._absent:
            return
        name = self._name_of(key)
        found = self._entry(name) if name in self._names else None
        if found is not None and found[0] == key:
            dict.__setitem__(self, key, found[1])
        else:
            self._absent.add(key)

    def _complete(self) -> None:
        if self._resolved:
            return
        entries = {}
        for name in self._names:
            found = self._entry(name)
            if found is not None:
                entries[found[0]] = found[1]
        dict.clear(self)
        dict.update(self, entries)
        self._resolved = True
        self._absent = set()

    def __getitem__(self, key):
        self._resolve(key)
        return dict.__getitem__(self, key)

    def get(self, key, default=None):
        self._resolve(key)
        return dict.get(self, key, default)

    def __contains__(self, key) -> bool:
        self._resolve(key)
        return dict.__contains__(self, key)

    def copy(self) -> dict:
        self._complete()
        return dict(dict.items(self))

    def __reduce_ex__(self, protocol):
        return (dict, (self.copy(),))

    def __eq__(self, other):
        self._complete()
        if isinstance(other, LazyTypeView):
            other._complete()
        return dict.__eq__(self, other)

    def __ne__(self, other):
        result = self.__eq__(other)
        return result if result is NotImplemented else not result

    __iter__ = _complete_first("__iter__")
    __len__ = _complete_first("__len__")
    __repr__ = _complete_first("__repr__")
    __reversed__ = _complete_first("__reversed__")
    __or__ = _complete_first("__or__")
    __ror__ = _complete_first("__ror__")
    __ior__ = _complete_first("__ior__")
    __setitem__ = _complete_first("__setitem__")
    __delitem__ = _complete_first("__delitem__")
    keys = _complete_first("keys")
    values = _complete_first("values")
    items = _complete_first("items")
    pop = _complete_first("pop")
    popitem = _complete_first("popitem")
    setdefault = _complete_first("setdefault")
    update = _complete_first("update")
    clear = _complete_first("clear")
    __hash__ = None
