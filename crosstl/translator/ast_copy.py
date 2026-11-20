"""Deep copies of CrossGL AST nodes that give the same result as :func:`copy.deepcopy`'s default protocol.

The generators copy whole programs before passes that rewrite them in place (retry loops, per-variant output), and
on large programs the default protocol (``__reduce_ex__``, ``_reconstruct``, a deep copy of each ``__dict__``) is a
large share of translation time. :func:`deepcopy_node` builds the same object graph directly:

- a new instance of the node's class, created without ``__init__``, whose ``__dict__`` holds a deep copy of every
  field, in the original order;
- the memo is shared with :func:`copy.deepcopy`, so an object reached twice (an alias field such as
  ``AssignmentNode.left`` and ``target``, or a ``parent`` back link) is copied once and shared in the copy as it is
  in the original, and a caller that passes its own memo finds every copied node, list and dictionary in it;
- every copied original is kept alive in the memo, as :func:`copy.deepcopy` keeps it, so its id is never reused
  while the memo exists;
- immutable scalars are shared, as :func:`copy.deepcopy` shares them; plain lists and dictionaries are copied
  inline; every other value goes to :func:`copy.deepcopy` with the same memo, so classes with their own
  ``__deepcopy__`` (``StageMap``, or a node subclass that defines one) or reduction keep it.
"""

from __future__ import annotations

import copy

# Exactly these types are shared rather than copied, as in ``copy._deepcopy_atomic``; subclasses are not.
_SHARED_TYPES = frozenset({str, int, float, bool, complex, bytes, type(None)})
_MISSING = object()

# Whether a type's instances are copied by :func:`deepcopy_node`, decided once per concrete type.
_DIRECT_TYPES = {}


def _is_direct(value_type):
    direct = _DIRECT_TYPES.get(value_type)
    if direct is None:
        direct = getattr(value_type, "__deepcopy__", None) is node_deepcopy
        _DIRECT_TYPES[value_type] = direct
    return direct


def _keep_list(memo):
    """The list that keeps copied originals alive, the one :func:`copy.deepcopy` keeps under ``id(memo)``."""
    keep = memo.get(id(memo))
    if keep is None:
        keep = memo[id(memo)] = []
    return keep


def _copy_value(value, memo, keep):
    """A deep copy of a field value that is not one of :data:`_SHARED_TYPES`."""
    found = memo.get(id(value), _MISSING)
    if found is not _MISSING:
        return found
    value_type = type(value)
    shared = _SHARED_TYPES
    if value_type is list:
        result = []
        memo[id(value)] = result
        keep.append(value)
        append = result.append
        for item in value:
            append(item if type(item) in shared else _copy_value(item, memo, keep))
        return result
    if value_type is dict:
        result = {}
        memo[id(value)] = result
        keep.append(value)
        for key, item in value.items():
            if type(key) not in shared:
                key = _copy_value(key, memo, keep)
            result[key] = (
                item if type(item) in shared else _copy_value(item, memo, keep)
            )
        return result
    if _is_direct(value_type):
        return _copy_node(value, memo, keep)
    return copy.deepcopy(value, memo)


def _copy_node(node, memo, keep):
    node_type = type(node)
    duplicate = node_type.__new__(node_type)
    memo[id(node)] = duplicate
    keep.append(node)
    shared = _SHARED_TYPES
    state = {}
    for name, value in node.__dict__.items():
        state[name] = value if type(value) in shared else _copy_value(value, memo, keep)
    duplicate.__dict__.update(state)
    return duplicate


def deepcopy_node(node, memo):
    """A deep copy of the AST node ``node``, sharing ``memo`` with :func:`copy.deepcopy`."""
    return _copy_node(node, memo, _keep_list(memo))


def node_deepcopy(self, memo):
    """``ASTNode.__deepcopy__``. A subclass that defines its own is copied through :func:`copy.deepcopy`."""
    return deepcopy_node(self, memo)
