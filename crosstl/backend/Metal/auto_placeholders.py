"""Declared types of Metal locals that use the ``auto`` placeholder.

A placeholder takes its type from the initializer ([dcl.type.auto.deduct]):

* ``auto x = e`` has the type of ``e``;
* ``auto* p = e`` requires ``e`` to be a pointer and has its type;
* ``const auto* p = e`` has the pointer type of ``e`` with a const pointee;
* ``const auto x = e`` and ``constexpr auto x = e`` have the type of ``e``,
  const-qualified (a pointer or reference initializer is left un-typed).

Any other placeholder spelling (references, address-space qualified
placeholders) or an initializer whose type is unknown gives the bare
placeholder ``auto``. That records the local as declared but un-typed, so it
still shadows outer bindings while template argument deduction from it fails
closed instead of binding a parameter to ``auto``.
"""

import re

_POINTER_PLACEHOLDER_RE = re.compile(r"(?P<const>const\s+)?auto\s*\*")


def is_auto_placeholder(type_text):
    """Whether a declared type spelling uses the ``auto`` placeholder."""
    return re.search(r"(?<![A-Za-z0-9_])auto(?![A-Za-z0-9_])", type_text) is not None


def deduced_auto_type(type_text, initializer_type):
    """Type of a local declared ``type_text`` with an initializer typed as given."""
    spelling = " ".join(type_text.split())
    # `static` is a storage class, and `constexpr` makes the object const without changing the deduced type.
    const_object = re.search(r"\bconstexpr\b", spelling) is not None
    spelling = " ".join(re.sub(r"\b(?:constexpr|static)\b", " ", spelling).split())
    if spelling in {"const auto", "auto const"}:
        const_object = True
        spelling = "auto"
    if spelling == "auto":
        if not initializer_type:
            return "auto"
        if const_object:
            value_type = re.sub(r"^\s*const\s+", "", " ".join(initializer_type.split()))
            if re.search(r"[*&]", value_type):
                return "auto"
            return f"const {value_type}"
        return initializer_type
    pointer = _POINTER_PLACEHOLDER_RE.fullmatch(spelling)
    if pointer is None or not initializer_type:
        return "auto"
    pointer_type = " ".join(initializer_type.split())
    if not pointer_type.endswith("*") or pointer_type.count("*") != 1:
        return "auto"
    if pointer.group("const") and not re.search(r"\bconst\b", pointer_type):
        pointer_type = f"const {pointer_type}"
    return pointer_type
