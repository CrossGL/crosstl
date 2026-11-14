"""Resolve substituted dependent member types of materialized structs in concrete code.

A function template argument that names a dependent member type, for example
``twiddle<scalar_T>`` with a local ``using scalar_T = typename IOTraits<in_T,
out_T>::scalar_T;``, is substituted before ``IOTraits<float2, float2>`` is
materialized.  The materialized function body then still spells
``typename IOTraits_float2_float2::scalar_T`` although the struct is concrete
and its alias names a concrete type.  The Metal parser accepts that spelling
in a declaration but not as a functional cast such as
``typename IOTraits_float2_float2::scalar_T(0)``, and no later stage resolves
it.  The owner may also be a local alias of the concrete struct, as in
``using kernel_t = Kern_half; threadgroup typename kernel_t::acc_type buf[4];``,
whose member type the importer otherwise reports as undeclared.

:func:`resolve_concrete_qualified_type_aliases` rewrites such a name outside
template declarations to the alias target: once the owner is a concrete
specialization the member is no longer dependent ([temp.res]).  Only names
that keep the ``typename`` disambiguator are rewritten, since that keyword
marks a substituted dependent spelling; a plain ``Owner::alias`` is valid
concrete code and keeps its spelling.  A name is rewritten only when its owner,
directly or through the type aliases visible at the name, is a concrete struct
and the member alias resolves in that struct's scope; anything else is left for
the existing diagnostics.
"""

from __future__ import annotations

import re
from typing import Any

_TYPENAME_QUALIFIED_RE = re.compile(
    r"(?<![A-Za-z0-9_:])typename\s+(?P<owner>(?:::)?[A-Za-z_]\w*(?:\s*::\s*[A-Za-z_]\w*)*)"
    r"\s*::\s*(?P<member>[A-Za-z_]\w*)(?![\w:<])"
)


def resolve_concrete_qualified_type_aliases(preprocessor: Any, code: str) -> str:
    """Return ``code`` with ``typename Owner::alias`` resolved for concrete owners."""
    if "typename" not in code:
        return code
    matches = list(_TYPENAME_QUALIFIED_RE.finditer(code))
    if not matches:
        return code
    structs = [
        struct
        for struct in preprocessor._find_concrete_struct_definitions(code)
        if any(
            alias not in struct.type_alias_templates for alias in struct.type_aliases
        )
    ]
    if not structs:
        return code
    owner_names = {struct.name for struct in structs} | {
        struct.qualified_name for struct in structs if struct.qualified_name
    }
    template_spans = preprocessor._find_template_declaration_spans(code)
    ignored = sorted(
        [*template_spans, *preprocessor._find_comment_and_literal_spans(code)]
    )
    local_aliases: dict | None = None
    replacements: list[tuple[int, int, str]] = []
    for match in matches:
        if preprocessor._containing_span(match.start(), ignored) is not None:
            continue
        owner = re.sub(r"\s+", "", match.group("owner")).lstrip(":")
        if owner not in owner_names:
            if local_aliases is None:
                local_aliases = preprocessor._collect_local_type_alias_bindings(
                    code, [(0, len(code))], skip_spans=template_spans
                )
            owner = re.sub(
                r"^typename\s+",
                "",
                preprocessor._resolve_type_aliases_at(
                    owner, local_aliases, match.start()
                ),
            ).lstrip(":")
            if owner not in owner_names:
                continue
        qualified = f"typename {owner}::{match.group('member')}"
        resolved = preprocessor._canonicalize_qualified_struct_type_aliases(
            qualified, structs
        )
        resolved = preprocessor._normalize_template_argument_text(resolved)
        if resolved != qualified and not resolved.startswith("typename"):
            replacements.append((match.start(), match.end(), resolved))
    if not replacements:
        return code
    return preprocessor._apply_text_replacements(code, replacements)
