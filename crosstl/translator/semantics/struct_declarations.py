"""Refuse two different structures or constants declared under one name, and drop repeated identical ones.

A structure name denotes one type in its scope: MSL follows C++17 section 6.2 [basic.def.odr], HLSL and Slang
follow C++, and GLSL 4.60 section 4.2.7 makes a second declaration of a name in the same scope a compile error.
The GLSL, HLSL and MSL generators already refused two different declarations of a stage's structures and wrote a
repeated identical one once. The SPIR-V, Slang, WGSL, CUDA and HIP generators wrote both, so a member access
resolved against whichever declaration the generator read last.

:func:`refuse_conflicting_structs` checks, for every target, the program's structures and each stage's structures
together with the program's, the scope each stage's output sees. It refuses different declarations of one name with
:class:`ConflictingStructError` and keeps the first of identical ones.

Program constants follow the same rule. Every generator wrote ``const float A = 1.0;`` and ``const int A = 2;``
both: MSL and GLSL then fail to compile, and SPIR-V gets two constants named ``A``, of which a use reads one.
:class:`ConflictingConstantError` refuses two constants of one name that differ in type, value or attributes.
"""

from __future__ import annotations

from ..codegen.stage_utils import declaration_signature
from ..diagnostics import CODE_PREFIX, TranslationDiagnostic
from ..stage_utils import normalize_stage_name

__all__ = [
    "ConflictingConstantError",
    "ConflictingStructError",
    "refuse_conflicting_structs",
]


class ConflictingStructError(TranslationDiagnostic):
    """Two structures with one name and different members in one scope."""

    def __init__(self, target, name, scope, *, source_location=None):
        self.source_location = source_location
        self.missing_capabilities = ()
        super().__init__(
            f"the structure '{name}' is declared twice with different members in {scope}; a name denotes one "
            "type in its scope (C++17 section 6.2 and GLSL 4.60 section 4.2.7)",
            target=target,
            code=f"{CODE_PREFIX}{target or 'crossgl'}-conflicting-struct",
        )


class ConflictingConstantError(TranslationDiagnostic):
    """Two program constants with one name and a different type, value or attributes."""

    def __init__(self, target, name, *, source_location=None):
        self.source_location = source_location
        self.missing_capabilities = ()
        super().__init__(
            f"the constant '{name}' is declared twice with a different type, value or attributes; a name "
            "denotes one constant in its scope (C++17 section 6.2 and GLSL 4.60 section 4.2.7)",
            target=target,
            code=f"{CODE_PREFIX}{target or 'crossgl'}-conflicting-constant",
        )


def _constant_signature(node):
    """What two declarations of one constant must agree on; a node without a structural repr never agrees."""
    value = getattr(node, "value", getattr(node, "initial_value", None))
    return (
        type(node).__name__,
        repr(node),
        repr(value),
        repr(getattr(node, "attributes", None)),
        repr(getattr(node, "qualifiers", None)),
    )


def _deduplicate_constants(constants, target):
    seen, kept = {}, []
    for constant in constants:
        name = _named(constant)
        if name is None:
            kept.append(constant)
            continue
        signature = _constant_signature(constant)
        if name not in seen:
            seen[name] = signature
            kept.append(constant)
        elif seen[name] != signature:
            raise ConflictingConstantError(
                target, name, source_location=getattr(constant, "source_location", None)
            )
    return kept


def _named(node):
    name = getattr(node, "name", None)
    return name if isinstance(name, str) and name else None


def _deduplicate(structs, seen, target, scope):
    """``structs`` without the repeats of declarations in ``seen``, which it extends; refuses a conflict."""
    kept = []
    for struct in structs:
        name = _named(struct)
        if name is None:
            kept.append(struct)
            continue
        signature = declaration_signature(struct)
        if name not in seen:
            seen[name] = signature
            kept.append(struct)
        elif seen[name] != signature:
            raise ConflictingStructError(
                target,
                name,
                scope,
                source_location=getattr(struct, "source_location", None),
            )
    return kept


def refuse_conflicting_structs(program, target) -> None:
    """Refuse different structures or constants of one name in a scope; keep one of identical repeats."""
    target = str(target or "")
    constants = list(getattr(program, "constants", None) or [])
    if constants:
        program.constants = _deduplicate_constants(constants, target)
    program_seen = {}
    structs = list(getattr(program, "structs", None) or [])
    if structs:
        program.structs = _deduplicate(structs, program_seen, target, "the program")
    stages = getattr(program, "stages", None) or {}
    for stage_type, stage in stages.items():
        local = list(getattr(stage, "local_structs", None) or [])
        if local:
            stage.local_structs = _deduplicate(
                local,
                dict(program_seen),
                target,
                f"the {normalize_stage_name(stage_type)} stage",
            )
