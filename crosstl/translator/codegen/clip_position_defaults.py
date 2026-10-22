"""CrossGL's default clip-space position for a vertex stage that returns a structure.

A vertex entry point whose output structure declares no position semantic still produces one clip-space position: the
first four-component float member named ``position``, ``clipPosition`` or ``glPosition`` (case and underscores are
ignored). The HLSL generator gives that member ``SV_Position`` and the Metal generator ``[[position]]``; the GLSL
generator writes it to ``gl_Position``. Without the rule a GLSL vertex shader never writes ``gl_Position`` and draws
nothing.
"""

DEFAULT_CLIP_POSITION_NAMES = frozenset({"position", "clipposition", "glposition"})


def is_default_clip_position_name(name):
    """Whether ``name`` is one of the member names that carry the default clip position."""
    return str(name or "").replace("_", "").lower() in DEFAULT_CLIP_POSITION_NAMES


def default_clip_position_member(
    members, *, has_semantic, declares_position, is_float4
):
    """The member of a vertex output structure that carries the clip position by default, or ``None``.

    ``has_semantic(member)`` tells whether a member declares any semantic, ``declares_position(member)`` whether
    that semantic is the clip position, and ``is_float4(member)`` whether the member is a four-component float
    vector. A structure in which a member declares the position keeps it; otherwise the first member without a
    semantic that has a default name and a float4 type carries it.
    """
    members = list(members or ())
    if any(declares_position(member) for member in members):
        return None
    for member in members:
        if has_semantic(member):
            continue
        if not is_default_clip_position_name(getattr(member, "name", None)):
            continue
        if is_float4(member):
            return member
    return None
