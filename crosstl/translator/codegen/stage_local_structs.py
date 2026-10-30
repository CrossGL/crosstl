"""Stage-local structs at module scope, for the emitters whose output is one C++ translation unit.

CrossGL lets a stage declare its own structs, among them the block type of a GLSL interface block such as
``buffer Data { vec4 values[]; } data;`` inside ``compute { ... }``. The CUDA and HIP emitters wrote only the shader's
structs, so the declaration naming a stage-local struct referred to an undeclared type. A stage-local struct is now
hoisted to module scope after the shader's own structs, in source order.
"""

from __future__ import annotations


def _struct_signature(struct):
    return tuple(
        (
            getattr(member, "name", None),
            str(getattr(member, "member_type", getattr(member, "vtype", None))),
        )
        for member in getattr(struct, "members", None) or []
    )


def module_structs(program, backend):
    """The shader's structs followed by each stage-local struct not declared already, in source order.

    Two stages that declare different structs of one name cannot share a translation unit, so that is refused.
    """
    structs = list(getattr(program, "structs", None) or [])
    seen = {
        getattr(struct, "name", None): struct
        for struct in structs
        if getattr(struct, "name", None)
    }
    for stage in (getattr(program, "stages", None) or {}).values():
        for struct in getattr(stage, "local_structs", None) or []:
            name = getattr(struct, "name", None)
            previous = seen.get(name) if name else None
            if previous is None:
                structs.append(struct)
                if name:
                    seen[name] = struct
                continue
            if previous is not struct and _struct_signature(
                previous
            ) != _struct_signature(struct):
                raise ValueError(
                    f"Conflicting {backend} stage-local struct '{name}': "
                    "two stages declare different members"
                )
    return structs
