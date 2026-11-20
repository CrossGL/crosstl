"""Running state of the per-item emission loops in ``GLSLCodeGen._generate_program_attempt``."""

from dataclasses import dataclass


@dataclass
class OpenGLStructEmission:
    """The text written so far while the program's structs are declared, and the stage facts each struct reads.

    ``code`` receives interface-block declarations and ``stage_io_code`` the stage input and output declarations.
    ``data_struct_nodes`` collects the structs that are declared later in dependency order. The ``emit_*_io``
    flags and ``legacy_*_names`` say which stage interfaces this program emits and whether they keep the
    unprefixed names.
    """

    code: str
    stage_io_code: str
    ast: object
    data_struct_nodes: list
    emit_vertex_io: bool
    emit_fragment_io: bool
    emit_graphics_io: bool
    legacy_vertex_names: bool
    legacy_fragment_names: bool


@dataclass
class OpenGLResourceEmission:
    """The text written so far while the global resources are declared, and the binding maps they share.

    ``resource_binding_cursors`` and ``used_resource_bindings`` are the maps that the binding allocation reads
    and updates. ``empty_struct_global_ids`` holds the ids of globals whose struct type has no members.
    """

    code: str
    empty_struct_global_ids: set
    resource_binding_cursors: dict
    used_resource_bindings: dict
