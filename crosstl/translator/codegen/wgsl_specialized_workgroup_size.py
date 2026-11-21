"""WGSL ``override`` constants for the workgroup dimensions a compute stage specializes.

A CrossGL ``layout(local_size_x_id = N) in;`` lets the pipeline set the x dimension through the specialization
constant ``N``. WGSL expresses that with a pipeline-overridable constant, ``@id(N) override name: u32 = default;``,
named in ``@workgroup_size`` (WGSL section 9.3.6, ``workgroup_size``; section 7.2.3, override declarations), which the
WebGPU pipeline sets from ``GPUProgrammableStage.constants``. ``gl_WorkGroupSize`` reads the same constants.

A Metal kernel whose size the source's host chooses at dispatch time (:mod:`..host_workgroup_size`) is specialized
through the ids of :mod:`.host_workgroup_size_ids`, the ones the SPIR-V and GLSL generators give the host, by
overrides without an initializer: WebGPU then refuses to create a pipeline whose ``constants`` leave one unset
(WebGPU section 10.1.2, ``GPUProgrammableStage``), instead of running groups of a size nobody chose.

The overrides are module-scope declarations, so the entries of one module share them: two entries that specialize a
dimension through different ids, or one id with different defaults, are refused, as are two dimensions of one id.
WGSL pipeline constant ids are below 65536 (WGSL section 12.3.1.3, ``id``), so a larger id is refused.

A program may also declare the specialization constant of that id itself, ``const uint size @constant_id(0) = 64u;``
beside ``local_size_x_id = 0``, as GLSL does with ``layout(constant_id = 0)`` and as the WGSL front end writes an
``override`` named in ``@workgroup_size``. That constant is already the module's ``@id(0) override``, and WGSL
pipeline constant ids are unique within a module, so ``@workgroup_size`` names it instead of declaring a second
override of the same id; its default is the constant's, as SPIR-V output's ``WorkgroupSize`` composite reads the
constant. A ``@workgroup_size`` argument is an ``i32`` or ``u32`` and all of them have one type (WGSL section 9.3.6),
so the overrides the module declares take that constant's type, and a constant of another type, or two of
different types, are refused.
"""

from __future__ import annotations

import re

from .host_workgroup_size_ids import (
    host_sized_workgroup_config,
    host_workgroup_size_ids,
    is_host_sized_workgroup,
)
from .specialized_workgroup_size import (
    SpecializedWorkgroupSizeError,
    workgroup_size_specialization_ids,
)

__all__ = [
    "WorkgroupSizeOverrides",
    "specialized_workgroup_size",
    "workgroup_size_u32_values",
]

_MAX_WGSL_ID = 65535
_ARGUMENT_TYPES = frozenset({"i32", "u32"})


def _override_name(axis: str) -> str:
    return f"crossgl_workgroup_size_{axis}"


def _default(value: str, type_name: str) -> str:
    text = str(value).strip()
    if re.fullmatch(r"[0-9]+", text):
        return f"{text}u" if type_name == "u32" else text
    return f"{type_name}({text})"


def _declaration(name: str, spec_id: int, value, type_name: str = "u32") -> str:
    if value is None:
        return f"@id({spec_id})\noverride {name}: {type_name};"
    return (
        f"@id({spec_id})\noverride {name}: {type_name} = {_default(value, type_name)};"
    )


def _argument_type(ids, constants) -> str:
    """The type of the ``@workgroup_size`` overrides: that of the program constants it names, else ``u32``."""
    types = {}
    for axis, spec_id in zip("xyz", ids):
        if spec_id in constants:
            name, type_name = constants[spec_id]
            if type_name not in _ARGUMENT_TYPES:
                raise SpecializedWorkgroupSizeError(
                    f"local_size_{axis}_id = {spec_id} names specialization constant '{name}' of type "
                    f"{type_name or 'unknown'}; a WGSL @workgroup_size argument is an i32 or u32",
                    target="WGSL",
                    specialization_ids=(spec_id,),
                    reason="workgroup-size-constant-type",
                )
            types[type_name] = spec_id
    if len(types) > 1:
        raise SpecializedWorkgroupSizeError(
            "the specialization constants that size the workgroup have types "
            f"{' and '.join(sorted(types))}; WGSL @workgroup_size arguments all have one type",
            target="WGSL",
            specialization_ids=tuple(sorted(types.values())),
            reason="workgroup-size-constant-type",
        )
    return next(iter(types), "u32")


def _specialized_axes(config, values, constants=None):
    """``(argument, spec id, declaration or None)`` of each dimension: ``None`` ids for fixed dimensions."""
    constants = constants or {}
    ids = workgroup_size_specialization_ids(config, "WGSL")
    type_name = _argument_type(ids, constants)
    axes = []
    for axis, value, spec_id in zip("xyz", values, ids):
        if spec_id is None:
            axes.append((value, None, None))
            continue
        if spec_id > _MAX_WGSL_ID:
            raise SpecializedWorkgroupSizeError(
                f"WGSL pipeline constant ids are below 65536; local_size_{axis}_id = {spec_id} is not",
                target="WGSL",
                specialization_ids=(spec_id,),
                reason="specialization-id-out-of-range",
            )
        if spec_id in constants:
            # The program's own override of this id is the dimension.
            axes.append((constants[spec_id][0], spec_id, None))
            continue
        name = _override_name(axis)
        axes.append((name, spec_id, _declaration(name, spec_id, value, type_name)))
    return axes


def specialized_workgroup_size(config, values, constants=None):
    """The ``@workgroup_size`` arguments and the override declarations that specialize them.

    ``values`` are the three default dimensions; a dimension the stage names an id for becomes its override, and a
    default of ``None`` leaves the override without an initializer. ``constants`` maps the id of each override the
    program declares itself to its WGSL name and type; a dimension of such an id names it and declares nothing.
    """
    axes = _specialized_axes(config, values, constants)
    arguments = tuple(argument for argument, _, _ in axes)
    declarations = [declaration for _, _, declaration in axes if declaration]
    return arguments, declarations


def workgroup_size_u32_values(config, values, constants=None):
    """The ``@workgroup_size`` arguments as the ``u32`` components ``gl_WorkGroupSize`` reads (``u32(X)`` of i32)."""
    axes = _specialized_axes(config, values, constants)
    ids = workgroup_size_specialization_ids(config, "WGSL")
    signed = _argument_type(ids, constants or {}) == "i32"
    return tuple(
        f"u32({argument})" if signed and spec_id is not None else argument
        for argument, spec_id, _ in axes
    )


class WorkgroupSizeOverrides:
    """The workgroup-size overrides of one WGSL module, declared once and shared by its entries."""

    def __init__(self, program):
        self._program = program
        self._host_ids = None
        self._declared: dict[str, str] = {}
        self._names_by_id: dict[int, str] = {}

    def effective_config(self, config):
        """``config``, specialized through the program's host-size ids when the host chooses the size."""
        config = config or {}
        if not is_host_sized_workgroup(config):
            return config
        if self._host_ids is None:
            self._host_ids = host_workgroup_size_ids(self._program)
        return host_sized_workgroup_config(config, self._host_ids)

    def new_declarations(self, config, values, constants=None):
        """The declarations of ``config``'s overrides that the module does not declare yet."""
        fresh = []
        for name, spec_id, declaration in _specialized_axes(config, values, constants):
            if declaration is None:
                continue
            owner = self._names_by_id.get(spec_id)
            previous = self._declared.get(name) or self._declared.get(owner)
            if previous == declaration and owner == name:
                continue
            if previous is not None:
                raise SpecializedWorkgroupSizeError(
                    "Entries of one WGSL module specialize their workgroup sizes through module-scope overrides "
                    f"that disagree: '{_one_line(previous)}' and '{_one_line(declaration)}'",
                    target="WGSL",
                    specialization_ids=(spec_id,),
                    reason="conflicting-workgroup-size-override",
                )
            self._declared[name] = declaration
            self._names_by_id[spec_id] = name
            fresh.append(declaration)
        return fresh


def _one_line(declaration: str) -> str:
    return declaration.replace("\n", " ")
