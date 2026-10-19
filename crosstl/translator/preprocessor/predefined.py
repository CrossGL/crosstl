"""The Metal compiler's and DXC's predefined macros, as dialects of the shared preprocessor.

Both compilers are Clang-based, so their preprocessors differ from Clang's C++ preprocessor only in the language
options and the predefined macros. ``tools/preprocessor_macros.py`` records those from the toolchains themselves
into ``data/msl.json`` and ``data/hlsl.json``; this module turns a compiler configuration (language standard,
platform and deployment target, or shader profile and options) into a :class:`~.dialects.Dialect`. A
configuration the recorded toolchain does not cover is refused with a typed diagnostic rather than approximated.
"""

from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Mapping, Sequence

from ..diagnostics import TranslationDiagnostic
from .dialects import Dialect, HeaderMacros
from .lexer import CPP_MODE, LexMode

DATA = Path(__file__).resolve().parent / "data"
CONFIGURATION_CODE = "project.translate.preprocessor-unsupported-configuration"


class ConfigurationError(TranslationDiagnostic):
    code = CONFIGURATION_CODE


@lru_cache(maxsize=None)
def _data(name: str) -> dict:
    return json.loads((DATA / f"{name}.json").read_text(encoding="utf-8"))


class _Sets:
    """Decodes the delta-encoded macro sets of one data file: ``{name: definition}`` per set key."""

    def __init__(self, data: dict):
        self.definitions = data["definitions"]
        self.sets = data["sets"]
        self._cache: dict[str, dict[str, str]] = {}

    def get(self, key: str) -> dict[str, str]:
        if key in self._cache:
            return self._cache[key]
        entry = self.sets[key]
        values = set(self.get(entry["base"]).values()) if entry["base"] else set()
        values -= {self.definitions[index] for index in entry["remove"]}
        values |= {self.definitions[index] for index in entry["add"]}
        macros = {_name(value): value for value in sorted(values)}
        self._cache[key] = macros
        return macros


@lru_cache(maxsize=None)
def _sets(name: str) -> _Sets:
    return _Sets(_data(name))


def _name(definition: str) -> str:
    return re.match(r"[A-Za-z_$][A-Za-z0-9_$]*", definition).group()


def _value(macros: Mapping[str, str], name: str) -> str:
    return macros[name][len(name) :].strip()


def _cplusplus_mode(cplusplus: int, **changes) -> LexMode:
    """Clang's lexical options for a C++ standard: digit separators from C++14, ``u8''`` from C++17."""
    fields = dict(
        digit_separators=cplusplus >= 201402,
        utf8_characters=cplusplus >= 201703,
    )
    fields.update(changes)
    return LexMode(**{**CPP_MODE.__dict__, **fields})


# Metal ---------------------------------------------------------------------------------------------------
METAL_PLATFORMS = ("macos", "ios")


def msl_dialect(
    std: str | None = None,
    *,
    platform: str = "macos",
    os_version: str | None = None,
    options: Sequence[str] = (),
) -> Dialect:
    """The Metal compiler's preprocessor for ``-std=<std>`` on ``platform`` (``macos`` or ``ios``).

    ``std`` defaults to the recorded compiler's default standard and ``os_version`` to the newest recorded
    deployment target. ``options`` names macro-changing compiler options: ``math-mode=relaxed``,
    ``math-mode=safe``, ``fp32-functions=precise``, ``O0``, ``Os`` and ``enable-logging``.
    """
    data = _data("msl")
    if std is None:
        std = data["defaultStandard"]
    elif std in data["aliases"]:
        std = data["aliases"][
            std
        ]  # e.g. osx-metal1.0 for macos-metal1.0; the result is validated below
    if platform not in METAL_PLATFORMS:
        raise ConfigurationError(
            f"Metal platform {platform!r} is not recorded; use one of {METAL_PLATFORMS}"
        )
    standards = data["standards"][platform]
    if std not in standards:
        raise ConfigurationError(
            f"-std={std} is not a {platform} Metal standard of {data['source']['toolchain']}"
        )
    versions = data["osVersions"][platform]
    os_version = os_version or versions[-1]
    if os_version not in versions:
        raise ConfigurationError(
            f"{platform} {os_version} is not a recorded deployment target; recorded: {', '.join(versions)}"
        )
    unknown = sorted(set(options) - set(data["options"]))
    if unknown:
        raise ConfigurationError(f"Metal compiler options {unknown} are not recorded")
    return _msl_dialect(std, platform, os_version, tuple(sorted(set(options))))


@lru_cache(maxsize=64)
def _msl_dialect(
    std: str, platform: str, os_version: str, options: tuple[str, ...]
) -> Dialect:
    data = _data("msl")
    sets = _sets("msl")
    key = f"{platform}/{std}"
    reference = data["osVersions"][platform][-1]
    predefined = dict(
        sets.get(
            f"predefined:{key}" + ("" if os_version == reference else f"@{os_version}")
        )
    )
    changed_headers = set()
    for option in options:
        delta = data["options"][option]
        for name in delta["undefine"]:
            predefined.pop(name, None)
        for definition in delta["define"]:
            predefined[_name(definition)] = definition
        changed_headers.update(delta["changesHeaders"])
    position = data["standards"][platform].index(std)
    names = data["conditionNames"]
    lists = data["headerLists"]
    headers: dict[str, HeaderMacros] = {}
    for header, entry in data["headers"].items():
        closure = entry["closure"][platform][position]
        conditions = frozenset(names[index] for index in entry["conditions"])
        if closure is None:
            refusal = f"<{header}> cannot be included with -std={std}"
            headers[header] = HeaderMacros(
                (), frozenset(), conditions, refusal, available=False
            )
            continue
        refusal = None
        if header in changed_headers:
            refusal = f"<{header}> is not recorded for the Metal compiler options {list(options)}"
        defines = tuple(sets.get(f"header:{header}:{key}").values())
        headers[header] = HeaderMacros(
            defines, frozenset(lists[closure]), conditions, refusal
        )
    cplusplus = int(_value(predefined, "__cplusplus").rstrip("L"))
    return Dialect(
        "msl",
        int_width=int(_value(predefined, "__INTMAX_WIDTH__")),
        lex_mode=_cplusplus_mode(cplusplus, raw_strings=False),
        retained_includes=frozenset(headers),
        headers=headers,
        implicit_headers=frozenset(
            lists[data["implicit"]["headers"][platform][position]]
        ),
        implicit_conditions=frozenset(
            names[index] for index in data["implicit"]["conditions"]
        ),
        predefined_definitions=tuple(predefined[name] for name in sorted(predefined)),
        toolchain=f"{data['source']['toolchain']} -std={std} {platform} {os_version} {' '.join(options)}".rstrip(),
    )


def metal_standards(platform: str = "macos") -> tuple[str, ...]:
    """The recorded ``-std`` values for ``platform``, oldest first."""
    return tuple(_data("msl")["standards"][platform])


def metal_default_standard() -> str:
    """The ``-std`` the recorded Metal compiler uses when none is given."""
    return _data("msl")["defaultStandard"]


def metal_os_versions(platform: str = "macos") -> tuple[str, ...]:
    """The recorded deployment targets for ``platform``, oldest first; the last is the default."""
    return tuple(_data("msl")["osVersions"][platform])


# DXC -----------------------------------------------------------------------------------------------------
def hlsl_dialect(
    profile: str = "cs_6_0",
    *,
    hlsl_version: str = "2021",
    enable_16bit_types: bool = False,
    spirv: str | None = None,
) -> Dialect:
    """DXC's preprocessor for ``-T <profile> -HV <hlsl_version>``, with ``-enable-16bit-types`` and, when
    ``spirv`` names a ``-fspv-target-env``, ``-spirv``. Options DXC rejects are refused the same way.
    """
    data = _data("hlsl")
    stage, _, model = profile.partition("_")
    if stage not in data["stages"] or model not in data["models"].get(stage, ()):
        raise ConfigurationError(
            f"DXC {data['source']['toolchain']} does not recognize the profile {profile!r}"
        )
    if hlsl_version not in data["hlslVersions"]:
        raise ConfigurationError(
            f"-HV {hlsl_version} is not accepted by DXC; accepted: {data['hlslVersions']}"
        )
    if spirv is not None and spirv not in data["spirvEnvironments"]:
        raise ConfigurationError(f"-fspv-target-env={spirv} is not accepted by DXC")
    if enable_16bit_types:
        required = data["sixteenBit"]
        major, minor = (int(part) for part in model.split("_"))
        if (major, minor) < tuple(required["minimumModel"]) or data["hlslVersions"][
            hlsl_version
        ] < required["minimumVersion"]:
            raise ConfigurationError(
                f"-enable-16bit-types needs shader model {required['minimumModel'][0]}.{required['minimumModel'][1]}"
                f" and -HV {required['minimumVersion']} or later"
            )
    return _hlsl_dialect(profile, hlsl_version, enable_16bit_types, spirv)


@lru_cache(maxsize=64)
def _hlsl_dialect(
    profile: str, hlsl_version: str, enable_16bit_types: bool, spirv: str | None
) -> Dialect:
    data = _data("hlsl")
    stage, _, model = profile.partition("_")
    major, minor = model.split("_")
    predefined = dict(_sets("hlsl").get("predefined"))
    computed = {
        "__HLSL_VERSION": data["hlslVersions"][hlsl_version],
        "__SHADER_TARGET_STAGE": data["stages"][stage],
        "__SHADER_TARGET_MAJOR": int(major),
        "__SHADER_TARGET_MINOR": int(minor),
    }
    if enable_16bit_types:
        computed["__HLSL_ENABLE_16_BIT"] = 1
    if spirv is not None:
        spirv_major, spirv_minor = data["spirvEnvironments"][spirv]
        computed.update(
            {
                "__spirv__": 1,
                "__SPIRV_MAJOR_VERSION__": spirv_major,
                "__SPIRV_MINOR_VERSION__": spirv_minor,
            }
        )
    for name, value in computed.items():
        predefined[name] = f"{name} {value}"
    return Dialect(
        "hlsl",
        int_width=64,
        lex_mode=_cplusplus_mode(
            0, digraphs=False, raw_strings=False, unicode_literals=False
        ),
        va_opt=False,
        elifdef=False,
        operator_names=False,
        file_name_macro=False,
        ident_directive=False,
        retained_includes=frozenset(data["builtinHeaders"]),
        headers={
            header: HeaderMacros(
                (),
                frozenset({header}),
                frozenset(),
                f"<{header}> is built into DXC and its expansion is not recorded; put its source on the include "
                f"path to preprocess it",
            )
            for header in data["builtinHeaders"]
        },
        predefined_definitions=tuple(predefined[name] for name in sorted(predefined)),
        toolchain=f"{data['source']['toolchain']} -T {profile} -HV {hlsl_version}"
        + (" -enable-16bit-types" if enable_16bit_types else "")
        + (f" -spirv -fspv-target-env={spirv}" if spirv else ""),
    )


def recorded_toolchain(name: str) -> str:
    """The toolchain version a data file was recorded from (``msl`` or ``hlsl``)."""
    return _data(name)["source"]["toolchain"]
