"""DirectX compiler requirements derived from generated HLSL."""

from __future__ import annotations

import ntpath
import re
import shutil
import sys
from pathlib import Path

_HLSL_NATIVE_16_BIT_TYPE_RE = re.compile(
    r"(?<![A-Za-z0-9_])(?:float16_t|int16_t|uint16_t)(?:[1-4])?" r"(?![A-Za-z0-9_])"
)
_HLSL_EXACT_WAVE_SIZE_RE = re.compile(
    r"\[\s*WaveSize\s*\(\s*[^,\]]+\s*\)\s*\]",
    re.IGNORECASE,
)
_DXC_PROFILE_RE = re.compile(r"^(?P<stage>[a-z]+)_(?P<major>\d+)_(?P<minor>\d+)$")
_DXC_NATIVE_16_BIT_MINIMUM_PROFILE = (6, 2)
_DXC_EXACT_WAVE_SIZE_MINIMUM_PROFILE = (6, 6)
_DXC_NATIVE_16_BIT_ARGUMENTS = ("-enable-16bit-types",)
_DIRECTX_TARGET_PROFILES = ("directx-11", "directx-12")
_DIRECTX_12_TARGET_PROFILES = ("directx-12",)


def dxc_file_path(path: Path) -> str:
    """Use extended-length Windows paths without relocating compiler inputs."""

    value = str(path)
    if sys.platform != "win32" or value.startswith(("\\\\?\\", "\\\\.\\")):
        return value
    absolute = ntpath.abspath(value)
    if len(absolute.encode("utf-16-le")) // 2 < 260:
        return value
    if absolute.startswith("\\\\"):
        return "\\\\?\\UNC\\" + absolute[2:]
    return "\\\\?\\" + absolute


def dxc_long_path_command(command: list[str], source_path: Path) -> list[str]:
    """Use DXC's include API when Windows CLI paths need normalization."""

    if sys.platform != "win32":
        return command
    paths = [str(source_path)] + [
        command[index + 1]
        for index, value in enumerate(command[:-1])
        if value in ("-I", "-Fo")
    ]
    if not any(len(value.encode("utf-16-le")) // 2 >= 260 for value in paths):
        return command
    if shutil.which(command[0]) is None:
        return command
    source = dxc_file_path(source_path)
    output_index = command.index("-Fo")
    source_index = command.index(source)
    arguments = [
        value
        for index, value in enumerate(command)
        if index not in (0, source_index, output_index, output_index + 1)
    ]
    return [
        sys.executable,
        str(Path(__file__).with_name("dxc_compiler.py")),
        "--compiler",
        command[0],
        "--source",
        str(source_path),
        "--output",
        command[output_index + 1],
        "--",
        *arguments,
    ]


def dxc_library_command_tool(command: list[str]) -> str | None:
    """Identify the compiler in a recorded long-path bridge invocation."""

    if (
        len(command) < 9
        or any(not isinstance(value, str) or not value for value in command)
        or command[2:9:2] != ["--compiler", "--source", "--output", "--"]
        or not command[1]
        .replace("\\", "/")
        .endswith("/crosstl/project/dxc_compiler.py")
        or re.fullmatch(
            r"python(?:\d+(?:\.\d+)*)?(?:\.exe)?",
            ntpath.basename(command[0]),
            re.IGNORECASE,
        )
        is None
    ):
        return None
    return command[3]


def _mask_hlsl_comments_and_literals(source: str) -> str:
    """Replace comments and quoted literals with whitespace."""

    text = str(source or "")
    masked = list(text)
    index = 0

    def mask(position: int) -> None:
        if text[position] not in "\r\n":
            masked[position] = " "

    while index < len(text):
        if text.startswith("//", index):
            while index < len(text) and text[index] not in "\r\n":
                mask(index)
                index += 1
            continue

        if text.startswith("/*", index):
            mask(index)
            mask(index + 1)
            index += 2
            while index < len(text):
                if text.startswith("*/", index):
                    mask(index)
                    mask(index + 1)
                    index += 2
                    break
                mask(index)
                index += 1
            continue

        quote = text[index]
        if quote not in {'"', "'"}:
            index += 1
            continue

        mask(index)
        index += 1
        while index < len(text):
            character = text[index]
            mask(index)
            index += 1
            if character == quote:
                break
            if character != "\\" or index >= len(text):
                continue
            if (
                text[index] == "\r"
                and index + 1 < len(text)
                and text[index + 1] == "\n"
            ):
                mask(index)
                mask(index + 1)
                index += 2
            else:
                mask(index)
                index += 1

    return "".join(masked)


def hlsl_requires_native_16bit_types(source: str) -> bool:
    """Return whether HLSL uses native-width 16-bit scalar or vector types."""

    code = _mask_hlsl_comments_and_literals(source)
    return _HLSL_NATIVE_16_BIT_TYPE_RE.search(code) is not None


def dxc_profile_for_source(profile: str, source: str) -> str:
    """Raise a DXC profile to satisfy generated HLSL feature requirements."""

    code = _mask_hlsl_comments_and_literals(source)
    minimum_profile = None
    if _HLSL_NATIVE_16_BIT_TYPE_RE.search(code) is not None:
        minimum_profile = _DXC_NATIVE_16_BIT_MINIMUM_PROFILE
    if _HLSL_EXACT_WAVE_SIZE_RE.search(code) is not None:
        minimum_profile = max(
            minimum_profile or _DXC_EXACT_WAVE_SIZE_MINIMUM_PROFILE,
            _DXC_EXACT_WAVE_SIZE_MINIMUM_PROFILE,
        )
    if minimum_profile is None:
        return profile
    match = _DXC_PROFILE_RE.fullmatch(str(profile or "").strip().lower())
    if match is None:
        return profile
    version = int(match.group("major")), int(match.group("minor"))
    if version >= minimum_profile:
        return profile
    return f"{match.group('stage')}_{minimum_profile[0]}_{minimum_profile[1]}"


def dxc_compiler_arguments_for_source(source: str) -> tuple[str, ...]:
    """Return compiler arguments required by generated HLSL types."""

    if hlsl_requires_native_16bit_types(source):
        return _DXC_NATIVE_16_BIT_ARGUMENTS
    return ()


def directx_target_profiles_for_source(source: str) -> tuple[str, ...]:
    """Return DirectX API profiles compatible with the generated source."""

    code = _mask_hlsl_comments_and_literals(source)
    if (
        _HLSL_NATIVE_16_BIT_TYPE_RE.search(code) is not None
        or _HLSL_EXACT_WAVE_SIZE_RE.search(code) is not None
    ):
        return _DIRECTX_12_TARGET_PROFILES
    return _DIRECTX_TARGET_PROFILES
