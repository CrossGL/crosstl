"""DirectX compiler requirements derived from generated HLSL."""

from __future__ import annotations

import re

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
