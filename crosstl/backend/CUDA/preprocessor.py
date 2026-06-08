"""Preprocessor support for CUDA source imports."""

from typing import Dict, List, Optional

from crosstl.backend.DirectX.preprocessor import HLSLPreprocessor

PRESERVED_INCLUDE_SENTINEL = "__CROSSGL_CUDA_PRESERVED_INCLUDE__ "


class CudaPreprocessor(HLSLPreprocessor):
    """Small CUDA preprocessor used before lexing imported source files."""

    DEFAULT_PLATFORM_DEFINES = {
        "__linux__": "1",
        "__unix__": "1",
    }
    PLATFORM_DEFINE_NAMES = {
        "__APPLE__",
        "__MACH__",
        "__linux__",
        "__unix__",
        "_WIN32",
        "_WIN64",
        "WIN32",
        "WIN64",
        "linux",
    }

    def __init__(
        self,
        include_paths: Optional[List[str]] = None,
        defines: Optional[Dict[str, str]] = None,
        strict: bool = False,
        max_expansion_depth: int = 64,
    ):
        merged_defines = defines
        if not strict:
            merged_defines = dict(self.DEFAULT_PLATFORM_DEFINES)
            if defines:
                if any(name in defines for name in self.PLATFORM_DEFINE_NAMES):
                    merged_defines = {}
                merged_defines.update(defines)

        super().__init__(
            include_paths=include_paths,
            defines=merged_defines,
            strict=strict,
            max_expansion_depth=max_expansion_depth,
        )

    def preprocess(self, code: str, file_path: Optional[str] = None) -> str:
        processed = super().preprocess(code, file_path=file_path)
        return processed.replace(PRESERVED_INCLUDE_SENTINEL, "#include ")

    def _handle_include(self, rest: str, file_path: Optional[str]):
        included = super()._handle_include(rest, file_path)
        if included is not None or self.strict:
            return included
        include_target = rest.strip()
        if include_target.startswith("<") and include_target.endswith(">"):
            return f"{PRESERVED_INCLUDE_SENTINEL}{include_target}"
        return None
