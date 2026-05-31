"""Preprocessor support for HIP source imports."""

from typing import Dict, List, Optional

from crosstl.backend.DirectX.preprocessor import HLSLPreprocessor

PRESERVED_INCLUDE_SENTINEL = "__CROSSGL_HIP_PRESERVED_INCLUDE__ "


class HipPreprocessor(HLSLPreprocessor):
    """Small HIP preprocessor used before lexing imported source files."""

    def __init__(
        self,
        include_paths: Optional[List[str]] = None,
        defines: Optional[Dict[str, str]] = None,
        strict: bool = False,
        max_expansion_depth: int = 64,
    ):
        super().__init__(
            include_paths=include_paths,
            defines=defines,
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
