"""Preprocessor support for Slang source imports."""

from typing import Dict, List, Optional

from crosstl.backend.DirectX.preprocessor import HLSLPreprocessor


class SlangPreprocessor(HLSLPreprocessor):
    """Small Slang preprocessor used before lexing imported source files."""

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
