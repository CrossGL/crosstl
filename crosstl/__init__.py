"""
CrossTL - Universal Programming Language Translator.
Production-ready system enabling O(n) translation complexity.

Supported Languages:
- CUDA (GPU Computing)
- Metal (Apple GPU)
- DirectX/HLSL (Microsoft GPU)
- OpenGL/GLSL (Cross-platform GPU)
- Vulkan/SPIR-V (Modern GPU)
- Rust (Systems Programming)
- Mojo (AI/ML Programming)
- HIP (AMD GPU)
- Slang (Real-time Rendering)

Example Usage:
    import crosstl

    # Translate a file
    result = crosstl.translate('shader.cu', 'metal')

    # Translate source code directly
    translated = crosstl.translate(source_code, 'cuda', 'metal')
"""

__version__ = "1.0.0"
__author__ = "CrossTL Development Team"
__license__ = "MIT"

# Import existing working components
from . import translator
from .translator.lexer import Lexer
from .translator.parser import Parser
from .translator.codegen import (
    GLSLCodeGen,
    HLSLCodeGen,
    MetalCodeGen,
    VulkanSPIRVCodeGen,
    MojoCodeGen,
    RustCodeGen,
    CudaCodeGen,
    HipCodeGen,
    SlangCodeGen,
)
from .translator.ast import ASTNode
from ._crosstl import translate as legacy_translate

# Import backend components
from .backend.CUDA import CudaLexer, CudaParser, CudaToCrossGLConverter
from .backend.Metal import MetalLexer, MetalParser, MetalToCrossGLConverter
from .backend.DirectX import HLSLLexer, HLSLParser, HLSLToCrossGLConverter
from .backend.GLSL import GLSLLexer, GLSLParser, GLSLToCrossGLConverter
from .backend.SPIRV import VulkanLexer, VulkanParser, VulkanToCrossGLConverter
from .backend.Rust import RustLexer, RustParser, RustToCrossGLConverter
from .backend.Mojo import MojoLexer, MojoParser, MojoToCrossGLConverter
from .backend.HIP import HipLexer, HipParser, HipToCrossGLConverter
from .backend.slang import SlangLexer, SlangParser, SlangToCrossGLConverter

import argparse
import sys
import os

try:
    from .formatter import format_shader_code

    FORMATTER_AVAILABLE = True
except ImportError:
    FORMATTER_AVAILABLE = False


# Supported language mappings
SUPPORTED_LANGUAGES = {
    "cuda": {
        "extensions": [".cu", ".cuh"],
        "lexer": CudaLexer,
        "parser": CudaParser,
        "to_crossgl": CudaToCrossGLConverter,
        "from_crossgl": CudaCodeGen,
    },
    "metal": {
        "extensions": [".metal"],
        "lexer": MetalLexer,
        "parser": MetalParser,
        "to_crossgl": MetalToCrossGLConverter,
        "from_crossgl": MetalCodeGen,
    },
    "directx": {
        "extensions": [".hlsl", ".fx"],
        "lexer": HLSLLexer,
        "parser": HLSLParser,
        "to_crossgl": HLSLToCrossGLConverter,
        "from_crossgl": HLSLCodeGen,
    },
    "opengl": {
        "extensions": [".glsl", ".vert", ".frag", ".comp"],
        "lexer": GLSLLexer,
        "parser": GLSLParser,
        "to_crossgl": GLSLToCrossGLConverter,
        "from_crossgl": GLSLCodeGen,
    },
    "vulkan": {
        "extensions": [".spv", ".spirv"],
        "lexer": VulkanLexer,
        "parser": VulkanParser,
        "to_crossgl": VulkanToCrossGLConverter,
        "from_crossgl": VulkanSPIRVCodeGen,
    },
    "rust": {
        "extensions": [".rs"],
        "lexer": RustLexer,
        "parser": RustParser,
        "to_crossgl": RustToCrossGLConverter,
        "from_crossgl": RustCodeGen,
    },
    "mojo": {
        "extensions": [".mojo", ".🔥"],
        "lexer": MojoLexer,
        "parser": MojoParser,
        "to_crossgl": MojoToCrossGLConverter,
        "from_crossgl": MojoCodeGen,
    },
    "hip": {
        "extensions": [".hip"],
        "lexer": HipLexer,
        "parser": HipParser,
        "to_crossgl": HipToCrossGLConverter,
        "from_crossgl": HipCodeGen,
    },
    "slang": {
        "extensions": [".slang"],
        "lexer": SlangLexer,
        "parser": SlangParser,
        "to_crossgl": SlangToCrossGLConverter,
        "from_crossgl": SlangCodeGen,
    },
    "crossgl": {
        "extensions": [".cgl"],
        "lexer": Lexer,
        "parser": Parser,
        "to_crossgl": None,  # Already in CrossGL format
        "from_crossgl": None,
    },
}


def detect_language(file_path: str, content: str = None) -> str:
    """Detect programming language from file extension or content."""
    from pathlib import Path

    # Try extension first
    ext = Path(file_path).suffix.lower()
    for lang, info in SUPPORTED_LANGUAGES.items():
        if ext in info["extensions"]:
            return lang

    # Try content analysis
    if content:
        content_lower = content.lower()

        # CUDA signatures
        if any(
            sig in content_lower
            for sig in ["__global__", "__device__", "threadidx", "blockidx"]
        ):
            return "cuda"

        # Metal signatures
        if any(
            sig in content_lower
            for sig in ["[[vertex]]", "[[fragment]]", "[[kernel]]", "texture2d"]
        ):
            return "metal"

        # DirectX signatures
        if any(
            sig in content_lower
            for sig in ["sv_position", "sv_target", "cbuffer", "texture2d"]
        ):
            return "directx"

        # OpenGL signatures
        if any(
            sig in content_lower
            for sig in ["gl_position", "gl_fragcolor", "uniform", "#version"]
        ):
            return "opengl"

        # Rust signatures
        if any(
            sig in content_lower
            for sig in ["fn ", "let ", "mut ", "impl ", "struct ", "enum "]
        ):
            return "rust"

        # Mojo signatures
        if any(
            sig in content_lower
            for sig in ["simd[", "from math import", "struct ", "fn "]
        ):
            return "mojo"

        # HIP signatures
        if any(sig in content_lower for sig in ["hipmalloc", "hipfree", "__global__"]):
            return "hip"

    # Default fallback
    return "crossgl"


def translate(
    source_code_or_file: str,
    target_language: str,
    source_language: str = None,
    output_file: str = None,
) -> str:
    """
    Universal translation function.

    Args:
        source_code_or_file: Source code string or path to source file
        target_language: Target language identifier
        source_language: Optional source language (auto-detected if None)
        output_file: Optional output file path

    Returns:
        Translated code string
    """
    # Check if input is file path or code
    if os.path.exists(source_code_or_file):
        # It's a file path
        with open(source_code_or_file, "r") as f:
            source_code = f.read()

        if source_language is None:
            source_language = detect_language(source_code_or_file, source_code)
    else:
        # It's source code
        source_code = source_code_or_file
        if source_language is None:
            source_language = detect_language("temp.code", source_code)

    # Validate languages
    if source_language not in SUPPORTED_LANGUAGES:
        raise ValueError(f"Unsupported source language: {source_language}")
    if target_language not in SUPPORTED_LANGUAGES:
        raise ValueError(f"Unsupported target language: {target_language}")

    # Get language components
    source_info = SUPPORTED_LANGUAGES[source_language]
    target_info = SUPPORTED_LANGUAGES[target_language]

    try:
        # Phase 1: Tokenize
        lexer = source_info["lexer"](source_code)
        tokens = lexer.tokenize()

        # Phase 2: Parse to AST
        parser = source_info["parser"](tokens)
        ast = parser.parse()

        # Phase 3: Convert to CrossGL (if needed)
        if source_language == "crossgl":
            crossgl_ast = ast
        else:
            converter = source_info["to_crossgl"]()
            crossgl_code = converter.generate(ast)

            # Re-parse CrossGL for target generation
            crossgl_lexer = Lexer(crossgl_code)
            crossgl_tokens = crossgl_lexer.get_tokens()
            crossgl_parser = Parser(crossgl_tokens)
            crossgl_ast = crossgl_parser.parse()

        # Phase 4: Generate target code
        if target_language == "crossgl":
            if source_language == "crossgl":
                target_code = source_code
            else:
                target_code = crossgl_code
        else:
            generator = target_info["from_crossgl"]()
            target_code = generator.generate(crossgl_ast)

        # Save to output file if specified
        if output_file:
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            with open(output_file, "w") as f:
                f.write(target_code)

        return target_code

    except Exception as e:
        raise RuntimeError(f"Translation failed: {str(e)}")


def translate_file(
    input_path: str,
    target_language: str,
    output_path: str = None,
    source_language: str = None,
) -> bool:
    """
    Translate a file from one language to another.

    Returns True if successful, False otherwise.
    """
    try:
        result = translate(input_path, target_language, source_language, output_path)
        return result is not None
    except Exception:
        return False


def translate_string(
    source_code: str, source_language: str, target_language: str
) -> str:
    """
    Translate source code string directly.
    """
    return translate(source_code, target_language, source_language)


def get_supported_languages() -> list:
    """Get list of supported languages."""
    return list(SUPPORTED_LANGUAGES.keys())


def info():
    """Print CrossTL system information."""
    print(f"CrossTL v{__version__}")
    print(f"Universal Programming Language Translator")
    print(f"License: {__license__}")
    print()

    languages = get_supported_languages()
    print(f"Supported Languages ({len(languages)}):")
    for lang in sorted(languages):
        extensions = SUPPORTED_LANGUAGES[lang]["extensions"]
        print(f"  - {lang}: {', '.join(extensions)}")


# Public API
__all__ = [
    # Primary functions
    "translate",
    "translate_file",
    "translate_string",
    "get_supported_languages",
    "info",
    # Core classes for advanced usage
    "Lexer",
    "Parser",
    "ASTNode",
    # Code generators
    "GLSLCodeGen",
    "HLSLCodeGen",
    "MetalCodeGen",
    "VulkanSPIRVCodeGen",
    "MojoCodeGen",
    "RustCodeGen",
    "CudaCodeGen",
    "HipCodeGen",
    "SlangCodeGen",
    # Legacy
    "legacy_translate",
]


def main():
    """Command-line interface entry point."""
    parser = argparse.ArgumentParser(
        description="CrossTL - Universal Programming Language Translator",
        epilog="Examples:\n"
        "  crosstl shader.cu metal\n"
        "  crosstl --source cuda --target metal input.cu output.metal\n"
        "  crosstl --info\n",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument("input", nargs="?", help="Input file or source code")
    parser.add_argument("target", nargs="?", help="Target language")
    parser.add_argument(
        "-s", "--source", help="Source language (auto-detected if not specified)"
    )
    parser.add_argument("-o", "--output", help="Output file path")
    parser.add_argument("--info", action="store_true", help="Show system information")
    parser.add_argument(
        "--list-languages", action="store_true", help="List supported languages"
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    parser.add_argument("--version", action="version", version=f"CrossTL {__version__}")

    args = parser.parse_args()

    # Handle info commands
    if args.info:
        info()
        return 0

    if args.list_languages:
        languages = get_supported_languages()
        print("Supported languages:")
        for lang in sorted(languages):
            print(f"  {lang}")
        return 0

    # Require input and target for translation
    if not args.input or not args.target:
        print("Error: Input file and target language are required", file=sys.stderr)
        parser.print_help()
        return 1

    try:
        # Perform translation
        result = translate(args.input, args.target, args.source, args.output)

        if args.output:
            print(f"Translation successful: {args.input} -> {args.output}")
        else:
            print(result)

        return 0

    except Exception as e:
        print(f"Translation failed: {e}", file=sys.stderr)
        if args.verbose:
            import traceback

            traceback.print_exc()

        return 1


if __name__ == "__main__":
    sys.exit(main())
