"""
Modernized CrossTL Translation Engine.
Provides a clean, extensible interface for universal code translation.
"""

import os
from typing import Optional, Dict, Any
from .backend_registry import BackendRegistry, initialize_builtin_backends
from .common.errors import (
    ErrorCollector,
    report_error,
    report_translation_error,
    SourceLocation,
    ErrorCode,
    ErrorSeverity,
)


class TranslationEngine:
    """Modern translation engine using centralized backend system."""

    def __init__(self):
        # Initialize backends
        initialize_builtin_backends()
        self.error_collector = ErrorCollector()

    def translate_file(
        self,
        input_path: str,
        target_language: str,
        output_path: Optional[str] = None,
        format_output: bool = True,
    ) -> Optional[str]:
        """
        Translate a source file to target language.

        Args:
            input_path: Path to input source file
            target_language: Target language (cuda, metal, directx, etc.)
            output_path: Optional output file path
            format_output: Whether to format the generated code

        Returns:
            Generated code string, or None if translation failed
        """
        try:
            # Validate input file
            if not os.path.exists(input_path):
                report_error(
                    f"Input file not found: {input_path}", ErrorCode.TRANSLATION_FAILED
                )
                return None

            # Read source code
            with open(input_path, "r", encoding="utf-8") as f:
                source_code = f.read()

            # Determine source language from file extension
            source_backend = BackendRegistry.get_backend_by_file(input_path)
            if not source_backend:
                report_error(
                    f"Unsupported source file type: {input_path}",
                    ErrorCode.UNSUPPORTED_FEATURE,
                )
                return None

            # Get target backend
            target_backend = BackendRegistry.get_backend(target_language)
            if not target_backend:
                report_error(
                    f"Unsupported target language: {target_language}",
                    ErrorCode.UNSUPPORTED_FEATURE,
                )
                return None

            # Perform translation
            translated_code = self._perform_translation(
                source_code, source_backend, target_backend, input_path
            )

            if translated_code is None:
                return None

            # Format output if requested
            if format_output:
                translated_code = self._format_code(translated_code, target_language)

            # Save to file if output path specified
            if output_path:
                self._save_output(translated_code, output_path)

            return translated_code

        except Exception as e:
            report_error(f"Translation failed: {str(e)}", ErrorCode.TRANSLATION_FAILED)
            return None

    def translate_code(
        self,
        source_code: str,
        source_language: str,
        target_language: str,
        format_output: bool = True,
    ) -> Optional[str]:
        """
        Translate source code directly.

        Args:
            source_code: Source code string
            source_language: Source language name
            target_language: Target language name
            format_output: Whether to format the generated code

        Returns:
            Generated code string, or None if translation failed
        """
        try:
            # Get backends
            source_backend = BackendRegistry.get_backend(source_language)
            if not source_backend:
                report_error(
                    f"Unsupported source language: {source_language}",
                    ErrorCode.UNSUPPORTED_FEATURE,
                )
                return None

            target_backend = BackendRegistry.get_backend(target_language)
            if not target_backend:
                report_error(
                    f"Unsupported target language: {target_language}",
                    ErrorCode.UNSUPPORTED_FEATURE,
                )
                return None

            # Perform translation
            translated_code = self._perform_translation(
                source_code, source_backend, target_backend
            )

            if translated_code is None:
                return None

            # Format output if requested
            if format_output:
                translated_code = self._format_code(translated_code, target_language)

            return translated_code

        except Exception as e:
            report_error(f"Translation failed: {str(e)}", ErrorCode.TRANSLATION_FAILED)
            return None

    def _perform_translation(
        self,
        source_code: str,
        source_backend,
        target_backend,
        filename: Optional[str] = None,
    ) -> Optional[str]:
        """Perform the actual translation process."""
        try:
            # Step 1: Tokenize source code
            lexer = source_backend.create_lexer(source_code)
            tokens = lexer.tokenize()

            # Step 2: Parse to AST
            parser = source_backend.create_parser(tokens)
            ast = parser.parse()

            # Step 3: Convert to CrossGL IR (if not already CrossGL)
            if source_backend.name != "crossgl":
                crossgl_converter = source_backend.create_to_crossgl_converter()
                crossgl_converter.generate(ast)

                # Parse CrossGL IR to AST for target generation
                # For now, pass the AST directly - in future, parse CrossGL IR
                crossgl_ast = ast
            else:
                crossgl_ast = ast

            # Step 4: Generate target code
            target_generator = target_backend.create_from_crossgl_converter()
            generated_code = target_generator.generate(crossgl_ast)

            return generated_code

        except Exception as e:
            location = SourceLocation(filename) if filename else None
            report_translation_error(str(e), location)
            return None

    def _format_code(self, code: str, language: str) -> str:
        """Format generated code using available formatters."""
        try:
            from .formatter import format_shader_code

            return format_shader_code(code, language)
        except ImportError:
            # Formatter not available, return as-is
            return code
        except Exception as e:
            report_error(
                f"Code formatting failed: {str(e)}",
                ErrorCode.TRANSLATION_FAILED,
                ErrorSeverity.WARNING,
            )
            return code

    def _save_output(self, code: str, output_path: str):
        """Save generated code to file."""
        try:
            # Create output directory if needed
            output_dir = os.path.dirname(output_path)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir)

            with open(output_path, "w", encoding="utf-8") as f:
                f.write(code)

        except Exception as e:
            report_error(
                f"Failed to save output file: {str(e)}", ErrorCode.TRANSLATION_FAILED
            )

    def get_supported_languages(self) -> Dict[str, Any]:
        """Get information about supported languages."""
        backends = BackendRegistry.get_all_backends()
        return {
            name: {"extensions": info.file_extensions, "description": info.description}
            for name, info in backends.items()
        }

    def get_errors(self) -> ErrorCollector:
        """Get the error collector for this engine."""
        return self.error_collector

    def clear_errors(self):
        """Clear all accumulated errors."""
        self.error_collector.clear_errors()


# Global translation engine instance
_global_engine = None


def get_translation_engine() -> TranslationEngine:
    """Get the global translation engine instance."""
    global _global_engine
    if _global_engine is None:
        _global_engine = TranslationEngine()
    return _global_engine


def translate_file(
    input_path: str,
    target_language: str,
    output_path: Optional[str] = None,
    format_output: bool = True,
) -> Optional[str]:
    """
    Convenience function for file translation.

    Args:
        input_path: Path to input source file
        target_language: Target language (cuda, metal, directx, etc.)
        output_path: Optional output file path
        format_output: Whether to format the generated code

    Returns:
        Generated code string, or None if translation failed
    """
    engine = get_translation_engine()
    return engine.translate_file(
        input_path, target_language, output_path, format_output
    )


def translate_code(
    source_code: str,
    source_language: str,
    target_language: str,
    format_output: bool = True,
) -> Optional[str]:
    """
    Convenience function for direct code translation.

    Args:
        source_code: Source code string
        source_language: Source language name
        target_language: Target language name
        format_output: Whether to format the generated code

    Returns:
        Generated code string, or None if translation failed
    """
    engine = get_translation_engine()
    return engine.translate_code(
        source_code, source_language, target_language, format_output
    )


def get_supported_languages() -> Dict[str, Any]:
    """Get information about supported languages."""
    engine = get_translation_engine()
    return engine.get_supported_languages()


def check_errors() -> bool:
    """Check if there are any translation errors."""
    engine = get_translation_engine()
    return engine.get_errors().has_errors()


def print_errors():
    """Print any accumulated errors."""
    engine = get_translation_engine()
    engine.get_errors().print_errors()


def clear_errors():
    """Clear all accumulated errors."""
    engine = get_translation_engine()
    engine.clear_errors()
