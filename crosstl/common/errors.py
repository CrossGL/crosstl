"""
Standardized error handling system for CrossTL.
Provides consistent error reporting across all backends.
"""

from typing import Optional, List
from enum import Enum


class ErrorSeverity(Enum):
    """Error severity levels."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class ErrorCode(Enum):
    """Standardized error codes."""

    # Lexical errors
    ILLEGAL_CHARACTER = "E001"
    UNTERMINATED_STRING = "E002"
    UNTERMINATED_COMMENT = "E003"
    INVALID_NUMBER_FORMAT = "E004"

    # Syntax errors
    UNEXPECTED_TOKEN = "E101"
    MISSING_SEMICOLON = "E102"
    MISSING_BRACE = "E103"
    MISSING_PARENTHESIS = "E104"
    INVALID_EXPRESSION = "E105"
    INVALID_DECLARATION = "E106"

    # Semantic errors
    UNDEFINED_IDENTIFIER = "E201"
    TYPE_MISMATCH = "E202"
    INVALID_FUNCTION_CALL = "E203"
    INVALID_OPERATION = "E204"
    REDEFINITION = "E205"

    # Backend-specific errors
    UNSUPPORTED_FEATURE = "E301"
    INVALID_ATTRIBUTE = "E302"
    MISSING_QUALIFIER = "E303"
    INVALID_SEMANTIC = "E304"

    # Translation errors
    TRANSLATION_FAILED = "E401"
    INCOMPATIBLE_TYPE = "E402"
    UNSUPPORTED_CONSTRUCT = "E403"


class SourceLocation:
    """Source code location information."""

    def __init__(
        self,
        filename: Optional[str] = None,
        line: int = 1,
        column: int = 1,
        offset: int = 0,
    ):
        self.filename = filename
        self.line = line
        self.column = column
        self.offset = offset

    def __str__(self) -> str:
        if self.filename:
            return f"{self.filename}:{self.line}:{self.column}"
        else:
            return f"{self.line}:{self.column}"

    def __repr__(self) -> str:
        return f"SourceLocation(filename={self.filename}, line={self.line}, column={self.column})"


class CrossTLError(Exception):
    """Base exception class for all CrossTL errors."""

    def __init__(
        self,
        message: str,
        error_code: ErrorCode,
        severity: ErrorSeverity = ErrorSeverity.ERROR,
        location: Optional[SourceLocation] = None,
        context: Optional[str] = None,
    ):
        self.message = message
        self.error_code = error_code
        self.severity = severity
        self.location = location
        self.context = context

        # Format error message
        formatted_message = f"[{error_code.value}] {message}"
        if location:
            formatted_message = f"{location}: {formatted_message}"
        if context:
            formatted_message += f" (in {context})"

        super().__init__(formatted_message)

    def __str__(self) -> str:
        return self.args[0]


class LexicalError(CrossTLError):
    """Lexical analysis errors."""

    def __init__(
        self, message: str, location: Optional[SourceLocation] = None, **kwargs
    ):
        super().__init__(
            message,
            ErrorCode.ILLEGAL_CHARACTER,
            ErrorSeverity.ERROR,
            location,
            **kwargs,
        )


class SyntaxError(CrossTLError):
    """Syntax analysis errors."""

    def __init__(
        self, message: str, location: Optional[SourceLocation] = None, **kwargs
    ):
        super().__init__(
            message, ErrorCode.UNEXPECTED_TOKEN, ErrorSeverity.ERROR, location, **kwargs
        )


class SemanticError(CrossTLError):
    """Semantic analysis errors."""

    def __init__(
        self, message: str, location: Optional[SourceLocation] = None, **kwargs
    ):
        super().__init__(
            message, ErrorCode.TYPE_MISMATCH, ErrorSeverity.ERROR, location, **kwargs
        )


class TranslationError(CrossTLError):
    """Translation/code generation errors."""

    def __init__(
        self, message: str, location: Optional[SourceLocation] = None, **kwargs
    ):
        super().__init__(
            message,
            ErrorCode.TRANSLATION_FAILED,
            ErrorSeverity.ERROR,
            location,
            **kwargs,
        )


class ErrorCollector:
    """Collects and manages errors during compilation."""

    def __init__(self):
        self.errors: List[CrossTLError] = []
        self.max_errors = 100  # Limit to prevent overflow

    def add_error(self, error: CrossTLError):
        """Add an error to the collection."""
        if len(self.errors) >= self.max_errors:
            return
        self.errors.append(error)

    def add_lexical_error(
        self, message: str, location: Optional[SourceLocation] = None
    ):
        """Add a lexical error."""
        self.add_error(LexicalError(message, location))

    def add_syntax_error(self, message: str, location: Optional[SourceLocation] = None):
        """Add a syntax error."""
        self.add_error(SyntaxError(message, location))

    def add_semantic_error(
        self, message: str, location: Optional[SourceLocation] = None
    ):
        """Add a semantic error."""
        self.add_error(SemanticError(message, location))

    def add_translation_error(
        self, message: str, location: Optional[SourceLocation] = None
    ):
        """Add a translation error."""
        self.add_error(TranslationError(message, location))

    def has_errors(self) -> bool:
        """Check if there are any errors."""
        return len(self.errors) > 0

    def has_critical_errors(self) -> bool:
        """Check if there are any critical errors."""
        return any(error.severity == ErrorSeverity.CRITICAL for error in self.errors)

    def get_error_count(self) -> int:
        """Get the number of errors."""
        return len(self.errors)

    def get_errors_by_severity(self, severity: ErrorSeverity) -> List[CrossTLError]:
        """Get errors of a specific severity."""
        return [error for error in self.errors if error.severity == severity]

    def clear_errors(self):
        """Clear all errors."""
        self.errors.clear()

    def format_errors(self) -> str:
        """Format all errors for display."""
        if not self.errors:
            return "No errors."

        lines = []
        for error in self.errors:
            lines.append(str(error))

        return "\n".join(lines)

    def print_errors(self):
        """Print all errors to console."""
        if self.errors:
            print("CrossTL Compilation Errors:")
            print(self.format_errors())
        else:
            print("No errors.")


# Global error collector instance
_global_error_collector = ErrorCollector()


def get_error_collector() -> ErrorCollector:
    """Get the global error collector."""
    return _global_error_collector


def reset_errors():
    """Reset the global error collector."""
    _global_error_collector.clear_errors()


def report_error(
    message: str,
    error_code: ErrorCode = ErrorCode.TRANSLATION_FAILED,
    severity: ErrorSeverity = ErrorSeverity.ERROR,
    location: Optional[SourceLocation] = None,
    context: Optional[str] = None,
):
    """Report an error to the global collector."""
    error = CrossTLError(message, error_code, severity, location, context)
    _global_error_collector.add_error(error)


def report_lexical_error(message: str, location: Optional[SourceLocation] = None):
    """Report a lexical error."""
    _global_error_collector.add_lexical_error(message, location)


def report_syntax_error(message: str, location: Optional[SourceLocation] = None):
    """Report a syntax error."""
    _global_error_collector.add_syntax_error(message, location)


def report_semantic_error(message: str, location: Optional[SourceLocation] = None):
    """Report a semantic error."""
    _global_error_collector.add_semantic_error(message, location)


def report_translation_error(message: str, location: Optional[SourceLocation] = None):
    """Report a translation error."""
    _global_error_collector.add_translation_error(message, location)


# Utility functions for creating common errors


def create_unexpected_token_error(
    expected: str, actual: str, location: Optional[SourceLocation] = None
) -> SyntaxError:
    """Create an unexpected token error."""
    return SyntaxError(f"Expected {expected}, got {actual}", location)


def create_undefined_identifier_error(
    identifier: str, location: Optional[SourceLocation] = None
) -> SemanticError:
    """Create an undefined identifier error."""
    return SemanticError(f"Undefined identifier '{identifier}'", location)


def create_type_mismatch_error(
    expected: str, actual: str, location: Optional[SourceLocation] = None
) -> SemanticError:
    """Create a type mismatch error."""
    return SemanticError(f"Type mismatch: expected {expected}, got {actual}", location)


def create_unsupported_feature_error(
    feature: str, backend: str, location: Optional[SourceLocation] = None
) -> TranslationError:
    """Create an unsupported feature error."""
    return TranslationError(
        f"Feature '{feature}' is not supported in {backend} backend", location
    )
