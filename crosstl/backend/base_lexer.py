"""
Base Lexer Infrastructure for all language backends.
This module provides common lexer functionality to reduce code duplication.
"""

import re
from typing import Iterator, Tuple, List, Dict, Optional, Set
from enum import Enum, auto
from abc import ABC, abstractmethod


class TokenType(Enum):
    """Standard token types used across all language backends."""
    
    # Literals
    IDENTIFIER = auto()
    NUMBER = auto()
    INTEGER = auto()
    FLOAT = auto()
    STRING = auto()
    CHAR_LIT = auto()
    BOOLEAN = auto()
    
    # Keywords - Control Flow
    IF = auto()
    ELSE = auto()
    ELIF = auto()
    FOR = auto()
    WHILE = auto()
    DO = auto()
    SWITCH = auto()
    CASE = auto()
    DEFAULT = auto()
    BREAK = auto()
    CONTINUE = auto()
    RETURN = auto()
    
    # Keywords - Declarations
    STRUCT = auto()
    FUNCTION = auto()
    VARIABLE = auto()
    CONSTANT = auto()
    TYPEDEF = auto()
    ENUM = auto()
    CLASS = auto()
    INTERFACE = auto()
    TRAIT = auto()
    IMPL = auto()
    
    # Keywords - Types
    VOID = auto()
    BOOL = auto()
    INT = auto()
    UINT = auto()
    FLOAT_TYPE = auto()
    DOUBLE = auto()
    CHAR = auto()
    SHORT = auto()
    LONG = auto()
    
    # Keywords - Modifiers
    CONST = auto()
    STATIC = auto()
    INLINE = auto()
    EXTERN = auto()
    PUBLIC = auto()
    PRIVATE = auto()
    PROTECTED = auto()
    MUTABLE = auto()
    VOLATILE = auto()
    
    # Operators - Arithmetic
    PLUS = auto()
    MINUS = auto()
    MULTIPLY = auto()
    DIVIDE = auto()
    MODULO = auto()
    POWER = auto()
    
    # Operators - Assignment
    ASSIGN = auto()
    PLUS_EQUALS = auto()
    MINUS_EQUALS = auto()
    MULTIPLY_EQUALS = auto()
    DIVIDE_EQUALS = auto()
    MODULO_EQUALS = auto()
    
    # Operators - Comparison
    EQUAL = auto()
    NOT_EQUAL = auto()
    LESS_THAN = auto()
    GREATER_THAN = auto()
    LESS_EQUAL = auto()
    GREATER_EQUAL = auto()
    
    # Operators - Logical
    LOGICAL_AND = auto()
    LOGICAL_OR = auto()
    LOGICAL_NOT = auto()
    
    # Operators - Bitwise
    BITWISE_AND = auto()
    BITWISE_OR = auto()
    BITWISE_XOR = auto()
    BITWISE_NOT = auto()
    SHIFT_LEFT = auto()
    SHIFT_RIGHT = auto()
    
    # Operators - Misc
    DOT = auto()
    ARROW = auto()
    SCOPE = auto()
    QUESTION = auto()
    COLON = auto()
    DOUBLE_COLON = auto()
    FAT_ARROW = auto()
    
    # Delimiters
    LPAREN = auto()
    RPAREN = auto()
    LBRACE = auto()
    RBRACE = auto()
    LBRACKET = auto()
    RBRACKET = auto()
    SEMICOLON = auto()
    COMMA = auto()
    
    # Special
    EOF = auto()
    NEWLINE = auto()
    WHITESPACE = auto()
    COMMENT_SINGLE = auto()
    COMMENT_MULTI = auto()
    PREPROCESSOR = auto()
    
    # Language-specific (to be extended by subclasses)
    LANGUAGE_SPECIFIC = auto()


class Token:
    """Standard token representation."""
    
    def __init__(self, token_type: TokenType, value: str, line: int = 1, column: int = 1):
        self.type = token_type
        self.value = value
        self.line = line
        self.column = column
    
    def __repr__(self) -> str:
        return f"Token({self.type.name}, '{self.value}', {self.line}:{self.column})"
    
    def __eq__(self, other) -> bool:
        if isinstance(other, Token):
            return self.type == other.type and self.value == other.value
        return False


class BaseLexer(ABC):
    """Base lexer class providing common functionality for all language backends."""
    
    # Common skip tokens
    SKIP_TOKENS = {TokenType.WHITESPACE, TokenType.COMMENT_SINGLE, TokenType.COMMENT_MULTI}
    
    # Common token patterns (can be overridden by subclasses)
    BASE_PATTERNS = [
        # Comments (should come first)
        (TokenType.COMMENT_SINGLE, r"//.*"),
        (TokenType.COMMENT_MULTI, r"/\*[\s\S]*?\*/"),
        
        # Multi-character operators (must come before single-character ones)
        (TokenType.DOUBLE_COLON, r"::"),
        (TokenType.SHIFT_LEFT, r"<<"),
        (TokenType.SHIFT_RIGHT, r">>"),
        (TokenType.PLUS_EQUALS, r"\+="),
        (TokenType.MINUS_EQUALS, r"-="),
        (TokenType.MULTIPLY_EQUALS, r"\*="),
        (TokenType.DIVIDE_EQUALS, r"/="),
        (TokenType.MODULO_EQUALS, r"%="),
        (TokenType.LOGICAL_AND, r"&&"),
        (TokenType.LOGICAL_OR, r"\|\|"),
        (TokenType.EQUAL, r"=="),
        (TokenType.NOT_EQUAL, r"!="),
        (TokenType.LESS_EQUAL, r"<="),
        (TokenType.GREATER_EQUAL, r">="),
        (TokenType.ARROW, r"->"),
        (TokenType.FAT_ARROW, r"=>"),
        
        # Keywords (will be expanded by subclasses)
        (TokenType.IF, r"\bif\b"),
        (TokenType.ELSE, r"\belse\b"),
        (TokenType.FOR, r"\bfor\b"),
        (TokenType.WHILE, r"\bwhile\b"),
        (TokenType.DO, r"\bdo\b"),
        (TokenType.RETURN, r"\breturn\b"),
        (TokenType.STRUCT, r"\bstruct\b"),
        (TokenType.VOID, r"\bvoid\b"),
        (TokenType.BOOL, r"\bbool\b"),
        (TokenType.INT, r"\bint\b"),
        (TokenType.FLOAT_TYPE, r"\bfloat\b"),
        (TokenType.BREAK, r"\bbreak\b"),
        (TokenType.CONTINUE, r"\bcontinue\b"),
        (TokenType.SWITCH, r"\bswitch\b"),
        (TokenType.CASE, r"\bcase\b"),
        (TokenType.DEFAULT, r"\bdefault\b"),
        
        # Identifiers and literals
        (TokenType.IDENTIFIER, r"[a-zA-Z_][a-zA-Z0-9_]*"),
        (TokenType.NUMBER, r"\d+(\.\d+)?([eE][+-]?\d+)?[fFdDlLuU]*"),
        (TokenType.STRING, r'"([^"\\]|\\.)*"'),
        (TokenType.CHAR_LIT, r"'([^'\\]|\\.)'"),
        
        # Single-character operators and delimiters
        (TokenType.PLUS, r"\+"),
        (TokenType.MINUS, r"-"),
        (TokenType.MULTIPLY, r"\*"),
        (TokenType.DIVIDE, r"/"),
        (TokenType.MODULO, r"%"),
        (TokenType.ASSIGN, r"="),
        (TokenType.LESS_THAN, r"<"),
        (TokenType.GREATER_THAN, r">"),
        (TokenType.LOGICAL_NOT, r"!"),
        (TokenType.BITWISE_AND, r"&"),
        (TokenType.BITWISE_OR, r"\|"),
        (TokenType.BITWISE_XOR, r"\^"),
        (TokenType.BITWISE_NOT, r"~"),
        (TokenType.QUESTION, r"\?"),
        (TokenType.COLON, r":"),
        (TokenType.DOT, r"\."),
        (TokenType.LPAREN, r"\("),
        (TokenType.RPAREN, r"\)"),
        (TokenType.LBRACE, r"\{"),
        (TokenType.RBRACE, r"\}"),
        (TokenType.LBRACKET, r"\["),
        (TokenType.RBRACKET, r"\]"),
        (TokenType.SEMICOLON, r";"),
        (TokenType.COMMA, r","),
        
        # Whitespace (must be last)
        (TokenType.WHITESPACE, r"\s+"),
    ]
    
    def __init__(self, code: str):
        self.code = code
        self.length = len(code)
        self.line = 1
        self.column = 1
        
        # Compile patterns
        self.token_patterns = []
        patterns = self.get_token_patterns()
        for token_type, pattern in patterns:
            self.token_patterns.append((token_type, re.compile(pattern)))
        
        # Keywords mapping
        self.keywords = self.get_keywords()
        
    @abstractmethod
    def get_token_patterns(self) -> List[Tuple[TokenType, str]]:
        """Get language-specific token patterns. Must be implemented by subclasses."""
        pass
    
    @abstractmethod
    def get_keywords(self) -> Dict[str, TokenType]:
        """Get language-specific keyword mappings. Must be implemented by subclasses."""
        pass
    
    def tokenize(self) -> List[Token]:
        """Tokenize the input code and return list of tokens."""
        tokens = []
        pos = 0
        
        while pos < self.length:
            # Track position for error reporting
            old_pos = pos
            
            # Try to match a token
            token_result = self._next_token(pos)
            if token_result is None:
                raise SyntaxError(f"Illegal character '{self.code[pos]}' at line {self.line}, column {self.column}")
            
            new_pos, token_type, text = token_result
            
            # Handle keyword recognition
            if token_type == TokenType.IDENTIFIER and text in self.keywords:
                token_type = self.keywords[text]
            
            # Create token with position info
            token = Token(token_type, text, self.line, self.column)
            
            # Update position tracking
            for char in text:
                if char == '\n':
                    self.line += 1
                    self.column = 1
                else:
                    self.column += 1
            
            # Skip certain token types
            if token_type not in self.SKIP_TOKENS:
                tokens.append(token)
            
            pos = new_pos
        
        # Add EOF token
        tokens.append(Token(TokenType.EOF, "", self.line, self.column))
        return tokens
    
    def _next_token(self, pos: int) -> Optional[Tuple[int, TokenType, str]]:
        """Find the next token starting at the given position."""
        for token_type, pattern in self.token_patterns:
            match = pattern.match(self.code, pos)
            if match:
                return match.end(), token_type, match.group(0)
        return None
    
    @classmethod
    def from_file(cls, filepath: str) -> 'BaseLexer':
        """Create a lexer instance from a file."""
        with open(filepath, 'r', encoding='utf-8') as f:
            return cls(f.read())


class StandardLexer(BaseLexer):
    """Standard lexer implementation for C-like languages."""
    
    def get_token_patterns(self) -> List[Tuple[TokenType, str]]:
        """Standard token patterns for C-like languages."""
        return self.BASE_PATTERNS
    
    def get_keywords(self) -> Dict[str, TokenType]:
        """Standard keyword mappings for C-like languages."""
        return {
            "if": TokenType.IF,
            "else": TokenType.ELSE,
            "for": TokenType.FOR,
            "while": TokenType.WHILE,
            "do": TokenType.DO,
            "switch": TokenType.SWITCH,
            "case": TokenType.CASE,
            "default": TokenType.DEFAULT,
            "break": TokenType.BREAK,
            "continue": TokenType.CONTINUE,
            "return": TokenType.RETURN,
            "struct": TokenType.STRUCT,
            "void": TokenType.VOID,
            "bool": TokenType.BOOL,
            "int": TokenType.INT,
            "float": TokenType.FLOAT_TYPE,
            "double": TokenType.DOUBLE,
            "char": TokenType.CHAR,
            "short": TokenType.SHORT,
            "long": TokenType.LONG,
            "const": TokenType.CONST,
            "static": TokenType.STATIC,
            "inline": TokenType.INLINE,
            "extern": TokenType.EXTERN,
            "typedef": TokenType.TYPEDEF,
            "enum": TokenType.ENUM,
            "true": TokenType.BOOLEAN,
            "false": TokenType.BOOLEAN,
        }


def create_compatibility_tokens(tokens: List[Token]) -> List[Tuple[str, str]]:
    """Convert Token objects to tuple format for backward compatibility."""
    return [(token.type.name, token.value) for token in tokens]


def create_legacy_token_tuple(token: Token) -> Tuple[str, str]:
    """Convert a single Token to tuple format."""
    return (token.type.name, token.value)


class LexerFactory:
    """Factory for creating appropriate lexers based on language."""
    
    _lexer_registry: Dict[str, type] = {}
    
    @classmethod
    def register_lexer(cls, language: str, lexer_class: type):
        """Register a lexer for a specific language."""
        cls._lexer_registry[language.lower()] = lexer_class
    
    @classmethod
    def create_lexer(cls, language: str, code: str) -> BaseLexer:
        """Create a lexer for the specified language."""
        language = language.lower()
        if language in cls._lexer_registry:
            return cls._lexer_registry[language](code)
        else:
            # Fallback to standard lexer
            return StandardLexer(code)
    
    @classmethod
    def get_supported_languages(cls) -> List[str]:
        """Get list of supported languages."""
        return list(cls._lexer_registry.keys())


# Common token patterns that can be reused
COMMON_COMMENT_PATTERNS = [
    (TokenType.COMMENT_SINGLE, r"//.*"),
    (TokenType.COMMENT_MULTI, r"/\*[\s\S]*?\*/"),
]

COMMON_OPERATOR_PATTERNS = [
    # Multi-character operators (order matters!)
    (TokenType.DOUBLE_COLON, r"::"),
    (TokenType.SHIFT_LEFT, r"<<"),
    (TokenType.SHIFT_RIGHT, r">>"),
    (TokenType.PLUS_EQUALS, r"\+="),
    (TokenType.MINUS_EQUALS, r"-="),
    (TokenType.MULTIPLY_EQUALS, r"\*="),
    (TokenType.DIVIDE_EQUALS, r"/="),
    (TokenType.MODULO_EQUALS, r"%="),
    (TokenType.LOGICAL_AND, r"&&"),
    (TokenType.LOGICAL_OR, r"\|\|"),
    (TokenType.EQUAL, r"=="),
    (TokenType.NOT_EQUAL, r"!="),
    (TokenType.LESS_EQUAL, r"<="),
    (TokenType.GREATER_EQUAL, r">="),
    (TokenType.ARROW, r"->"),
    (TokenType.FAT_ARROW, r"=>"),
    
    # Single-character operators
    (TokenType.PLUS, r"\+"),
    (TokenType.MINUS, r"-"),
    (TokenType.MULTIPLY, r"\*"),
    (TokenType.DIVIDE, r"/"),
    (TokenType.MODULO, r"%"),
    (TokenType.ASSIGN, r"="),
    (TokenType.LESS_THAN, r"<"),
    (TokenType.GREATER_THAN, r">"),
    (TokenType.LOGICAL_NOT, r"!"),
    (TokenType.BITWISE_AND, r"&"),
    (TokenType.BITWISE_OR, r"\|"),
    (TokenType.BITWISE_XOR, r"\^"),
    (TokenType.BITWISE_NOT, r"~"),
    (TokenType.QUESTION, r"\?"),
    (TokenType.COLON, r":"),
    (TokenType.DOT, r"\."),
]

COMMON_DELIMITER_PATTERNS = [
    (TokenType.LPAREN, r"\("),
    (TokenType.RPAREN, r"\)"),
    (TokenType.LBRACE, r"\{"),
    (TokenType.RBRACE, r"\}"),
    (TokenType.LBRACKET, r"\["),
    (TokenType.RBRACKET, r"\]"),
    (TokenType.SEMICOLON, r";"),
    (TokenType.COMMA, r","),
]

COMMON_LITERAL_PATTERNS = [
    (TokenType.IDENTIFIER, r"[a-zA-Z_][a-zA-Z0-9_]*"),
    (TokenType.NUMBER, r"\d+(\.\d+)?([eE][+-]?\d+)?[fFdDlLuU]*"),
    (TokenType.STRING, r'"([^"\\]|\\.)*"'),
    (TokenType.CHAR_LIT, r"'([^'\\]|\\.)'"),
]

COMMON_WHITESPACE_PATTERNS = [
    (TokenType.WHITESPACE, r"\s+"),
]


def build_standard_patterns() -> List[Tuple[TokenType, str]]:
    """Build standard token patterns in the correct order."""
    patterns = []
    patterns.extend(COMMON_COMMENT_PATTERNS)
    patterns.extend(COMMON_OPERATOR_PATTERNS)
    patterns.extend(COMMON_LITERAL_PATTERNS)
    patterns.extend(COMMON_DELIMITER_PATTERNS)
    patterns.extend(COMMON_WHITESPACE_PATTERNS)
    return patterns


class LanguageSpecificMixin:
    """Mixin to help implement language-specific lexer features."""
    
    def add_language_patterns(self, base_patterns: List[Tuple[TokenType, str]], 
                            language_patterns: List[Tuple[TokenType, str]]) -> List[Tuple[TokenType, str]]:
        """Add language-specific patterns to base patterns."""
        # Insert language patterns after comments but before other patterns
        result = []
        
        # Add comments first
        for token_type, pattern in base_patterns:
            if token_type in [TokenType.COMMENT_SINGLE, TokenType.COMMENT_MULTI]:
                result.append((token_type, pattern))
        
        # Add language-specific patterns
        result.extend(language_patterns)
        
        # Add remaining base patterns
        for token_type, pattern in base_patterns:
            if token_type not in [TokenType.COMMENT_SINGLE, TokenType.COMMENT_MULTI]:
                result.append((token_type, pattern))
        
        return result
    
    def merge_keywords(self, base_keywords: Dict[str, TokenType], 
                      language_keywords: Dict[str, TokenType]) -> Dict[str, TokenType]:
        """Merge base keywords with language-specific keywords."""
        merged = base_keywords.copy()
        merged.update(language_keywords)
        return merged


# Utility functions for backward compatibility

def convert_token_types_to_strings(tokens: List[Token]) -> List[Tuple[str, str]]:
    """Convert new Token objects to old string tuple format."""
    return [(token.type.name, token.value) for token in tokens]

def convert_string_tokens_to_objects(string_tokens: List[Tuple[str, str]]) -> List[Token]:
    """Convert old string tuple format to new Token objects."""
    tokens = []
    for token_type_str, value in string_tokens:
        try:
            token_type = TokenType[token_type_str]
        except KeyError:
            # Handle unknown token types
            token_type = TokenType.LANGUAGE_SPECIFIC
        tokens.append(Token(token_type, value))
    return tokens
