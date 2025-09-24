"""
Base Parser Infrastructure for all language backends.
This module provides common parser functionality to reduce code duplication.
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any, Union, Tuple
from .base_lexer import Token, TokenType
from .base_ast import *


class ParseError(Exception):
    """Custom exception for parsing errors."""
    
    def __init__(self, message: str, token: Optional[Token] = None):
        self.message = message
        self.token = token
        if token:
            super().__init__(f"Parse error at line {token.line}, column {token.column}: {message}")
        else:
            super().__init__(f"Parse error: {message}")


class BaseParser(ABC):
    """Base parser class providing common functionality for all language backends."""
    
    def __init__(self, tokens: List[Token]):
        self.tokens = tokens
        self.current_index = 0
        self.current_token = tokens[0] if tokens else None
        
        # Error recovery
        self.error_recovery_points = [TokenType.SEMICOLON, TokenType.RBRACE, TokenType.EOF]
        self.synchronization_tokens = [TokenType.STRUCT, TokenType.FUNCTION, TokenType.IF, TokenType.FOR]
        
        # Context tracking
        self.context_stack = []
        self.symbol_table = {}
    
    def error(self, message: str) -> None:
        """Raise a parsing error with context."""
        raise ParseError(message, self.current_token)
    
    def advance(self) -> Token:
        """Move to the next token."""
        if self.current_index < len(self.tokens) - 1:
            self.current_index += 1
            self.current_token = self.tokens[self.current_index]
        else:
            self.current_token = Token(TokenType.EOF, "", self.current_token.line if self.current_token else 1)
        return self.current_token
    
    def peek(self, offset: int = 1) -> Optional[Token]:
        """Look ahead at future tokens."""
        peek_index = self.current_index + offset
        if peek_index < len(self.tokens):
            return self.tokens[peek_index]
        return None
    
    def eat(self, expected_type: TokenType) -> Token:
        """Consume a token of the expected type."""
        if not self.current_token:
            self.error(f"Expected {expected_type.name} but reached end of input")
        
        if self.current_token.type != expected_type:
            self.error(f"Expected {expected_type.name}, got {self.current_token.type.name}")
        
        token = self.current_token
        self.advance()
        return token
    
    def match(self, *token_types: TokenType) -> bool:
        """Check if current token matches any of the given types."""
        if not self.current_token:
            return False
        return self.current_token.type in token_types
    
    def consume_if_match(self, token_type: TokenType) -> bool:
        """Consume token if it matches the given type."""
        if self.match(token_type):
            self.advance()
            return True
        return False
    
    def skip_until(self, *token_types: TokenType) -> bool:
        """Skip tokens until one of the specified types is found."""
        while self.current_token and not self.match(*token_types):
            if self.current_token.type == TokenType.EOF:
                return False
            self.advance()
        return self.current_token is not None
    
    def synchronize(self) -> None:
        """Synchronize parser state for error recovery."""
        self.skip_until(*self.error_recovery_points)
        if self.match(TokenType.SEMICOLON):
            self.advance()
    
    def push_context(self, context: str):
        """Push a parsing context for better error reporting."""
        self.context_stack.append(context)
    
    def pop_context(self) -> Optional[str]:
        """Pop the current parsing context."""
        return self.context_stack.pop() if self.context_stack else None
    
    def get_context(self) -> str:
        """Get current parsing context."""
        return " -> ".join(self.context_stack) if self.context_stack else "global"
    
    # Common parsing methods that can be shared
    
    def parse_identifier(self) -> str:
        """Parse an identifier."""
        token = self.eat(TokenType.IDENTIFIER)
        return token.value
    
    def parse_number(self) -> Union[int, float]:
        """Parse a numeric literal."""
        token = self.eat(TokenType.NUMBER)
        value = token.value
        
        # Try to parse as appropriate numeric type
        if '.' in value or 'e' in value.lower():
            return float(value.rstrip('fFdD'))
        else:
            return int(value.rstrip('lLuU'))
    
    def parse_string_literal(self) -> str:
        """Parse a string literal."""
        token = self.eat(TokenType.STRING)
        # Remove quotes and handle escape sequences
        return token.value[1:-1]  # Remove surrounding quotes
    
    def parse_block_or_statement(self) -> Union[List[StatementNode], StatementNode]:
        """Parse either a block or a single statement."""
        if self.match(TokenType.LBRACE):
            return self.parse_block()
        else:
            return self.parse_statement()
    
    def parse_block(self) -> List[StatementNode]:
        """Parse a block of statements."""
        self.push_context("block")
        self.eat(TokenType.LBRACE)
        
        statements = []
        while not self.match(TokenType.RBRACE, TokenType.EOF):
            try:
                stmt = self.parse_statement()
                if stmt:
                    statements.append(stmt)
            except ParseError as e:
                # Error recovery
                print(f"Warning: {e}")
                self.synchronize()
        
        self.eat(TokenType.RBRACE)
        self.pop_context()
        return statements
    
    def parse_parameter_list(self) -> List[BaseParameterNode]:
        """Parse function parameter list."""
        self.eat(TokenType.LPAREN)
        params = []
        
        while not self.match(TokenType.RPAREN):
            param = self.parse_parameter()
            params.append(param)
            
            if self.match(TokenType.COMMA):
                self.advance()
            elif not self.match(TokenType.RPAREN):
                self.error("Expected ',' or ')' in parameter list")
        
        self.eat(TokenType.RPAREN)
        return params
    
    def parse_argument_list(self) -> List[ExpressionNode]:
        """Parse function call argument list."""
        self.eat(TokenType.LPAREN)
        args = []
        
        while not self.match(TokenType.RPAREN):
            arg = self.parse_expression()
            args.append(arg)
            
            if self.match(TokenType.COMMA):
                self.advance()
            elif not self.match(TokenType.RPAREN):
                self.error("Expected ',' or ')' in argument list")
        
        self.eat(TokenType.RPAREN)
        return args
    
    def parse_binary_expression(self, min_precedence: int = 0) -> ExpressionNode:
        """Parse binary expressions with precedence climbing."""
        left = self.parse_unary_expression()
        
        while True:
            if not self.current_token or not self._is_binary_operator(self.current_token.type):
                break
            
            precedence = self._get_precedence(self.current_token.type)
            if precedence < min_precedence:
                break
            
            op_token = self.current_token
            self.advance()
            
            # Handle right-associative operators
            next_min_prec = precedence + (1 if self._is_left_associative(op_token.type) else 0)
            right = self.parse_binary_expression(next_min_prec)
            
            left = BaseBinaryOpNode(left, op_token.value, right)
        
        return left
    
    def _is_binary_operator(self, token_type: TokenType) -> bool:
        """Check if token type is a binary operator."""
        binary_ops = {
            TokenType.PLUS, TokenType.MINUS, TokenType.MULTIPLY, TokenType.DIVIDE, TokenType.MODULO,
            TokenType.EQUAL, TokenType.NOT_EQUAL, TokenType.LESS_THAN, TokenType.GREATER_THAN,
            TokenType.LESS_EQUAL, TokenType.GREATER_EQUAL, TokenType.LOGICAL_AND, TokenType.LOGICAL_OR,
            TokenType.BITWISE_AND, TokenType.BITWISE_OR, TokenType.BITWISE_XOR,
            TokenType.SHIFT_LEFT, TokenType.SHIFT_RIGHT
        }
        return token_type in binary_ops
    
    def _get_precedence(self, token_type: TokenType) -> int:
        """Get operator precedence (higher number = higher precedence)."""
        precedence_map = {
            TokenType.LOGICAL_OR: 1,
            TokenType.LOGICAL_AND: 2,
            TokenType.BITWISE_OR: 3,
            TokenType.BITWISE_XOR: 4,
            TokenType.BITWISE_AND: 5,
            TokenType.EQUAL: 6, TokenType.NOT_EQUAL: 6,
            TokenType.LESS_THAN: 7, TokenType.GREATER_THAN: 7,
            TokenType.LESS_EQUAL: 7, TokenType.GREATER_EQUAL: 7,
            TokenType.SHIFT_LEFT: 8, TokenType.SHIFT_RIGHT: 8,
            TokenType.PLUS: 9, TokenType.MINUS: 9,
            TokenType.MULTIPLY: 10, TokenType.DIVIDE: 10, TokenType.MODULO: 10,
        }
        return precedence_map.get(token_type, 0)
    
    def _is_left_associative(self, token_type: TokenType) -> bool:
        """Check if operator is left-associative."""
        # Most operators are left-associative
        right_associative = {TokenType.ASSIGN, TokenType.PLUS_EQUALS, TokenType.MINUS_EQUALS}
        return token_type not in right_associative
    
    # Abstract methods that must be implemented by subclasses
    
    @abstractmethod
    def parse(self) -> BaseShaderNode:
        """Parse the entire program. Must be implemented by subclasses."""
        pass
    
    @abstractmethod
    def parse_statement(self) -> Optional[StatementNode]:
        """Parse a statement. Must be implemented by subclasses."""
        pass
    
    @abstractmethod
    def parse_expression(self) -> ExpressionNode:
        """Parse an expression. Must be implemented by subclasses."""
        pass
    
    @abstractmethod
    def parse_type(self) -> str:
        """Parse a type specification. Must be implemented by subclasses."""
        pass
    
    @abstractmethod
    def parse_parameter(self) -> BaseParameterNode:
        """Parse a function parameter. Must be implemented by subclasses."""
        pass
    
    @abstractmethod
    def parse_unary_expression(self) -> ExpressionNode:
        """Parse a unary expression. Must be implemented by subclasses."""
        pass
    
    # Common implementations that can be used by subclasses
    
    def parse_if_statement(self) -> BaseIfNode:
        """Parse an if statement (common implementation)."""
        self.eat(TokenType.IF)
        self.eat(TokenType.LPAREN)
        condition = self.parse_expression()
        self.eat(TokenType.RPAREN)
        
        if_body = self.parse_block_or_statement()
        
        else_body = None
        if self.match(TokenType.ELSE):
            self.advance()
            else_body = self.parse_block_or_statement()
        
        return BaseIfNode(condition, if_body, else_body)
    
    def parse_for_statement(self) -> BaseForNode:
        """Parse a for statement (common implementation)."""
        self.eat(TokenType.FOR)
        self.eat(TokenType.LPAREN)
        
        # Parse init
        init = None
        if not self.match(TokenType.SEMICOLON):
            init = self.parse_statement() if self.is_declaration() else self.parse_expression()
        self.eat(TokenType.SEMICOLON)
        
        # Parse condition
        condition = None
        if not self.match(TokenType.SEMICOLON):
            condition = self.parse_expression()
        self.eat(TokenType.SEMICOLON)
        
        # Parse update
        update = None
        if not self.match(TokenType.RPAREN):
            update = self.parse_expression()
        self.eat(TokenType.RPAREN)
        
        body = self.parse_block_or_statement()
        
        return BaseForNode(init, condition, update, body)
    
    def parse_while_statement(self) -> BaseWhileNode:
        """Parse a while statement (common implementation)."""
        self.eat(TokenType.WHILE)
        self.eat(TokenType.LPAREN)
        condition = self.parse_expression()
        self.eat(TokenType.RPAREN)
        
        body = self.parse_block_or_statement()
        
        return BaseWhileNode(condition, body)
    
    def parse_return_statement(self) -> BaseReturnNode:
        """Parse a return statement (common implementation)."""
        self.eat(TokenType.RETURN)
        
        value = None
        if not self.match(TokenType.SEMICOLON):
            value = self.parse_expression()
        
        self.eat(TokenType.SEMICOLON)
        return BaseReturnNode(value)
    
    def is_declaration(self) -> bool:
        """Check if current position is a declaration."""
        # Look ahead to determine if this is a declaration
        return self.match(TokenType.INT, TokenType.FLOAT_TYPE, TokenType.BOOL, 
                         TokenType.VOID, TokenType.STRUCT, TokenType.IDENTIFIER)
    
    def is_assignment_operator(self, token_type: TokenType) -> bool:
        """Check if token type is an assignment operator."""
        assignment_ops = {
            TokenType.ASSIGN, TokenType.PLUS_EQUALS, TokenType.MINUS_EQUALS,
            TokenType.MULTIPLY_EQUALS, TokenType.DIVIDE_EQUALS, TokenType.MODULO_EQUALS
        }
        return token_type in assignment_ops
    
    def parse_attributes(self) -> List[BaseAttributeNode]:
        """Parse attributes/annotations (language-specific override needed)."""
        return []  # Default implementation
    
    # Utility methods for common parsing patterns
    
    def parse_comma_separated_list(self, parse_func, terminator: TokenType) -> List[Any]:
        """Parse a comma-separated list of items."""
        items = []
        
        while not self.match(terminator, TokenType.EOF):
            item = parse_func()
            items.append(item)
            
            if self.match(TokenType.COMMA):
                self.advance()
            elif not self.match(terminator):
                self.error(f"Expected ',' or '{terminator.name}' in list")
        
        return items
    
    def parse_optional_semicolon(self) -> bool:
        """Parse optional semicolon and return True if found."""
        if self.match(TokenType.SEMICOLON):
            self.advance()
            return True
        return False
    
    def parse_qualifiers(self) -> List[str]:
        """Parse common qualifiers."""
        qualifiers = []
        qualifier_tokens = {TokenType.CONST, TokenType.STATIC, TokenType.INLINE, TokenType.EXTERN}
        
        while self.match(*qualifier_tokens):
            qualifiers.append(self.current_token.value)
            self.advance()
        
        return qualifiers


class StandardParser(BaseParser):
    """Standard parser implementation for C-like languages."""
    
    def parse(self) -> BaseShaderNode:
        """Parse a standard C-like program."""
        includes = []
        structs = []
        functions = []
        global_variables = []
        
        while not self.match(TokenType.EOF):
            try:
                if self.match(TokenType.PREPROCESSOR):
                    includes.append(self.parse_preprocessor())
                elif self.match(TokenType.STRUCT):
                    structs.append(self.parse_struct())
                elif self.is_function_declaration():
                    functions.append(self.parse_function())
                elif self.is_variable_declaration():
                    global_variables.append(self.parse_variable_declaration())
                else:
                    self.advance()  # Skip unknown tokens
            except ParseError as e:
                print(f"Warning: {e}")
                self.synchronize()
        
        return BaseShaderNode(includes, structs, functions, global_variables)
    
    def parse_statement(self) -> Optional[StatementNode]:
        """Parse a statement."""
        if self.match(TokenType.IF):
            return self.parse_if_statement()
        elif self.match(TokenType.FOR):
            return self.parse_for_statement()
        elif self.match(TokenType.WHILE):
            return self.parse_while_statement()
        elif self.match(TokenType.RETURN):
            return self.parse_return_statement()
        elif self.match(TokenType.BREAK):
            self.advance()
            self.eat(TokenType.SEMICOLON)
            return BaseBreakNode()
        elif self.match(TokenType.CONTINUE):
            self.advance()
            self.eat(TokenType.SEMICOLON)
            return BaseContinueNode()
        elif self.match(TokenType.LBRACE):
            return self.parse_block()
        elif self.is_variable_declaration():
            var = self.parse_variable_declaration()
            self.eat(TokenType.SEMICOLON)
            return var
        else:
            # Expression statement
            expr = self.parse_expression()
            self.eat(TokenType.SEMICOLON)
            return expr
    
    def parse_expression(self) -> ExpressionNode:
        """Parse an expression."""
        return self.parse_assignment_expression()
    
    def parse_assignment_expression(self) -> ExpressionNode:
        """Parse assignment expressions."""
        left = self.parse_ternary_expression()
        
        if self.is_assignment_operator(self.current_token.type if self.current_token else None):
            op_token = self.current_token
            self.advance()
            right = self.parse_assignment_expression()
            return BaseAssignmentNode(left, right, op_token.value)
        
        return left
    
    def parse_ternary_expression(self) -> ExpressionNode:
        """Parse ternary conditional expressions."""
        expr = self.parse_binary_expression()
        
        if self.match(TokenType.QUESTION):
            self.advance()
            true_expr = self.parse_expression()
            self.eat(TokenType.COLON)
            false_expr = self.parse_expression()
            return BaseTernaryOpNode(expr, true_expr, false_expr)
        
        return expr
    
    def parse_unary_expression(self) -> ExpressionNode:
        """Parse unary expressions."""
        unary_ops = {TokenType.PLUS, TokenType.MINUS, TokenType.LOGICAL_NOT, 
                    TokenType.BITWISE_NOT, TokenType.MULTIPLY, TokenType.BITWISE_AND}
        
        if self.match(*unary_ops):
            op_token = self.current_token
            self.advance()
            operand = self.parse_unary_expression()
            return BaseUnaryOpNode(op_token.value, operand)
        
        return self.parse_postfix_expression()
    
    def parse_postfix_expression(self) -> ExpressionNode:
        """Parse postfix expressions."""
        expr = self.parse_primary_expression()
        
        while True:
            if self.match(TokenType.DOT):
                self.advance()
                member = self.parse_identifier()
                expr = BaseMemberAccessNode(expr, member)
            elif self.match(TokenType.ARROW):
                self.advance()
                member = self.parse_identifier()
                expr = BaseMemberAccessNode(expr, member, is_pointer=True)
            elif self.match(TokenType.LBRACKET):
                self.advance()
                index = self.parse_expression()
                self.eat(TokenType.RBRACKET)
                expr = BaseArrayAccessNode(expr, index)
            elif self.match(TokenType.LPAREN):
                args = self.parse_argument_list()
                expr = BaseFunctionCallNode(expr, args)
            else:
                break
        
        return expr
    
    def parse_primary_expression(self) -> ExpressionNode:
        """Parse primary expressions."""
        if self.match(TokenType.IDENTIFIER):
            name = self.parse_identifier()
            return name  # Simple string for now, can be wrapped in IdentifierNode
        elif self.match(TokenType.NUMBER):
            value = self.parse_number()
            return str(value)  # Simple string for now, can be wrapped in LiteralNode
        elif self.match(TokenType.STRING):
            value = self.parse_string_literal()
            return value
        elif self.match(TokenType.LPAREN):
            self.advance()
            expr = self.parse_expression()
            self.eat(TokenType.RPAREN)
            return expr
        else:
            self.error(f"Unexpected token in expression: {self.current_token.type.name if self.current_token else 'EOF'}")
    
    def parse_type(self) -> str:
        """Parse a type specification."""
        if self.match(TokenType.VOID, TokenType.BOOL, TokenType.INT, TokenType.UINT, 
                     TokenType.FLOAT_TYPE, TokenType.DOUBLE, TokenType.CHAR):
            type_name = self.current_token.value
            self.advance()
            return type_name
        elif self.match(TokenType.IDENTIFIER):
            return self.parse_identifier()
        else:
            self.error(f"Expected type, got {self.current_token.type.name if self.current_token else 'EOF'}")
    
    def parse_parameter(self) -> BaseParameterNode:
        """Parse a function parameter."""
        param_type = self.parse_type()
        param_name = self.parse_identifier()
        return BaseParameterNode(param_type, param_name)
    
    def parse_struct(self) -> BaseStructNode:
        """Parse a struct declaration."""
        self.eat(TokenType.STRUCT)
        name = self.parse_identifier()
        self.eat(TokenType.LBRACE)
        
        members = []
        while not self.match(TokenType.RBRACE, TokenType.EOF):
            member = self.parse_struct_member()
            members.append(member)
        
        self.eat(TokenType.RBRACE)
        self.parse_optional_semicolon()
        
        return BaseStructNode(name, members)
    
    def parse_struct_member(self) -> BaseVariableNode:
        """Parse a struct member."""
        member_type = self.parse_type()
        member_name = self.parse_identifier()
        self.eat(TokenType.SEMICOLON)
        return BaseVariableNode(member_type, member_name)
    
    def parse_function(self) -> BaseFunctionNode:
        """Parse a function declaration."""
        qualifiers = self.parse_qualifiers()
        return_type = self.parse_type()
        name = self.parse_identifier()
        params = self.parse_parameter_list()
        
        body = None
        if self.match(TokenType.LBRACE):
            body = self.parse_block()
        else:
            self.eat(TokenType.SEMICOLON)
        
        return BaseFunctionNode(return_type, name, params, body, qualifiers)
    
    def parse_variable_declaration(self) -> BaseVariableNode:
        """Parse a variable declaration."""
        qualifiers = self.parse_qualifiers()
        var_type = self.parse_type()
        name = self.parse_identifier()
        
        value = None
        if self.match(TokenType.ASSIGN):
            self.advance()
            value = self.parse_expression()
        
        return BaseVariableNode(var_type, name, value, qualifiers)
    
    def parse_preprocessor(self) -> BasePreprocessorNode:
        """Parse a preprocessor directive."""
        token = self.eat(TokenType.PREPROCESSOR)
        content = token.value
        
        if content.startswith("#include"):
            return BasePreprocessorNode("include", content[8:].strip())
        elif content.startswith("#define"):
            return BasePreprocessorNode("define", content[7:].strip())
        else:
            return BasePreprocessorNode("other", content)
    
    def is_function_declaration(self) -> bool:
        """Check if current position is a function declaration."""
        # Simple heuristic: look for type identifier (
        saved_index = self.current_index
        
        try:
            # Skip qualifiers
            while self.match(TokenType.CONST, TokenType.STATIC, TokenType.INLINE, TokenType.EXTERN):
                self.advance()
            
            # Check for type
            if not self.match(TokenType.VOID, TokenType.BOOL, TokenType.INT, TokenType.UINT,
                             TokenType.FLOAT_TYPE, TokenType.DOUBLE, TokenType.IDENTIFIER):
                return False
            self.advance()
            
            # Check for identifier
            if not self.match(TokenType.IDENTIFIER):
                return False
            self.advance()
            
            # Check for opening parenthesis
            return self.match(TokenType.LPAREN)
        
        finally:
            # Restore position
            self.current_index = saved_index
            self.current_token = self.tokens[self.current_index] if self.current_index < len(self.tokens) else None
    
    def is_variable_declaration(self) -> bool:
        """Check if current position is a variable declaration."""
        # Simple heuristic: type followed by identifier, not followed by (
        saved_index = self.current_index
        
        try:
            # Skip qualifiers
            while self.match(TokenType.CONST, TokenType.STATIC, TokenType.EXTERN):
                self.advance()
            
            # Check for type
            if not self.match(TokenType.VOID, TokenType.BOOL, TokenType.INT, TokenType.UINT,
                             TokenType.FLOAT_TYPE, TokenType.DOUBLE, TokenType.IDENTIFIER):
                return False
            self.advance()
            
            # Check for identifier
            if not self.match(TokenType.IDENTIFIER):
                return False
            self.advance()
            
            # Should not be followed by opening parenthesis (that would be a function)
            return not self.match(TokenType.LPAREN)
        
        finally:
            # Restore position
            self.current_index = saved_index
            self.current_token = self.tokens[self.current_index] if self.current_index < len(self.tokens) else None


class ParserFactory:
    """Factory for creating appropriate parsers based on language."""
    
    _parser_registry: Dict[str, type] = {}
    
    @classmethod
    def register_parser(cls, language: str, parser_class: type):
        """Register a parser for a specific language."""
        cls._parser_registry[language.lower()] = parser_class
    
    @classmethod
    def create_parser(cls, language: str, tokens: List[Token]) -> BaseParser:
        """Create a parser for the specified language."""
        language = language.lower()
        if language in cls._parser_registry:
            return cls._parser_registry[language](tokens)
        else:
            # Fallback to standard parser
            return StandardParser(tokens)
    
    @classmethod
    def get_supported_languages(cls) -> List[str]:
        """Get list of supported languages."""
        return list(cls._parser_registry.keys())
