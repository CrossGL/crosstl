"""
HIP Language Parser

This module provides parsing functionality for HIP (HIP Is a Portable GPU Runtime) code.
HIP is AMD's CUDA-compatible runtime API for GPU programming.
"""

from typing import List, Optional, Union, Any
from ..HIP.HipLexer import HipLexer
from ..HIP.HipAst import *


class HipParser:
    """Parser for HIP language constructs"""
    
    def __init__(self, tokens: List[Any]):
        self.tokens = tokens
        self.pos = 0
        self.current_token = self.tokens[0] if tokens else None
        
    def error(self, message: str):
        """Raise a parsing error with context"""
        token_info = f"at token '{self.current_token.value}'" if self.current_token else "at end of input"
        raise SyntaxError(f"Parse error {token_info}: {message}")
    
    def advance(self):
        """Move to the next token"""
        if self.pos < len(self.tokens) - 1:
            self.pos += 1
            self.current_token = self.tokens[self.pos]
        else:
            self.current_token = None
    
    def peek(self, offset: int = 1) -> Optional[Any]:
        """Look ahead at the next token without advancing"""
        peek_pos = self.pos + offset
        if peek_pos < len(self.tokens):
            return self.tokens[peek_pos]
        return None
    
    def consume(self, expected_type: str) -> Any:
        """Consume a token of the expected type"""
        if not self.current_token:
            self.error(f"Expected {expected_type} but reached end of input")
        
        if self.current_token.type != expected_type:
            self.error(f"Expected {expected_type}, got {self.current_token.type}")
        
        token = self.current_token
        self.advance()
        return token
    
    def match(self, *token_types: str) -> bool:
        """Check if current token matches any of the given types"""
        if not self.current_token:
            return False
        return self.current_token.type in token_types
    
    def parse(self) -> HipProgramNode:
        """Parse the entire HIP program"""
        statements = []
        
        while self.current_token:
            if self.match('NEWLINE'):
                self.advance()
                continue
                
            stmt = self.parse_statement()
            if stmt:
                statements.append(stmt)
        
        return HipProgramNode(statements)
    
    def parse_statement(self) -> Optional[HipASTNode]:
        """Parse a single statement"""
        if not self.current_token:
            return None
            
        # Skip newlines and semicolons
        if self.match('NEWLINE', 'SEMICOLON'):
            self.advance()
            return None
        
        # Parse preprocessor directives
        if self.match('HASH'):
            return self.parse_preprocessor()
        
        # Parse extern "C" blocks
        if self.match('EXTERN'):
            return self.parse_extern_block()
        
        # Parse device/host qualifiers
        if self.match('__DEVICE__', '__HOST__', '__GLOBAL__'):
            return self.parse_function_with_qualifier()
        
        # Parse struct definitions
        if self.match('STRUCT'):
            return self.parse_struct()
        
        # Parse class definitions
        if self.match('CLASS'):
            return self.parse_class()
        
        # Parse enum definitions
        if self.match('ENUM'):
            return self.parse_enum()
        
        # Parse typedef
        if self.match('TYPEDEF'):
            return self.parse_typedef()
        
        # Parse namespace
        if self.match('NAMESPACE'):
            return self.parse_namespace()
        
        # Parse using declarations
        if self.match('USING'):
            return self.parse_using()
        
        # Parse template declarations
        if self.match('TEMPLATE'):
            return self.parse_template()
        
        # Parse function declarations/definitions
        if self.is_function_declaration():
            return self.parse_function()
        
        # Parse variable declarations
        if self.is_variable_declaration():
            return self.parse_variable_declaration()
        
        # Parse expressions and other statements
        return self.parse_expression_statement()
    
    def parse_preprocessor(self) -> HipPreprocessorNode:
        """Parse preprocessor directives"""
        self.consume('HASH')
        
        if not self.current_token:
            self.error("Expected preprocessor directive after #")
        
        directive = self.current_token.value
        self.advance()
        
        # Parse the rest of the line
        content = []
        while self.current_token and not self.match('NEWLINE'):
            content.append(self.current_token.value)
            self.advance()
        
        return HipPreprocessorNode(directive, ' '.join(content))
    
    def parse_extern_block(self) -> HipExternBlockNode:
        """Parse extern "C" blocks"""
        self.consume('EXTERN')
        
        linkage = None
        if self.match('STRING'):
            linkage = self.current_token.value.strip('"')
            self.advance()
        
        body = []
        if self.match('LBRACE'):
            self.consume('LBRACE')
            while self.current_token and not self.match('RBRACE'):
                stmt = self.parse_statement()
                if stmt:
                    body.append(stmt)
            self.consume('RBRACE')
        else:
            # Single statement
            stmt = self.parse_statement()
            if stmt:
                body.append(stmt)
        
        return HipExternBlockNode(linkage, body)
    
    def parse_function_with_qualifier(self) -> HipFunctionNode:
        """Parse function with device/host qualifiers"""
        qualifiers = []
        
        # Parse all qualifiers
        while self.match('__DEVICE__', '__HOST__', '__GLOBAL__', '__FORCEINLINE__', '__NOINLINE__'):
            qualifiers.append(self.current_token.value)
            self.advance()
        
        # Parse the rest of the function
        return_type = self.parse_type()
        name = self.consume('IDENTIFIER').value
        
        # Parse parameters
        self.consume('LPAREN')
        params = self.parse_parameter_list()
        self.consume('RPAREN')
        
        # Parse function body or semicolon for declaration
        body = None
        if self.match('LBRACE'):
            body = self.parse_block()
        elif self.match('SEMICOLON'):
            self.advance()
        
        return HipFunctionNode(name, return_type, params, body, qualifiers)
    
    def parse_struct(self) -> HipStructNode:
        """Parse struct definitions"""
        self.consume('STRUCT')
        
        name = None
        if self.match('IDENTIFIER'):
            name = self.current_token.value
            self.advance()
        
        # Parse inheritance (if any)
        inheritance = []
        if self.match('COLON'):
            self.advance()
            inheritance = self.parse_inheritance_list()
        
        # Parse struct body
        fields = []
        if self.match('LBRACE'):
            self.consume('LBRACE')
            while self.current_token and not self.match('RBRACE'):
                if self.match('NEWLINE', 'SEMICOLON'):
                    self.advance()
                    continue
                
                # Parse access specifiers
                access = None
                if self.match('PUBLIC', 'PRIVATE', 'PROTECTED'):
                    access = self.current_token.value
                    self.advance()
                    self.consume('COLON')
                    continue
                
                # Parse member
                member = self.parse_struct_member()
                if member:
                    fields.append(member)
            
            self.consume('RBRACE')
        
        # Optional semicolon
        if self.match('SEMICOLON'):
            self.advance()
        
        return HipStructNode(name, fields, inheritance)
    
    def parse_class(self) -> HipClassNode:
        """Parse class definitions"""
        self.consume('CLASS')
        
        name = self.consume('IDENTIFIER').value
        
        # Parse inheritance
        inheritance = []
        if self.match('COLON'):
            self.advance()
            inheritance = self.parse_inheritance_list()
        
        # Parse class body
        members = []
        if self.match('LBRACE'):
            self.consume('LBRACE')
            
            current_access = 'private'  # Default access for class
            
            while self.current_token and not self.match('RBRACE'):
                if self.match('NEWLINE', 'SEMICOLON'):
                    self.advance()
                    continue
                
                # Parse access specifiers
                if self.match('PUBLIC', 'PRIVATE', 'PROTECTED'):
                    current_access = self.current_token.value
                    self.advance()
                    self.consume('COLON')
                    continue
                
                # Parse member
                member = self.parse_class_member()
                if member:
                    member.access = current_access
                    members.append(member)
            
            self.consume('RBRACE')
        
        # Optional semicolon
        if self.match('SEMICOLON'):
            self.advance()
        
        return HipClassNode(name, members, inheritance)
    
    def parse_enum(self) -> HipEnumNode:
        """Parse enum definitions"""
        self.consume('ENUM')
        
        # Optional enum class
        is_class = False
        if self.match('CLASS'):
            is_class = True
            self.advance()
        
        name = None
        if self.match('IDENTIFIER'):
            name = self.current_token.value
            self.advance()
        
        # Optional underlying type
        underlying_type = None
        if self.match('COLON'):
            self.advance()
            underlying_type = self.parse_type()
        
        # Parse enum body
        values = []
        if self.match('LBRACE'):
            self.consume('LBRACE')
            
            while self.current_token and not self.match('RBRACE'):
                if self.match('NEWLINE', 'COMMA'):
                    self.advance()
                    continue
                
                # Parse enum value
                value_name = self.consume('IDENTIFIER').value
                value_expr = None
                
                if self.match('ASSIGN'):
                    self.advance()
                    value_expr = self.parse_expression()
                
                values.append(HipEnumValueNode(value_name, value_expr))
                
                if self.match('COMMA'):
                    self.advance()
            
            self.consume('RBRACE')
        
        return HipEnumNode(name, values, underlying_type, is_class)
    
    def parse_typedef(self) -> HipTypedefNode:
        """Parse typedef statements"""
        self.consume('TYPEDEF')
        
        base_type = self.parse_type()
        alias = self.consume('IDENTIFIER').value
        
        self.consume('SEMICOLON')
        
        return HipTypedefNode(alias, base_type)
    
    def parse_namespace(self) -> HipNamespaceNode:
        """Parse namespace definitions"""
        self.consume('NAMESPACE')
        
        name = None
        if self.match('IDENTIFIER'):
            name = self.current_token.value
            self.advance()
        
        body = []
        if self.match('LBRACE'):
            self.consume('LBRACE')
            while self.current_token and not self.match('RBRACE'):
                stmt = self.parse_statement()
                if stmt:
                    body.append(stmt)
            self.consume('RBRACE')
        
        return HipNamespaceNode(name, body)
    
    def parse_using(self) -> HipUsingNode:
        """Parse using declarations"""
        self.consume('USING')
        
        # Parse using directive or declaration
        if self.match('NAMESPACE'):
            self.advance()
            namespace = self.parse_qualified_name()
            self.consume('SEMICOLON')
            return HipUsingNode(namespace, None, True)
        else:
            alias = None
            target = None
            
            # Check for alias
            if self.peek() and self.peek().type == 'ASSIGN':
                alias = self.consume('IDENTIFIER').value
                self.consume('ASSIGN')
            
            target = self.parse_type()
            self.consume('SEMICOLON')
            
            return HipUsingNode(target, alias, False)
    
    def parse_template(self) -> HipTemplateNode:
        """Parse template declarations"""
        self.consume('TEMPLATE')
        self.consume('LT')
        
        # Parse template parameters
        params = []
        while self.current_token and not self.match('GT'):
            if self.match('COMMA'):
                self.advance()
                continue
            
            param = self.parse_template_parameter()
            params.append(param)
        
        self.consume('GT')
        
        # Parse template body
        body = self.parse_statement()
        
        return HipTemplateNode(params, body)
    
    def parse_template_parameter(self) -> HipTemplateParameterNode:
        """Parse template parameter"""
        param_type = None
        name = None
        default_value = None
        
        if self.match('TYPENAME', 'CLASS'):
            param_type = self.current_token.value
            self.advance()
            
            if self.match('IDENTIFIER'):
                name = self.current_token.value
                self.advance()
        else:
            # Non-type template parameter
            param_type = self.parse_type()
            name = self.consume('IDENTIFIER').value
        
        # Parse default value
        if self.match('ASSIGN'):
            self.advance()
            default_value = self.parse_expression()
        
        return HipTemplateParameterNode(param_type, name, default_value)
    
    def parse_function(self) -> HipFunctionNode:
        """Parse function declarations and definitions"""
        # Parse template (if any)
        template = None
        if self.match('TEMPLATE'):
            template = self.parse_template()
        
        # Parse qualifiers
        qualifiers = []
        while self.match('INLINE', 'STATIC', 'EXTERN', 'VIRTUAL', 'CONST', 'CONSTEXPR'):
            qualifiers.append(self.current_token.value)
            self.advance()
        
        # Parse return type
        return_type = self.parse_type()
        
        # Parse function name
        name = self.consume('IDENTIFIER').value
        
        # Parse template specialization (if any)
        if self.match('LT'):
            self.advance()
            # Skip template arguments for now
            bracket_count = 1
            while bracket_count > 0 and self.current_token:
                if self.match('LT'):
                    bracket_count += 1
                elif self.match('GT'):
                    bracket_count -= 1
                self.advance()
        
        # Parse parameters
        self.consume('LPAREN')
        params = self.parse_parameter_list()
        self.consume('RPAREN')
        
        # Parse const qualifier (for member functions)
        if self.match('CONST'):
            qualifiers.append('const')
            self.advance()
        
        # Parse function body or semicolon
        body = None
        if self.match('LBRACE'):
            body = self.parse_block()
        elif self.match('SEMICOLON'):
            self.advance()
        
        return HipFunctionNode(name, return_type, params, body, qualifiers, template)
    
    def parse_parameter_list(self) -> List[HipParameterNode]:
        """Parse function parameter list"""
        params = []
        
        while self.current_token and not self.match('RPAREN'):
            if self.match('COMMA'):
                self.advance()
                continue
            
            param = self.parse_parameter()
            params.append(param)
        
        return params
    
    def parse_parameter(self) -> HipParameterNode:
        """Parse a single parameter"""
        param_type = self.parse_type()
        
        name = None
        if self.match('IDENTIFIER'):
            name = self.current_token.value
            self.advance()
        
        # Default value
        default_value = None
        if self.match('ASSIGN'):
            self.advance()
            default_value = self.parse_expression()
        
        return HipParameterNode(name, param_type, default_value)
    
    def parse_struct_member(self) -> Optional[HipStructMemberNode]:
        """Parse struct member"""
        # Parse member type
        member_type = self.parse_type()
        
        # Parse member name
        if not self.match('IDENTIFIER'):
            self.error("Expected member name")
        
        name = self.current_token.value
        self.advance()
        
        # Parse array dimensions
        dimensions = []
        while self.match('LBRACKET'):
            self.advance()
            if not self.match('RBRACKET'):
                dimensions.append(self.parse_expression())
            else:
                dimensions.append(None)
            self.consume('RBRACKET')
        
        # Parse bitfield
        bitfield = None
        if self.match('COLON'):
            self.advance()
            bitfield = self.parse_expression()
        
        # Parse initializer
        initializer = None
        if self.match('ASSIGN'):
            self.advance()
            initializer = self.parse_expression()
        
        self.consume('SEMICOLON')
        
        return HipStructMemberNode(name, member_type, dimensions, bitfield, initializer)
    
    def parse_class_member(self) -> Optional[HipClassMemberNode]:
        """Parse class member"""
        # Could be constructor, destructor, method, or field
        if self.match('IDENTIFIER'):
            # Check if it's a constructor
            if self.current_token.value == self.peek_class_name():
                return self.parse_constructor()
            # Check if it's a destructor
            elif self.current_token.value.startswith('~'):
                return self.parse_destructor()
        
        # Check for operator overloading
        if self.match('OPERATOR'):
            return self.parse_operator_overload()
        
        # Regular member function or variable
        return self.parse_member_function_or_variable()
    
    def parse_constructor(self) -> HipConstructorNode:
        """Parse constructor"""
        name = self.consume('IDENTIFIER').value
        
        self.consume('LPAREN')
        params = self.parse_parameter_list()
        self.consume('RPAREN')
        
        # Parse member initializer list
        initializers = []
        if self.match('COLON'):
            self.advance()
            initializers = self.parse_member_initializer_list()
        
        # Parse body
        body = None
        if self.match('LBRACE'):
            body = self.parse_block()
        elif self.match('SEMICOLON'):
            self.advance()
        
        return HipConstructorNode(name, params, initializers, body)
    
    def parse_destructor(self) -> HipDestructorNode:
        """Parse destructor"""
        name = self.consume('IDENTIFIER').value  # ~ClassName
        
        self.consume('LPAREN')
        self.consume('RPAREN')
        
        # Parse body
        body = None
        if self.match('LBRACE'):
            body = self.parse_block()
        elif self.match('SEMICOLON'):
            self.advance()
        
        return HipDestructorNode(name, body)
    
    def parse_operator_overload(self) -> HipOperatorOverloadNode:
        """Parse operator overload"""
        self.consume('OPERATOR')
        
        # Parse operator
        operator = self.current_token.value
        self.advance()
        
        self.consume('LPAREN')
        params = self.parse_parameter_list()
        self.consume('RPAREN')
        
        # Parse body
        body = None
        if self.match('LBRACE'):
            body = self.parse_block()
        elif self.match('SEMICOLON'):
            self.advance()
        
        return HipOperatorOverloadNode(operator, params, body)
    
    def parse_member_function_or_variable(self) -> HipClassMemberNode:
        """Parse member function or variable"""
        # This is a simplified version - in practice, this would be more complex
        member_type = self.parse_type()
        name = self.consume('IDENTIFIER').value
        
        if self.match('LPAREN'):
            # It's a function
            self.advance()
            params = self.parse_parameter_list()
            self.consume('RPAREN')
            
            body = None
            if self.match('LBRACE'):
                body = self.parse_block()
            elif self.match('SEMICOLON'):
                self.advance()
            
            return HipMemberFunctionNode(name, member_type, params, body)
        else:
            # It's a variable
            self.consume('SEMICOLON')
            return HipMemberVariableNode(name, member_type)
    
    def parse_variable_declaration(self) -> HipVariableDeclarationNode:
        """Parse variable declaration"""
        var_type = self.parse_type()
        
        variables = []
        while True:
            name = self.consume('IDENTIFIER').value
            
            # Parse array dimensions
            dimensions = []
            while self.match('LBRACKET'):
                self.advance()
                if not self.match('RBRACKET'):
                    dimensions.append(self.parse_expression())
                else:
                    dimensions.append(None)
                self.consume('RBRACKET')
            
            # Parse initializer
            initializer = None
            if self.match('ASSIGN'):
                self.advance()
                initializer = self.parse_expression()
            
            variables.append(HipVariableNode(name, var_type, initializer, dimensions))
            
            if self.match('COMMA'):
                self.advance()
            else:
                break
        
        self.consume('SEMICOLON')
        
        return HipVariableDeclarationNode(variables)
    
    def parse_type(self) -> HipTypeNode:
        """Parse type specification"""
        # Parse type qualifiers
        qualifiers = []
        while self.match('CONST', 'VOLATILE', 'RESTRICT', 'UNSIGNED', 'SIGNED'):
            qualifiers.append(self.current_token.value)
            self.advance()
        
        # Parse base type
        if not self.current_token:
            self.error("Expected type")
        
        base_type = self.current_token.value
        self.advance()
        
        # Parse template arguments
        template_args = []
        if self.match('LT'):
            self.advance()
            while self.current_token and not self.match('GT'):
                if self.match('COMMA'):
                    self.advance()
                    continue
                template_args.append(self.parse_type())
            self.consume('GT')
        
        # Parse pointer/reference modifiers
        modifiers = []
        while self.match('STAR', 'AMPERSAND', 'AMPERSAND_AMPERSAND'):
            modifiers.append(self.current_token.value)
            self.advance()
        
        return HipTypeNode(base_type, qualifiers, template_args, modifiers)
    
    def parse_block(self) -> HipBlockNode:
        """Parse a block of statements"""
        self.consume('LBRACE')
        
        statements = []
        while self.current_token and not self.match('RBRACE'):
            if self.match('NEWLINE'):
                self.advance()
                continue
            
            stmt = self.parse_statement()
            if stmt:
                statements.append(stmt)
        
        self.consume('RBRACE')
        
        return HipBlockNode(statements)
    
    def parse_expression_statement(self) -> HipExpressionStatementNode:
        """Parse expression statement"""
        expr = self.parse_expression()
        
        if self.match('SEMICOLON'):
            self.advance()
        
        return HipExpressionStatementNode(expr)
    
    def parse_expression(self) -> HipExpressionNode:
        """Parse expression"""
        return self.parse_assignment_expression()
    
    def parse_assignment_expression(self) -> HipExpressionNode:
        """Parse assignment expression"""
        expr = self.parse_logical_or_expression()
        
        if self.match('ASSIGN', 'PLUS_ASSIGN', 'MINUS_ASSIGN', 'STAR_ASSIGN', 'SLASH_ASSIGN'):
            op = self.current_token.value
            self.advance()
            right = self.parse_assignment_expression()
            return HipBinaryOpNode(expr, op, right)
        
        return expr
    
    def parse_logical_or_expression(self) -> HipExpressionNode:
        """Parse logical OR expression"""
        expr = self.parse_logical_and_expression()
        
        while self.match('OR'):
            op = self.current_token.value
            self.advance()
            right = self.parse_logical_and_expression()
            expr = HipBinaryOpNode(expr, op, right)
        
        return expr
    
    def parse_logical_and_expression(self) -> HipExpressionNode:
        """Parse logical AND expression"""
        expr = self.parse_equality_expression()
        
        while self.match('AND'):
            op = self.current_token.value
            self.advance()
            right = self.parse_equality_expression()
            expr = HipBinaryOpNode(expr, op, right)
        
        return expr
    
    def parse_equality_expression(self) -> HipExpressionNode:
        """Parse equality expression"""
        expr = self.parse_relational_expression()
        
        while self.match('EQ', 'NE'):
            op = self.current_token.value
            self.advance()
            right = self.parse_relational_expression()
            expr = HipBinaryOpNode(expr, op, right)
        
        return expr
    
    def parse_relational_expression(self) -> HipExpressionNode:
        """Parse relational expression"""
        expr = self.parse_additive_expression()
        
        while self.match('LT', 'GT', 'LE', 'GE'):
            op = self.current_token.value
            self.advance()
            right = self.parse_additive_expression()
            expr = HipBinaryOpNode(expr, op, right)
        
        return expr
    
    def parse_additive_expression(self) -> HipExpressionNode:
        """Parse additive expression"""
        expr = self.parse_multiplicative_expression()
        
        while self.match('PLUS', 'MINUS'):
            op = self.current_token.value
            self.advance()
            right = self.parse_multiplicative_expression()
            expr = HipBinaryOpNode(expr, op, right)
        
        return expr
    
    def parse_multiplicative_expression(self) -> HipExpressionNode:
        """Parse multiplicative expression"""
        expr = self.parse_unary_expression()
        
        while self.match('STAR', 'SLASH', 'PERCENT'):
            op = self.current_token.value
            self.advance()
            right = self.parse_unary_expression()
            expr = HipBinaryOpNode(expr, op, right)
        
        return expr
    
    def parse_unary_expression(self) -> HipExpressionNode:
        """Parse unary expression"""
        if self.match('PLUS', 'MINUS', 'NOT', 'STAR', 'AMPERSAND', 'INCREMENT', 'DECREMENT'):
            op = self.current_token.value
            self.advance()
            expr = self.parse_unary_expression()
            return HipUnaryOpNode(op, expr)
        
        return self.parse_postfix_expression()
    
    def parse_postfix_expression(self) -> HipExpressionNode:
        """Parse postfix expression"""
        expr = self.parse_primary_expression()
        
        while True:
            if self.match('LBRACKET'):
                # Array access
                self.advance()
                index = self.parse_expression()
                self.consume('RBRACKET')
                expr = HipArrayAccessNode(expr, index)
            elif self.match('LPAREN'):
                # Function call
                self.advance()
                args = []
                while self.current_token and not self.match('RPAREN'):
                    if self.match('COMMA'):
                        self.advance()
                        continue
                    args.append(self.parse_expression())
                self.consume('RPAREN')
                expr = HipFunctionCallNode(expr, args)
            elif self.match('DOT'):
                # Member access
                self.advance()
                member = self.consume('IDENTIFIER').value
                expr = HipMemberAccessNode(expr, member)
            elif self.match('ARROW'):
                # Pointer member access
                self.advance()
                member = self.consume('IDENTIFIER').value
                expr = HipPointerMemberAccessNode(expr, member)
            elif self.match('INCREMENT', 'DECREMENT'):
                # Postfix increment/decrement
                op = self.current_token.value
                self.advance()
                expr = HipPostfixOpNode(expr, op)
            else:
                break
        
        return expr
    
    def parse_primary_expression(self) -> HipExpressionNode:
        """Parse primary expression"""
        if self.match('IDENTIFIER'):
            name = self.current_token.value
            self.advance()
            return HipIdentifierNode(name)
        
        elif self.match('INTEGER'):
            value = self.current_token.value
            self.advance()
            return HipLiteralNode(value, 'int')
        
        elif self.match('FLOAT'):
            value = self.current_token.value
            self.advance()
            return HipLiteralNode(value, 'float')
        
        elif self.match('STRING'):
            value = self.current_token.value
            self.advance()
            return HipLiteralNode(value, 'string')
        
        elif self.match('CHAR'):
            value = self.current_token.value
            self.advance()
            return HipLiteralNode(value, 'char')
        
        elif self.match('TRUE', 'FALSE'):
            value = self.current_token.value
            self.advance()
            return HipLiteralNode(value, 'bool')
        
        elif self.match('LPAREN'):
            self.advance()
            expr = self.parse_expression()
            self.consume('RPAREN')
            return expr
        
        elif self.match('LBRACE'):
            # Array/struct initializer
            return self.parse_initializer_list()
        
        else:
            self.error(f"Unexpected token in expression: {self.current_token.type if self.current_token else 'EOF'}")
    
    def parse_initializer_list(self) -> HipInitializerListNode:
        """Parse initializer list"""
        self.consume('LBRACE')
        
        elements = []
        while self.current_token and not self.match('RBRACE'):
            if self.match('COMMA'):
                self.advance()
                continue
            
            elements.append(self.parse_expression())
        
        self.consume('RBRACE')
        
        return HipInitializerListNode(elements)
    
    def parse_inheritance_list(self) -> List[HipInheritanceNode]:
        """Parse inheritance list"""
        inheritance = []
        
        while True:
            access = 'private'  # Default for class
            if self.match('PUBLIC', 'PRIVATE', 'PROTECTED'):
                access = self.current_token.value
                self.advance()
            
            base_class = self.parse_qualified_name()
            inheritance.append(HipInheritanceNode(base_class, access))
            
            if self.match('COMMA'):
                self.advance()
            else:
                break
        
        return inheritance
    
    def parse_member_initializer_list(self) -> List[HipMemberInitializerNode]:
        """Parse member initializer list"""
        initializers = []
        
        while True:
            member = self.consume('IDENTIFIER').value
            
            self.consume('LPAREN')
            args = []
            while self.current_token and not self.match('RPAREN'):
                if self.match('COMMA'):
                    self.advance()
                    continue
                args.append(self.parse_expression())
            self.consume('RPAREN')
            
            initializers.append(HipMemberInitializerNode(member, args))
            
            if self.match('COMMA'):
                self.advance()
            else:
                break
        
        return initializers
    
    def parse_qualified_name(self) -> str:
        """Parse qualified name (e.g., std::vector)"""
        name_parts = []
        
        if self.match('SCOPE'):
            name_parts.append('::')
            self.advance()
        
        name_parts.append(self.consume('IDENTIFIER').value)
        
        while self.match('SCOPE'):
            name_parts.append('::')
            self.advance()
            name_parts.append(self.consume('IDENTIFIER').value)
        
        return ''.join(name_parts)
    
    def is_function_declaration(self) -> bool:
        """Check if current position is a function declaration"""
        # This is a simplified heuristic
        # In practice, this would need more sophisticated lookahead
        saved_pos = self.pos
        
        try:
            # Skip qualifiers
            while self.match('INLINE', 'STATIC', 'EXTERN', 'VIRTUAL', 'CONST', 'CONSTEXPR'):
                self.advance()
            
            # Skip return type
            self.parse_type()
            
            # Check for identifier followed by parentheses
            if self.match('IDENTIFIER'):
                self.advance()
                if self.match('LPAREN'):
                    return True
        except:
            pass
        finally:
            # Restore position
            self.pos = saved_pos
            self.current_token = self.tokens[self.pos] if self.pos < len(self.tokens) else None
        
        return False
    
    def is_variable_declaration(self) -> bool:
        """Check if current position is a variable declaration"""
        # This is a simplified heuristic
        saved_pos = self.pos
        
        try:
            # Skip qualifiers
            while self.match('CONST', 'VOLATILE', 'STATIC', 'EXTERN'):
                self.advance()
            
            # Check for type followed by identifier
            if self.match('IDENTIFIER'):
                self.advance()
                if self.match('IDENTIFIER'):
                    return True
        except:
            pass
        finally:
            # Restore position
            self.pos = saved_pos
            self.current_token = self.tokens[self.pos] if self.pos < len(self.tokens) else None
        
        return False
    
    def peek_class_name(self) -> str:
        """Peek at the current class name (for constructor detection)"""
        # This would need to be implemented based on the current parsing context
        # For now, return empty string
        return ""


def parse_hip_code(code: str) -> HipProgramNode:
    """
    Parse HIP code and return the AST
    
    Args:
        code: HIP source code as string
        
    Returns:
        HipProgramNode representing the parsed AST
    """
    lexer = HipLexer(code)
    tokens = lexer.tokenize()
    parser = HipParser(tokens)
    return parser.parse() 