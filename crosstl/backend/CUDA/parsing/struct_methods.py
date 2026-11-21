"""Member functions defined in CUDA struct and class bodies.

A struct body can define member functions next to its data members. The parser keeps each one whose receiver
CrossGL can pass as an ordinary parameter: a named, non-static, non-virtual, non-template member function with a
body. It stores them on the struct node as ``methods``, and :mod:`crosstl.backend.CUDA.struct_methods` turns each
into a free function that takes the receiver first. Every other member function, such as a constructor, an
operator, a template, a member of a struct template or a declaration defined outside the class, is skipped as
before. The name of every member function, kept or skipped, goes into ``struct_method_names`` on the program, so
that a member call the importer cannot resolve to a kept function is refused instead of written through as a member
call that no target declares.
"""

from ..CudaAst import FunctionNode, KernelNode

__all__ = ["CudaStructMethodMixin"]


class _StructMethodSink:
    """The member functions kept for one struct body while it is parsed."""

    def __init__(self, owner, keep):
        self.owner = owner
        self.keep = keep
        self.methods = []


class CudaStructMethodMixin:
    """Struct member-function members of ``CudaParser``."""

    STRUCT_METHOD_REJECTED_IDENTIFIERS = frozenset({"friend", "explicit", "virtual"})
    STRUCT_METHOD_POST_QUALIFIERS = frozenset({"noexcept"})

    @property
    def struct_method_sinks(self):
        sinks = self.__dict__.get("_struct_method_sinks")
        if sinks is None:
            sinks = self.__dict__["_struct_method_sinks"] = []
        return sinks

    @property
    def struct_method_names(self):
        names = self.__dict__.get("_struct_method_names")
        if names is None:
            names = self.__dict__["_struct_method_names"] = set()
        return names

    def begin_struct_methods(self, owner, start_index):
        """Start keeping the member functions of the struct named ``owner`` declared at ``start_index``.

        A struct that a ``template`` parameter list precedes keeps none: its member functions depend on the
        template's parameters.
        """
        previous = self.tokens[start_index - 1][0] if start_index > 0 else None
        keep = "<" not in str(owner) and previous not in {"GREATER_THAN", "SHIFT_RIGHT"}
        self.struct_method_sinks.append(_StructMethodSink(owner, keep))

    def end_struct_methods(self, node):
        """Attach the member functions kept since :meth:`begin_struct_methods` to ``node``."""
        sink = self.struct_method_sinks.pop()
        node.methods = list(sink.methods)
        return node

    def parse_or_skip_struct_method(self):
        """Keep the member function at the current token when it can be lowered; otherwise skip it."""
        sink = self.struct_method_sinks[-1] if self.struct_method_sinks else None
        owner = sink.owner if sink is not None else None
        name, plain, is_const = self.struct_method_signature_at(
            self.current_index, owner
        )
        if name is not None:
            self.struct_method_names.add(name)
        if sink is not None and sink.keep and plain:
            known_functions = set(self.user_function_names)
            function = self.parse_function()
            # A member function's name is not a free function name of the program.
            self.user_function_names = known_functions
            if (
                isinstance(function, FunctionNode)
                and not isinstance(function, KernelNode)
                and function.body is not None
            ):
                function.struct_method_owner = owner
                function.struct_method_is_const = is_const
                sink.methods.append(function)
            return
        self.skip_struct_method()

    def struct_method_signature_at(self, index, owner):
        """``(name, plain, is_const)`` of the member function that starts at ``index``.

        ``name`` is ``None`` for a constructor, destructor, conversion function or operator, which no member call
        names. ``plain`` says whether the parser keeps the function (see the module docstring), and ``is_const``
        whether it is declared ``const``.
        """
        tokens = self.tokens
        plain = True
        if index < len(tokens) and tokens[index][0] == "TEMPLATE":
            plain = False
            index = self.skip_optional_template_declaration_at_index(index)
        prefix_start = index
        while True:
            start = index
            index = self.skip_cpp_attribute_specifiers_at_index(index)
            index = self.skip_struct_method_specifiers_at_index(index)
            if index == start:
                break
        for token_type, token_value in tokens[prefix_start:index]:
            if token_type in {"STATIC", "VIRTUAL"} or (
                token_type == "IDENTIFIER"
                and token_value in self.STRUCT_METHOD_REJECTED_IDENTIFIERS
            ):
                plain = False
        if index >= len(tokens):
            return None, False, False
        token_type, token_value = tokens[index]
        if token_value == "operator" or token_type == "BITWISE_NOT":
            return None, False, False
        if token_type == "IDENTIFIER" and index + 1 < len(tokens):
            if tokens[index + 1][0] == "LPAREN" and token_value == owner:
                return None, False, False
        name_index = self.skip_type_at_index(
            index, allow_unknown_identifier_declarator=True
        )
        if name_index is None or name_index >= len(tokens):
            return None, False, False
        name_index = self.skip_return_pointer_suffix_at_index(name_index)
        while name_index < len(tokens) and (
            tokens[name_index][0] in self.POST_RETURN_FUNCTION_SPECIFIER_TOKENS
            or self.is_function_identifier_specifier_at_index(name_index)
        ):
            name_index += 1
        if name_index + 1 >= len(tokens) or tokens[name_index][1] == "operator":
            return None, False, False
        if not self.is_function_name_token(tokens[name_index]):
            return None, False, False
        name = tokens[name_index][1]
        if tokens[name_index + 1][0] != "LPAREN":
            return name, False, False
        index = self.skip_balanced_tokens_at_index(name_index + 1, "LPAREN", "RPAREN")
        if index is None:
            return name, False, False
        is_const = False
        while index < len(tokens):
            token_type, token_value = tokens[index]
            if token_type == "CONST":
                is_const = True
                index += 1
                continue
            if (
                token_type == "IDENTIFIER"
                and token_value in self.STRUCT_METHOD_POST_QUALIFIERS
            ):
                index += 1
                if index < len(tokens) and tokens[index][0] == "LPAREN":
                    plain = False
                continue
            break
        # A body or a plain declaration; ``volatile``, ref-qualifiers, ``override``, ``final``, ``= default``,
        # ``= delete``, ``= 0`` and trailing return types are not kept.
        if index >= len(tokens) or tokens[index][0] not in {"LBRACE", "SEMICOLON"}:
            plain = False
        return name, plain, is_const
