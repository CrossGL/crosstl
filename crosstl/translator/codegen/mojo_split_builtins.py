"""Mojo lowering of ``frexp(x, e)`` and ``modf(x, i)``, whose second argument is an output.

Mojo's ``frexp(x)`` returns the pair (mantissa, exponent), the exponent as a float of the argument's type, and
``modf(x)`` returns (integral part, fractional part). CrossGL's two-argument forms return the mantissa or the
fractional part and store the other half through their second argument, so the call becomes a temporary that
holds the pair, a store to the output argument in the statement's prelude, and a read of the returned half.
"""

from ..refusals import MojoRefusal

__all__ = ["MojoSplitBuiltinMixin"]

# For each built-in: the index of the half the call returns and the index of the half stored to its output.
_SPLIT_BUILTINS = {"frexp": (0, 1), "modf": (1, 0)}


class MojoSplitBuiltinMixin:
    """``frexp`` and ``modf`` call members of ``MojoCodeGen``."""

    def generate_mojo_split_builtin_call(self, func_name, args):
        """The Mojo expression of a two-argument ``frexp`` or ``modf`` call, or ``None`` for any other call."""
        halves = _SPLIT_BUILTINS.get(func_name)
        if halves is None or len(args) != 2:
            return None
        if func_name in self.function_return_types:
            return None
        if not self.expression_prelude_stack:
            raise MojoRefusal(
                f"Mojo {func_name} with an output argument needs a statement to store the output in, "
                "and this call is not inside one"
            )
        returned, stored = halves
        value_dtype = self.expression_mojo_dtype(args[0])
        output_dtype = self.expression_mojo_dtype(args[1])
        if value_dtype is None or output_dtype is None:
            raise MojoRefusal(
                f"Mojo {func_name}: the types of its arguments are unknown, so the output cannot be converted"
            )
        self.validate_resource_write_access(args[1], f"{func_name} output")
        value = self.generate_expression(args[0])
        output = self.generate_expression(args[1])
        context = self.expression_prelude_stack[-1]
        indent_str = "    " * context["indent"]
        temp_name = f"__cgl_{func_name}_{self.lambda_counter}"
        self.lambda_counter += 1
        stored_value = f"{temp_name}[{stored}]"
        if output_dtype != value_dtype:
            stored_value = f"{stored_value}.cast[{output_dtype}]()"
        context["lines"].append(f"{indent_str}var {temp_name} = {func_name}({value})\n")
        context["lines"].append(f"{indent_str}{output} = {stored_value}\n")
        return f"{temp_name}[{returned}]"
