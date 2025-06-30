import torch
import torch.fx as fx
from typing import Tuple

# local import

from .to_jax import fx_to_jax


class TorchLexer(torch.nn.Module):
    def __init__(self, code: str):
        super(TorchLexer, self).__init__()
        self.code = code
        self.length = len(code)

        # Token IDs as constants (ToDo : Definitely a better way to store data)
        self.PLUS_ID = 1
        self.MINUS_ID = 2
        self.MULTIPLY_ID = 3
        self.DIVIDE_ID = 4
        self.EQUALS_ID = 5
        self.NUMBER_ID = 6
        self.LPAREN_ID = 7
        self.RPAREN_ID = 8
        self.IDENTIFIER_ID = 9
        self.DOT_ID = 10
        self.COMMA_ID = 12
        self.UNKNOWN_ID = 13
        self.EOF_ID = 14
        self.TORCH_ID = 15
        self.FUNC_CALL_ID = 16
        self.MODULE_ID = 17

        # List of valid torch functions ( Maybe break into frontend)
        self.TORCH_FUNCTIONS = (
            "tensor",
            "add",
            "sub",
            "mul",
            "div",
            "cat",
            "reshape",
        )

    def _is_digit(self, c: str) -> bool:
        """Check if character is a digit."""
        return "0" <= c <= "9"

    def _is_identifier_start(self, c: str) -> bool:
        """Check if the character can start an identifier."""
        return c.isalpha() or c == "_"

    def _is_identifier_char(self, c: str) -> bool:
        """Check if the character can be part of an identifier."""
        return c.isalnum() or c == "_"

    def _match_number(self, pos: int) -> Tuple[int, int, str]:
        """Match a number."""
        number_value = ""
        while pos < self.length and self._is_digit(self.code[pos]):
            number_value += self.code[pos]
            pos += 1
        return pos, self.NUMBER_ID, number_value

    def _match_identifier(self, pos: int) -> Tuple[int, int, str]:
        """Match an identifier or keyword."""
        identifier = ""
        while pos < self.length and self._is_identifier_char(self.code[pos]):
            identifier += self.code[pos]
            pos += 1

        # Check for 'torch' as a prefix
        if identifier == "torch":
            if pos < self.length and self.code[pos] == ".":
                pos += 1
                func_name = ""
                while pos < self.length and self._is_identifier_char(self.code[pos]):
                    func_name += self.code[pos]
                    pos += 1
                # Check against the list of valid torch functions
                if func_name in self.TORCH_FUNCTIONS:
                    return pos, self.FUNC_CALL_ID, f"torch.{func_name}"
                elif func_name == "nn":
                    return pos, self.MODULE_ID, "torch.nn"
                return pos, self.UNKNOWN_ID, f"torch.{func_name}"
            return pos, self.TORCH_ID, "torch"

        return pos, self.IDENTIFIER_ID, identifier

    def _match_operator(self, pos: int) -> Tuple[int, int, str]:
        """Match single-character operators."""
        c = self.code[pos]
        if c == "+":
            return pos + 1, self.PLUS_ID, "+"
        elif c == "-":
            return pos + 1, self.MINUS_ID, "-"
        elif c == "*":
            return pos + 1, self.MULTIPLY_ID, "*"
        elif c == "/":
            return pos + 1, self.DIVIDE_ID, "/"
        elif c == "=":
            return pos + 1, self.EQUALS_ID, "="
        elif c == ".":
            return pos + 1, self.DOT_ID, "."
        elif c == ",":
            return pos + 1, self.COMMA_ID, ","
        elif c == "(":
            return pos + 1, self.LPAREN_ID, "("
        elif c == ")":
            return pos + 1, self.RPAREN_ID, ")"
        return pos + 1, self.UNKNOWN_ID, c

    def tokenize(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Tokenize the input code and return tensors."""
        pos = 0
        token_ids = []
        token_values = []

        while pos < self.length:
            c = self.code[pos]

            # Skip whitespace
            if c.isspace():
                pos += 1
                continue

            # Match numbers
            if self._is_digit(c):
                pos, token_id, text = self._match_number(pos)

            # Match identifiers or keywords
            elif self._is_identifier_start(c):
                pos, token_id, text = self._match_identifier(pos)

            # Match operators and symbols
            else:
                pos, token_id, text = self._match_operator(pos)

            # Add token if valid
            if token_id != -1:
                token_ids.append(token_id)
                token_values.extend(ord(ch) for ch in text)

        # Add EOF token
        token_ids.append(self.EOF_ID)
        token_values.append(0)

        # Convert to tensor
        token_ids_tensor = torch.tensor(token_ids, dtype=torch.int32)
        token_values_tensor = torch.tensor(token_values, dtype=torch.int32)
        return token_ids_tensor, token_values_tensor


# ToDo : Find a better wrapping method!
class TorchLexerFX(torch.nn.Module):
    def __init__(self, code: str):
        super(TorchLexerFX, self).__init__()
        self.lexer = TorchLexer(code)

    def forward(self, pos: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Tokenize using FX-compatible tracing."""
        return self.lexer.tokenize()


# Testing (ToDo : improve it for niche cases)
code = """
x = torch.tensor([1, 2, 3])
y = torch.add(x, 5)
m = torch.nn.Module()
"""


tracer = TorchLexerFX(code)
graph_module = fx.symbolic_trace(tracer)

token_ids, token_values = graph_module(torch.tensor([0], dtype=torch.int32))

print("=== FX Graph with Metadata ===")
for node in graph_module.graph.nodes:
    if node.op == "get_attr":
        target_name = node.target
        tensor_value = getattr(graph_module, target_name)
        print(
            f"Node: {node.name}\n"
            f"  Type: {node.op}\n"
            f"  Target: {node.target}\n"
            f"  Shape: {list(tensor_value.shape)}\n"
            f"  Dtype: {tensor_value.dtype}\n"
            f"  Value: {tensor_value.tolist()}\n"
        )
    else:
        print(
            f"Node: {node.name}\n"
            f"  Type: {node.op}\n"
            f"  Target: {node.target}\n"
            f"  Args: {node.args}\n"
            f"  Kwargs: {node.kwargs}\n"
        )


print("=== Token Details ===")
print("Token IDs:", token_ids.tolist())
print("Token Values:", token_values.tolist())


# No constant folding test case


class MyModel(torch.nn.Module):
    def forward(self, x):
        y = x + 5
        small_tensor = torch.tensor([1, 2, 3])
        large_tensor = torch.tensor([[1, 2], [3, 4]])  # Large tensor (dynamic)
        z = y * small_tensor
        w = z + large_tensor
        return w


model = MyModel()
graph_module = fx.symbolic_trace(model)

jax_code = fx_to_jax(graph_module)

print("=== Generated JAX Code ===")
print(jax_code)


"""
=== Corrected JAX Code ===
import jax
import jax.numpy as jnp


def jax_function(pos):
    x = pos  # Input placeholder
    add = jnp.add(x, 5)
    _tensor_constant0 = jnp.array([1, 2, 3], dtype=jnp.int32)  # Constant tensor
    mul = jnp.multiply(add, _tensor_constant0)
    _tensor_constant1 = jnp.array([[1, 2], [3, 4]], dtype=jnp.int32)  # Constant tensor
    add_1 = jnp.add(mul, _tensor_constant1)
    return add_1
"""
