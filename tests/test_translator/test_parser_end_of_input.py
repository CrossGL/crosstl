"""The CrossGL parser rejects a construct that the end of input cuts off, instead of reading past the end.

Found by the importer crash fuzzer: ``layout(`` at the end of a stage block looped forever, because ``eat`` kept
consuming the end-of-input token it synthesizes once the token list is exhausted.
"""

from __future__ import annotations

import threading

import pytest

from crosstl.translator.lexer import Lexer
from crosstl.translator.parser import Parser


def parse(code):
    return Parser(Lexer(code).get_tokens()).parse()


def test_eat_never_consumes_the_end_of_input():
    parser = Parser(Lexer("").get_tokens())
    with pytest.raises(SyntaxError, match="Unexpected end of input"):
        parser.eat("EOF")
    assert parser.current_token[0] == "EOF"


@pytest.mark.parametrize(
    "code",
    [
        "shader S {\n    compute {\n        layout(",
        "shader S {\n    compute {\n        layout(local_size_x = 1,",
    ],
)
def test_layout_qualifier_cut_off_by_the_end_of_input_is_rejected(code):
    outcome = {}

    def run():
        try:
            parse(code)
        except SyntaxError as error:
            outcome["error"] = error

    worker = threading.Thread(target=run, daemon=True)
    worker.start()
    worker.join(20)
    assert not worker.is_alive(), "the parser did not finish"
    assert "error" in outcome


def test_complete_layout_qualifier_still_parses():
    parse(
        "shader S {\n    compute {\n        layout(local_size_x = 8) in;\n"
        "        void main() { }\n    }\n}\n"
    )
