"""Parser lookaheads undo their ``>>`` splits without copying the token list."""

import random

from crosstl.translator.lexer import Lexer
from crosstl.translator.parser import Parser


def shift_tokens():
    return Lexer("a >> b >> c >> d >> e >> f").get_tokens()


def split_at(parser, position):
    parser.pos = position
    parser.current_token = parser.tokens[position]
    parser.eat_type_greater_than()


def test_undo_restores_the_tokens_and_keeps_the_list_object():
    tokens = shift_tokens()
    original = list(tokens)
    parser = Parser(tokens)
    mark = parser.mark_token_edits()
    split_at(parser, 1)
    split_at(parser, 4)
    assert parser.tokens is tokens
    assert len(tokens) == len(original) + 2
    parser.undo_token_edits(mark)
    assert parser.tokens is tokens
    assert tokens == original


def test_nested_marks_match_list_snapshots():
    rng = random.Random(5)
    for _ in range(200):
        parser = Parser(shift_tokens())
        marks = []
        for _ in range(12):
            action = rng.random()
            if action < 0.4:
                marks.append((parser.mark_token_edits(), list(parser.tokens)))
            elif action < 0.75:
                candidates = [
                    index
                    for index, token in enumerate(parser.tokens)
                    if token[0] == "BITWISE_SHIFT_RIGHT"
                ]
                if candidates:
                    split_at(parser, rng.choice(candidates))
            elif marks:
                mark, snapshot = marks.pop()
                parser.undo_token_edits(mark)
                assert parser.tokens == snapshot


def test_generic_lookahead_still_splits_and_restores_shift_tokens():
    source = """
    shader S {
        struct Box<T> { T value; };
        compute {
            void main() {
                Box<Box<int>> nested;
                int shifted = 8 >> 2;
                int compared = shifted < 3 ? 1 : 0;
            }
        }
    }
    """
    ast = Parser(Lexer(source).get_tokens()).parse()
    assert ast is not None
