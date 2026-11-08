"""Receiver declarations that restate a recorded local are merged by looking only at the text between them.

``_restates_declaration`` copied the source between every earlier declaration of a name and the new one to test it
for whitespace. In a unit where a name recurs in each function, that copied most of the unit for every pair: on
PyTorch's ``BinaryKernel.metal`` it took 79 of the 180 profiled seconds. The checks below pin the merge rule.
"""

from crosstl.backend.Metal.member_address_types import merge_receiver_variable_types
from crosstl.backend.Metal.preprocessor import _MetalAddressableValueType


def _local(position, pointer="thread Pair*"):
    return (position, _MetalAddressableValueType("Pair", pointer_type=pointer))


def test_a_receiver_entry_after_only_whitespace_restates_the_local():
    # The local scanner records the statement start, the receiver scanner the type token after it.
    code = "x = 0;\n    Pair value;"
    locals_ = {"value": [_local(6)]}

    merge_receiver_variable_types(
        code, locals_, {"value": [(code.index("Pair"), "Pair")]}
    )

    assert [position for position, _type in locals_["value"]] == [6]


def test_a_receiver_entry_after_other_text_is_kept():
    code = "thread Pair value; " + "x = 1; " * 10_000 + "Pair value;"
    later = code.rindex("Pair value")
    locals_ = {"value": [_local(0)]}

    merge_receiver_variable_types(code, locals_, {"value": [(later, "Pair")]})

    assert [position for position, _type in locals_["value"]] == [0, later]


def test_an_entry_before_the_local_is_not_a_restatement():
    code = "Pair value; thread Pair value;"
    locals_ = {"value": [_local(12)]}

    merge_receiver_variable_types(code, locals_, {"value": [(0, "Pair")]})

    assert [position for position, _type in locals_["value"]] == [0, 12]
