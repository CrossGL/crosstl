import json

from . import typed_refusals
from .typed_refusals import exception_classes, scan, scan_source, untyped_names

CLASSES = """
class TranslationDiagnostic(ValueError):
    pass


class LoweringRefused(TranslationDiagnostic):
    pass


class CodedSyntaxError(SyntaxError):
    project_diagnostic_code = "project.translate.parse-failed"


class InitCodedError(ValueError):
    def __init__(self, message):
        super().__init__(message)
        self.project_diagnostic_code = "project.translate.x"


class PlainError(ValueError):
    pass


class NotAnException:
    pass


class _Signal(Exception):
    pass
"""

RAISES = """
def refuse(kind):
    if kind == 1:
        raise ValueError("untyped")
    if kind == 2:
        raise PlainError("untyped subclass")
    if kind == 3:
        raise SyntaxError("untyped parse error")
    if kind == 4:
        raise LoweringRefused(["typed"])
    if kind == 5:
        raise CodedSyntaxError("typed parse error")
    if kind == 6:
        raise InitCodedError("typed through __init__")
    if kind == 7:
        raise _Signal()
    if kind == 8:
        raise AssertionError("invariant")
    try:
        pass
    except KeyError as error:
        raise error
    raise


class Generator:
    def emit(self):
        raise NotImplementedError
"""


def test_exception_classes_are_typed_by_ancestry_or_a_diagnostic_code():
    classes = exception_classes([CLASSES])

    assert classes == {
        "CodedSyntaxError": True,
        "InitCodedError": True,
        "LoweringRefused": True,
        "PlainError": False,
        "TranslationDiagnostic": True,
        "_Signal": False,
    }


def test_untyped_raises_are_sites_and_typed_ones_are_not():
    untyped = untyped_names([CLASSES])

    assert "PlainError" in untyped and "_Signal" not in untyped
    assert scan_source(RAISES, untyped) == {
        ("refuse", "untyped-raise"): 3,
        ("Generator.emit", "untyped-raise"): 1,
    }


def test_a_name_defined_twice_is_typed_only_if_both_definitions_are():
    other = "class LoweringRefused(Exception):\n    pass\n"

    assert exception_classes([CLASSES, other])["LoweringRefused"] is False


def test_the_committed_baseline_matches_the_tree():
    baseline = json.loads(typed_refusals.BASELINE.read_text())
    assert baseline["schemaVersion"] == 1
    assert baseline["sites"] == scan()
