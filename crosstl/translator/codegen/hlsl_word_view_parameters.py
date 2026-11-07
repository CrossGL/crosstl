"""Overloads of HLSL helpers whose storage pointer parameters receive word views.

A Metal helper may declare ``device atomic<float>* object`` while its caller
passes ``reinterpret_cast<device atomic<float>*>(words)`` of an ``atomic<uint>``
buffer, as ``Sum<complex64_t>::atomic_update`` does with the lanes of its
output. DirectX passes a storage pointer as a ``RWStructuredBuffer<T>`` and an
offset, so the declared parameter cannot receive the ``uint`` buffer (#4025).

When every element of the caller's view is one whole 32-bit backing word, the
emitter adds an overload of the helper whose parameter takes the backing buffer
and addresses it through the caller's word view: loads and stores become bit
casts, and atomics run on the backing words, as they do for a view the kernel
holds itself. HLSL overloads functions on resource parameter types, so each call
site resolves to the overload for its arguments. The declared helper is kept.

Call-site views are only known while a program is generated, and an overload's
own calls can forward its views further, so the program is generated again until
the recorded views stop growing.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any

from ..ast import PointerReinterpretNode
from .ast_walk import walk_ast_values
from .pointer_reinterpret import scalar_storage_layout
from .stage_utils import STAGE_QUALIFIER_NAMES, function_stage_name
from .storage_word_views import is_same_width_word_view

__all__ = ["HLSLWordViewParametersMixin", "WordViewArgument"]


@dataclass(frozen=True)
class WordViewArgument:
    """The word view one call site passes to one helper parameter."""

    parameter: str
    backing_type: str
    view_element_type: str
    element_type: str
    source_element_type: str
    source_member: str | None
    source_layout: Any
    target_layout: Any


def _parameters(function):
    return list(getattr(function, "parameters", getattr(function, "params", [])) or [])


def _has_pointer_reinterpretation(root):
    return any(
        isinstance(node, PointerReinterpretNode) for node in walk_ast_values(root)
    )


def _variant_order(variant):
    return tuple(
        (
            argument.parameter,
            argument.backing_type,
            argument.view_element_type,
            argument.source_member or "",
        )
        for argument in sorted(variant, key=lambda argument: argument.parameter)
    )


class HLSLWordViewParametersMixin:
    """Overloads of ``HLSLCodeGen`` helpers for word-view storage pointer arguments."""

    # Each pass adds at least one view; a forwarding chain needs as many passes.
    HLSL_WORD_VIEW_PASS_LIMIT = 16
    # Word-view overloads of one helper beyond this are refused, not emitted.
    HLSL_WORD_VIEW_VARIANT_LIMIT = 16

    hlsl_word_view_variants = {}
    hlsl_word_view_calls = None
    current_hlsl_word_view_parameters = {}

    def hlsl_storage_pointer_helpers(self, ast):
        """Whether a callable (non-entry) function has a storage pointer parameter."""
        functions = list(getattr(ast, "functions", []) or [])
        entries = set()
        for _stage_type, stage in getattr(ast, "stages", {}).items():
            functions.extend(getattr(stage, "local_functions", []) or [])
            entry = getattr(stage, "entry_point", None)
            if entry is not None:
                entries.add(id(entry))
        return any(
            id(function) not in entries
            and function_stage_name(function) not in STAGE_QUALIFIER_NAMES
            and any(
                self.hlsl_resource_pointer_parameter(parameter)
                for parameter in _parameters(function)
            )
            for function in functions
        )

    def generate_hlsl_program_with_word_view_parameters(self, ast, target_stage):
        """Generate HLSL, adding word-view overloads until every call site has one."""
        self.hlsl_word_view_variants = {}
        self.current_hlsl_word_view_parameters = {}
        if not (
            self.hlsl_storage_pointer_helpers(ast)
            and _has_pointer_reinterpretation(ast)
        ):
            self.hlsl_word_view_calls = None
            return self.generate_hlsl_program_with_resource_access_variants(
                ast, target_stage
            )

        pristine = deepcopy(ast)
        variants = {}
        program = ast
        try:
            for _attempt in range(self.HLSL_WORD_VIEW_PASS_LIMIT):
                self.hlsl_word_view_variants = {
                    name: tuple(sorted(combinations, key=_variant_order))
                    for name, combinations in sorted(variants.items())
                }
                self.hlsl_word_view_calls = {}
                code = self.generate_hlsl_program_with_resource_access_variants(
                    program, target_stage
                )
                merged = {
                    name: set(combinations) for name, combinations in variants.items()
                }
                for name, combinations in self.hlsl_word_view_calls.items():
                    merged.setdefault(name, set()).update(combinations)
                if merged == variants:
                    return code
                self.hlsl_check_word_view_variant_limit(merged)
                variants = merged
                program = deepcopy(pristine)
        finally:
            self.hlsl_word_view_calls = None
        raise self.hlsl_word_view_parameter_error(
            "DirectX word-view overloads of storage pointer helpers did not "
            f"converge in {self.HLSL_WORD_VIEW_PASS_LIMIT} passes",
            reason="word-view-overloads-unresolved",
        )

    def hlsl_check_word_view_variant_limit(self, variants):
        for name in sorted(variants):
            if len(variants[name]) > self.HLSL_WORD_VIEW_VARIANT_LIMIT:
                raise self.hlsl_word_view_parameter_error(
                    f"DirectX helper '{name}' needs more than "
                    f"{self.HLSL_WORD_VIEW_VARIANT_LIMIT} word-view overloads for "
                    "its storage pointer parameters",
                    function_name=name,
                    reason="word-view-overload-limit-exceeded",
                )

    def hlsl_word_view_parameter_error(self, message, **details):
        from .directx_codegen import DirectXResourcePointerParameterError

        return DirectXResourcePointerParameterError(message, **details)

    def hlsl_word_view_argument(self, parameter_name, expected_type, binding):
        """Return the word view ``binding`` passes for ``expected_type``, or ``None``.

        The view qualifies when it addresses whole 32-bit backing words and its
        elements have the layout the parameter's own elements have.
        """
        reinterpretation = binding.get("pointer_reinterpretation")
        if not is_same_width_word_view(reinterpretation):
            return None
        if reinterpretation.get("physical_storage_mapping"):
            return None
        backing_type = self.hlsl_resource_pointer_element_type(
            binding.get("resource_type")
        )
        view_layout = scalar_storage_layout(expected_type)
        if (
            not backing_type
            or view_layout is None
            or view_layout != reinterpretation.get("target_layout")
        ):
            return None
        return WordViewArgument(
            parameter=str(parameter_name),
            backing_type=str(backing_type),
            view_element_type=str(binding.get("view_element_type") or expected_type),
            element_type=str(binding.get("element_type") or expected_type),
            source_element_type=str(binding.get("source_element_type") or backing_type),
            source_member=reinterpretation.get("source_member"),
            source_layout=reinterpretation["source_layout"],
            target_layout=reinterpretation["target_layout"],
        )

    def record_hlsl_word_view_call(self, function_name, arguments):
        """Record the word views one call passes, or refuse them outside a pass."""
        calls = self.hlsl_word_view_calls
        if not arguments:
            return True
        if calls is None:
            return False
        key = (
            function_name
            if function_name in self.function_hlsl_resource_pointer_parameters
            else self.hlsl_materialized_function_name(function_name)
        )
        calls.setdefault(key, set()).add(frozenset(arguments))
        return True

    def hlsl_word_view_function_variants(self, function):
        """The word-view combinations to emit for ``function`` besides its declaration."""
        if not self.hlsl_word_view_variants:
            return ()
        name = self.hlsl_function_declaration_name(function)
        return self.hlsl_word_view_variants.get(name, ())

    def hlsl_word_view_parameters_for(self, function, variant):
        """Map the parameters of ``function`` that ``variant`` views to their views."""
        views = {argument.parameter: argument for argument in variant}
        return {
            id(parameter): views[parameter.name]
            for parameter in _parameters(function)
            if getattr(parameter, "name", None) in views
        }

    def hlsl_word_view_parameter(self, parameter):
        """The word view the overload being emitted gives ``parameter``, or ``None``."""
        return self.current_hlsl_word_view_parameters.get(id(parameter))

    def hlsl_word_view_parameter_contract(self, parameter, contract):
        """Bind ``parameter``'s contract to its view's backing buffer type."""
        view = self.hlsl_word_view_parameter(parameter)
        if view is None or contract is None:
            return contract
        resource_name = self.hlsl_resource_type_name(contract["resource_type"])
        return {**contract, "resource_type": f"{resource_name}<{view.backing_type}>"}

    def hlsl_word_view_parameter_binding(self, parameter, binding):
        """Address the overload's backing buffer through the caller's word view."""
        view = self.hlsl_word_view_parameter(parameter)
        if view is None:
            return binding
        return {
            **binding,
            "byte_offset": "0",
            "element_type": view.element_type,
            "view_element_type": view.view_element_type,
            "source_element_type": view.source_element_type,
            "pointer_reinterpretation": {
                "source_layout": view.source_layout,
                "target_layout": view.target_layout,
                "target_width": 1,
                "source_member": view.source_member,
            },
        }
