"""Writable overloads of HLSL helpers whose storage pointer parameters are read-only.

A read-only storage pointer parameter (CrossGL ``const buffer T* p``, MSL ``const device T* p``) lowers to a
``StructuredBuffer<T>`` parameter. MSL and CrossGL let a writable pointer initialize a pointer to const, but HLSL
has no conversion from ``RWStructuredBuffer<T>`` to ``StructuredBuffer<T>``: DXC rejects such a call with "no
matching function". HLSL does overload functions on resource parameter types, so the emitter adds an overload of
the helper whose read-only parameters take ``RWStructuredBuffer<T>`` for every combination that a call site passes
a writable buffer to. The declared overload is always kept.

Call-site resources are only known while a program is generated, and an overload's own calls can forward its new
writable parameters further, so the program is generated again until the recorded combinations stop growing.
"""

from __future__ import annotations

from copy import deepcopy

from .stage_utils import STAGE_QUALIFIER_NAMES, function_stage_name

__all__ = ["HLSLResourceAccessVariantsMixin"]


def _parameters(function):
    return list(getattr(function, "parameters", getattr(function, "params", [])) or [])


def _variant_order(variant):
    return (len(variant), tuple(sorted(variant)))


class HLSLResourceAccessVariantsMixin:
    """Overloads of ``HLSLCodeGen`` helpers for writable arguments to read-only storage pointers."""

    # Each pass adds at least one combination; a call chain of this depth needs as many passes.
    HLSL_RESOURCE_ACCESS_PASS_LIMIT = 16
    # Combinations of one helper beyond this are refused rather than emitted.
    HLSL_RESOURCE_ACCESS_VARIANT_LIMIT = 16

    hlsl_resource_access_variants = {}
    hlsl_resource_access_calls = None
    current_hlsl_resource_access_promotions = frozenset()

    def hlsl_readonly_storage_pointer_helpers(self, ast):
        """Whether a callable (non-entry) function has a read-only storage pointer parameter."""
        functions = list(getattr(ast, "functions", []) or [])
        entries = set()
        for _stage_type, stage in getattr(ast, "stages", {}).items():
            functions.extend(getattr(stage, "local_functions", []) or [])
            entry = getattr(stage, "entry_point", None)
            if entry is not None:
                entries.add(id(entry))
        for function in functions:
            if id(function) in entries:
                continue
            if function_stage_name(function) in STAGE_QUALIFIER_NAMES:
                continue
            for parameter in _parameters(function):
                if (
                    self.hlsl_resource_pointer_parameter(parameter)
                    and self.hlsl_resource_pointer_parameter_access(parameter) == "read"
                ):
                    return True
        return False

    def generate_hlsl_program_with_resource_access_variants(self, ast, target_stage):
        """Generate HLSL, adding writable overloads until every call site has one."""
        self.hlsl_resource_access_variants = {}
        self.current_hlsl_resource_access_promotions = frozenset()
        if not self.hlsl_readonly_storage_pointer_helpers(ast):
            self.hlsl_resource_access_calls = None
            return self.generate_hlsl_program_with_narrow_pointer_layouts(
                ast, target_stage
            )

        pristine = deepcopy(ast)
        variants = {}
        program = ast
        try:
            for _attempt in range(self.HLSL_RESOURCE_ACCESS_PASS_LIMIT):
                self.hlsl_resource_access_variants = {
                    name: tuple(sorted(combinations, key=_variant_order))
                    for name, combinations in variants.items()
                }
                self.hlsl_resource_access_calls = {}
                code = self.generate_hlsl_program_with_narrow_pointer_layouts(
                    program, target_stage
                )
                merged = {
                    name: set(combinations) for name, combinations in variants.items()
                }
                for name, combinations in self.hlsl_resource_access_calls.items():
                    merged.setdefault(name, set()).update(combinations)
                if merged == variants:
                    return code
                self.hlsl_check_resource_access_variant_limit(merged)
                variants = merged
                program = deepcopy(pristine)
        finally:
            self.hlsl_resource_access_calls = None
        raise self.hlsl_resource_access_variant_error(
            "DirectX writable overloads of read-only storage pointer helpers did "
            f"not converge in {self.HLSL_RESOURCE_ACCESS_PASS_LIMIT} passes",
            reason="resource-access-variants-unresolved",
        )

    def hlsl_check_resource_access_variant_limit(self, variants):
        for name in sorted(variants):
            if len(variants[name]) > self.HLSL_RESOURCE_ACCESS_VARIANT_LIMIT:
                raise self.hlsl_resource_access_variant_error(
                    f"DirectX helper '{name}' needs more than "
                    f"{self.HLSL_RESOURCE_ACCESS_VARIANT_LIMIT} writable overloads "
                    "for its read-only storage pointer parameters",
                    function_name=name,
                    reason="resource-access-variant-limit-exceeded",
                )

    def hlsl_resource_access_variant_error(self, message, **details):
        from .directx_codegen import DirectXResourcePointerParameterError

        return DirectXResourcePointerParameterError(message, **details)

    def record_hlsl_resource_access_call(self, function_name, resource_bindings):
        """Record which read-only storage pointer parameters a call passes a writable buffer to."""
        calls = self.hlsl_resource_access_calls
        if calls is None or not resource_bindings:
            return
        promoted = frozenset(
            binding["call_parameter_name"]
            for binding in resource_bindings.values()
            if (binding.get("call_contract") or {}).get("access") == "read"
            and self.hlsl_resource_type_name(binding.get("resource_type"))
            == "RWStructuredBuffer"
        )
        if not promoted:
            return
        key = (
            function_name
            if function_name in self.function_hlsl_resource_pointer_parameters
            else self.hlsl_materialized_function_name(function_name)
        )
        calls.setdefault(key, set()).add(promoted)

    def hlsl_resource_access_function_variants(self, function):
        """The writable combinations to emit for ``function`` besides its declaration."""
        if not self.hlsl_resource_access_variants:
            return ()
        name = self.hlsl_function_declaration_name(function)
        return self.hlsl_resource_access_variants.get(name, ())

    def hlsl_resource_access_promotions_for(self, function, variant):
        """Identities of the parameters of ``function`` that ``variant`` makes writable."""
        return frozenset(
            id(parameter)
            for parameter in _parameters(function)
            if getattr(parameter, "name", None) in variant
        )

    def hlsl_resource_access_promoted(self, parameter):
        """Whether the overload being emitted takes ``parameter`` as a writable buffer."""
        return id(parameter) in self.current_hlsl_resource_access_promotions
