"""Typed refusals for the sites that raise at the point where a construct is rejected.

Generators and front ends reject a program at thousands of sites, each with a reason that names the construct. Those
sites used to raise a built-in ``ValueError``, ``SyntaxError`` or ``TypeError``, which the project pipeline, the
corpus runner and the fuzzer cannot tell apart from a translator bug. Each site now raises the class here for the
component it belongs to, which carries a stable code (``project.translate.<component>-<kind>``) and keeps everything
else unchanged:

- the built-in base, so ``except ValueError`` and ``except SyntaxError`` handlers behave as before;
- the message, which is the reason exactly as the site wrote it;
- the place in the refusal order. A site refusal *defers to pending refusals*: when an emitter or importer has
  already recorded refusals (:mod:`.codegen.lowering_refusals`) or an undeclared name
  (:mod:`.semantics.declared_identifiers`), those are reported, with the site's message listed last, because a
  site refusal raised after them is often the consequence of the placeholder an earlier refusal left. A diagnostic
  with a code of its own (any other :class:`~.diagnostics.TranslationDiagnostic`) still takes precedence.

The codes are grouped by component rather than by site: the message says which construct was refused. A site whose
construct needs its own code gets a dedicated class instead (``WGSLEnumUnsupportedError``, ``LoweringRefused``).
An exception that signals a translator defect, rather than a refusal, is an ``AssertionError``.
"""

from __future__ import annotations

from .diagnostics import CODE_PREFIX, TranslationDiagnostic

__all__ = [
    "CrossGLRefusal",
    "CrossGLSyntaxError",
    "CudaRefusal",
    "CudaSyntaxError",
    "DirectXRefusal",
    "DirectXSourceRefusal",
    "DirectXSyntaxError",
    "HipRefusal",
    "HipSourceRefusal",
    "HipSyntaxError",
    "MetadataContractError",
    "MetalRefusal",
    "MetalSourceRefusal",
    "MetalSyntaxError",
    "MojoRefusal",
    "MojoSyntaxError",
    "OpenCLSyntaxError",
    "OpenGLRefusal",
    "OpenGLSourceRefusal",
    "OpenGLSyntaxError",
    "RegistryRefusal",
    "RustRefusal",
    "RustSyntaxError",
    "SlangRefusal",
    "SlangSyntaxError",
    "SourceSyntaxError",
    "SpirvSyntaxError",
    "TargetRefusal",
    "TranslationRefusal",
    "TranslatorArgumentError",
    "TranslatorIndexError",
    "TranslatorKeyError",
    "VulkanRefusal",
    "WGSLRefusal",
    "supersedes_pending_refusals",
]


def _is_typed(error) -> bool:
    code = getattr(error, "project_diagnostic_code", None)
    return isinstance(error, TranslationDiagnostic) or (
        isinstance(code, str) and bool(code)
    )


def supersedes_pending_refusals(error) -> bool:
    """Whether ``error`` is a typed diagnostic that replaces the refusals recorded before it.

    A site refusal (``defers_to_pending``) does not: the recorded refusals are reported with it.
    """
    return _is_typed(error) and not getattr(error, "defers_to_pending", False)


class TranslationRefusal(TranslationDiagnostic):
    """A program refused at the site that found the construct; the message is the site's reason, unchanged.

    Subclasses set ``code`` and ``default_target``. The class is a ``ValueError``, like the site's former exception.
    """

    code = f"{CODE_PREFIX}refused"
    default_target: str | None = None
    defers_to_pending = True

    def __init__(self, reason="", *, span=None, target=None):
        self.reason = str(reason)
        self.span = span
        self.target = self.default_target if target is None else target
        # The message stays exactly what the site wrote; the code is an attribute, not part of the text.
        ValueError.__init__(self, reason)


class SourceSyntaxError(SyntaxError):
    """Source text a front end cannot parse. The class is a ``SyntaxError`` and the message is unchanged.

    Subclasses set ``code`` and ``source_name``. It is deliberately not a ``ValueError``: handlers that catch
    ``ValueError`` around a parse did not catch the former ``SyntaxError`` either.
    """

    code = f"{CODE_PREFIX}syntax-error"
    source_name: str | None = None
    defers_to_pending = True

    @property
    def project_diagnostic_code(self) -> str:
        return self.code

    @property
    def reason(self) -> str:
        return str(self.msg if self.msg is not None else self)

    @property
    def target(self) -> str:
        return "cgl"

    def to_json(self) -> dict:
        return {
            "code": self.code,
            "reason": self.reason,
            "span": None,
            "target": self.target,
        }


class TranslatorArgumentError(TypeError):
    """A translator API called with an argument of the wrong type, for example a non-boolean option."""

    code = f"{CODE_PREFIX}invalid-argument"
    defers_to_pending = True

    @property
    def project_diagnostic_code(self) -> str:
        return self.code


class TranslatorKeyError(KeyError):
    """A name a translator lookup does not know. The class is a ``KeyError``, as the lookup's callers expect."""

    code = f"{CODE_PREFIX}unknown-name"
    defers_to_pending = True

    @property
    def project_diagnostic_code(self) -> str:
        return self.code


class TranslatorIndexError(IndexError):
    """An index outside a translator table. The class is an ``IndexError``, as the table's callers expect."""

    code = f"{CODE_PREFIX}index-out-of-range"
    defers_to_pending = True

    @property
    def project_diagnostic_code(self) -> str:
        return self.code


# CrossGL itself and the metadata that travels with a program.


class CrossGLSyntaxError(SourceSyntaxError):
    """CrossGL source the CrossGL lexer or parser cannot read."""

    code = f"{CODE_PREFIX}crossgl-syntax-error"
    source_name = "cgl"


class CrossGLRefusal(TranslationRefusal):
    """A CrossGL program that violates a rule of the IR: a type, a role, a stage or an attribute."""

    code = f"{CODE_PREFIX}crossgl-refused"
    default_target = "cgl"


class MetadataContractError(TranslationRefusal):
    """Program metadata (artifact contracts, dispatch regions, resource identity or storage) that is malformed."""

    code = f"{CODE_PREFIX}metadata-contract-invalid"


class RegistryRefusal(TranslationRefusal):
    """A backend or source language the registries do not know, or a conflicting registration."""

    code = f"{CODE_PREFIX}registry-refused"


# Emitters: CrossGL to a target language.


class TargetRefusal(TranslationRefusal):
    """A construct refused by a helper that several emitters share; the message names the target."""

    code = f"{CODE_PREFIX}target-refused"


class DirectXRefusal(TranslationRefusal):
    """A construct the HLSL emitter refuses."""

    code = f"{CODE_PREFIX}directx-refused"
    default_target = "directx"


class OpenGLRefusal(TranslationRefusal):
    """A construct the GLSL emitter refuses."""

    code = f"{CODE_PREFIX}opengl-refused"
    default_target = "opengl"


class MetalRefusal(TranslationRefusal):
    """A construct the Metal Shading Language emitter refuses."""

    code = f"{CODE_PREFIX}metal-refused"
    default_target = "metal"


class VulkanRefusal(TranslationRefusal):
    """A construct the SPIR-V emitter refuses."""

    code = f"{CODE_PREFIX}vulkan-refused"
    default_target = "vulkan"


class WGSLRefusal(TranslationRefusal):
    """A construct the WGSL emitter refuses."""

    code = f"{CODE_PREFIX}wgsl-refused"
    default_target = "wgsl"


class SlangRefusal(TranslationRefusal):
    """A construct the Slang emitter refuses."""

    code = f"{CODE_PREFIX}slang-refused"
    default_target = "slang"


class CudaRefusal(TranslationRefusal):
    """A construct the CUDA emitter refuses."""

    code = f"{CODE_PREFIX}cuda-refused"
    default_target = "cuda"


class HipRefusal(TranslationRefusal):
    """A construct the HIP emitter refuses."""

    code = f"{CODE_PREFIX}hip-refused"
    default_target = "hip"


class MojoRefusal(TranslationRefusal):
    """A construct the Mojo emitter refuses."""

    code = f"{CODE_PREFIX}mojo-refused"
    default_target = "mojo"


class RustRefusal(TranslationRefusal):
    """A construct the Rust emitter refuses."""

    code = f"{CODE_PREFIX}rust-refused"
    default_target = "rust"


# Front ends: a source language to CrossGL.


class DirectXSyntaxError(SourceSyntaxError):
    """HLSL source the HLSL lexer, preprocessor or parser cannot read."""

    code = f"{CODE_PREFIX}directx-syntax-error"
    source_name = "directx"


class DirectXSourceRefusal(TranslationRefusal):
    """An HLSL construct the HLSL importer refuses to carry into CrossGL."""

    code = f"{CODE_PREFIX}directx-source-refused"
    default_target = "cgl"


class OpenGLSyntaxError(SourceSyntaxError):
    """GLSL source the GLSL lexer, preprocessor or parser cannot read."""

    code = f"{CODE_PREFIX}opengl-syntax-error"
    source_name = "opengl"


class OpenGLSourceRefusal(TranslationRefusal):
    """A GLSL construct the GLSL importer refuses to carry into CrossGL."""

    code = f"{CODE_PREFIX}opengl-source-refused"
    default_target = "cgl"


class MetalSyntaxError(SourceSyntaxError):
    """Metal Shading Language source the Metal lexer, preprocessor or parser cannot read."""

    code = f"{CODE_PREFIX}metal-syntax-error"
    source_name = "metal"


class MetalSourceRefusal(TranslationRefusal):
    """A Metal construct the Metal importer refuses to carry into CrossGL."""

    code = f"{CODE_PREFIX}metal-source-refused"
    default_target = "cgl"


class SpirvSyntaxError(SourceSyntaxError):
    """SPIR-V assembly the SPIR-V lexer or parser cannot read."""

    code = f"{CODE_PREFIX}spirv-syntax-error"
    source_name = "vulkan"


class CudaSyntaxError(SourceSyntaxError):
    """CUDA C++ source the CUDA lexer or parser cannot read."""

    code = f"{CODE_PREFIX}cuda-syntax-error"
    source_name = "cuda"


class HipSyntaxError(SourceSyntaxError):
    """HIP C++ source the HIP lexer or parser cannot read."""

    code = f"{CODE_PREFIX}hip-syntax-error"
    source_name = "hip"


class HipSourceRefusal(TranslationRefusal):
    """A HIP construct the HIP importer refuses to carry into CrossGL."""

    code = f"{CODE_PREFIX}hip-source-refused"
    default_target = "cgl"


class SlangSyntaxError(SourceSyntaxError):
    """Slang source the Slang lexer or parser cannot read."""

    code = f"{CODE_PREFIX}slang-syntax-error"
    source_name = "slang"


class MojoSyntaxError(SourceSyntaxError):
    """Mojo source the Mojo lexer or parser cannot read."""

    code = f"{CODE_PREFIX}mojo-syntax-error"
    source_name = "mojo"


class RustSyntaxError(SourceSyntaxError):
    """Rust source the Rust lexer or parser cannot read."""

    code = f"{CODE_PREFIX}rust-syntax-error"
    source_name = "rust"


class OpenCLSyntaxError(SourceSyntaxError):
    """OpenCL C source the OpenCL lexer cannot read."""

    code = f"{CODE_PREFIX}opencl-syntax-error"
    source_name = "opencl"
