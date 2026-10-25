"""GLSL extension types that CrossGL cannot represent.

GL_ARM_tensors declares ``uniform tensorARM<T, rank>`` resources that ``tensorReadARM``, ``tensorWriteARM`` and
``tensorSizeARM`` access (GLSL_ARM_tensors, "Tensor Types"). CrossGL has no tensor resource, and HLSL has no
counterpart; the importer used to drop the declaration, so every target named an undeclared identifier.
:func:`refuse_unsupported_glsl_types` raises :class:`GLSLTypeUnsupportedError` for such a declaration instead.
"""

from __future__ import annotations

from crosstl.translator.diagnostics import CODE_PREFIX, TranslationDiagnostic

__all__ = ["GLSLTypeUnsupportedError", "refuse_unsupported_glsl_types"]

# Type name -> (extension, what it declares).
_UNSUPPORTED = {"tensorARM": ("GL_ARM_tensors", "a tensor resource")}


class GLSLTypeUnsupportedError(TranslationDiagnostic):
    """A GLSL declaration of an extension type with no CrossGL representation."""

    def __init__(self, name, type_name):
        extension, what = _UNSUPPORTED[type_name]
        self.type_name = type_name
        self.missing_capabilities = (f"opengl.import.{type_name}",)
        super().__init__(
            f"'{name}' is {what} ({type_name}, {extension}), which CrossGL cannot represent",
            target="cgl",
            code=f"{CODE_PREFIX}opengl-import-type-unsupported",
        )


def _base(type_text):
    return str(type_text or "").split("<", 1)[0].strip()


def refuse_unsupported_glsl_types(shader):
    """Raise :class:`GLSLTypeUnsupportedError` for the first declaration of an unsupported extension type."""
    for group in ("uniforms", "global_variables"):
        for declaration in getattr(shader, group, None) or []:
            base = _base(getattr(declaration, "vtype", None))
            if base in _UNSUPPORTED:
                raise GLSLTypeUnsupportedError(getattr(declaration, "name", "?"), base)
