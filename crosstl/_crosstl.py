"""High-level translation API and command-line entry point for CrossGL Translator."""

import argparse
import os
import sys
from typing import Optional

from .translator.codegen import (
    backend_names,
    get_backend_extension,
    get_codegen,
    normalize_backend_name,
)
from .translator.plugin_loader import discover_backend_plugins
from .translator.source_registry import (
    BINARY_SPIRV_UNSUPPORTED_MESSAGE,
    SOURCE_REGISTRY,
    register_default_sources,
)

try:
    from .formatter import format_shader_code

    FORMATTER_AVAILABLE = True
except ImportError:
    FORMATTER_AVAILABLE = False


SPIRV_BINARY_MAGIC_PREFIXES = (b"\x03\x02\x23\x07", b"\x07\x23\x02\x03")


def _read_shader_source(file_path: str, source_name: str) -> str:
    with open(file_path, "rb") as file:
        shader_bytes = file.read()

    if source_name == "vulkan" and shader_bytes.startswith(SPIRV_BINARY_MAGIC_PREFIXES):
        raise ValueError(BINARY_SPIRV_UNSUPPORTED_MESSAGE)

    return shader_bytes.decode("utf-8", errors="replace")


def translate(
    file_path: str,
    backend: str = "cgl",
    save_shader: Optional[str] = None,
    format_output: bool = True,
) -> str:
    """Translate a shader file to another language.

    Args:
        file_path (str): The path to the shader file
        backend (str, optional): The target language to translate to. Defaults to "cgl".
        save_shader (str, optional): The path to save the translated shader. Defaults to None.
        format_output (bool, optional): Whether to format the generated code. Defaults to True.

    Returns:
        str: The translated shader code
    """
    register_default_sources()
    discover_backend_plugins()
    backend = (backend or "cgl").strip().lower()

    source_spec = SOURCE_REGISTRY.get_by_extension(file_path)
    if not source_spec:
        supported = ", ".join(SOURCE_REGISTRY.extensions())
        raise ValueError(
            f"Unsupported shader file type: {file_path}. Supported: {supported}"
        )

    shader_code = _read_shader_source(file_path, source_spec.name)

    ast = source_spec.parse(shader_code, file_path=file_path)

    requested_backend = backend
    normalized_backend = normalize_backend_name(requested_backend) or requested_backend

    if source_spec.name == "cgl":
        if normalized_backend in ["cgl", "crossgl"]:
            generated_code = shader_code
        else:
            codegen = get_codegen(normalized_backend)
            generated_code = codegen.generate(ast)
    else:
        if normalized_backend in ["cgl", "crossgl"]:
            if not source_spec.reverse_codegen_factory:
                raise ValueError(f"Reverse translation not supported for: {file_path}")
            codegen = source_spec.reverse_codegen_factory()
            generated_code = codegen.generate(ast)
        else:
            if not source_spec.reverse_codegen_factory:
                raise ValueError(
                    f"Unsupported translation scenario: {file_path} to {backend}"
                )
            # Translate to CrossGL first, then to target backend
            reverse_codegen = source_spec.reverse_codegen_factory()
            intermediate_code = reverse_codegen.generate(ast)
            cgl_spec = SOURCE_REGISTRY.get("cgl")
            if not cgl_spec:
                raise ValueError("CrossGL parser not available for intermediate step")
            cgl_ast = cgl_spec.parse(intermediate_code)
            codegen = get_codegen(normalized_backend)
            generated_code = codegen.generate(cgl_ast)

    if (
        format_output
        and FORMATTER_AVAILABLE
        and normalized_backend not in ["cgl", "crossgl"]
    ):
        generated_code = format_shader_code(
            generated_code, normalized_backend, save_shader
        )

    if save_shader is not None:
        with open(save_shader, "w", encoding="utf-8") as file:
            file.write(generated_code)

    return generated_code


def main():
    """Command-line entry point for CrossGL translation."""
    parser = argparse.ArgumentParser(description="CrossGL Shader Translator")

    parser.add_argument("input", help="Input shader file path")
    supported_backends = ", ".join(backend_names() + ["cgl"])
    parser.add_argument(
        "--backend",
        "-b",
        default="cgl",
        help=f"Target backend ({supported_backends})",
    )
    parser.add_argument("--output", "-o", help="Output file path")
    parser.add_argument(
        "--no-format", action="store_true", help="Disable code formatting"
    )

    args = parser.parse_args()

    try:
        if not os.path.exists(args.input):
            print(f"Error: Input file {args.input} not found")
            return 1

        output_path = args.output
        if not output_path:
            base, _ = os.path.splitext(args.input)
            normalized_backend = normalize_backend_name(args.backend) or args.backend
            if normalized_backend in ["cgl", "crossgl"]:
                ext = ".cgl"
            else:
                ext = get_backend_extension(normalized_backend) or ".out"
            output_path = base + ext

        translate(
            args.input,
            backend=args.backend,
            save_shader=output_path,
            format_output=not args.no_format,
        )

        print(f"Successfully translated to {output_path}")
        return 0

    except Exception as e:
        print(f"Error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
