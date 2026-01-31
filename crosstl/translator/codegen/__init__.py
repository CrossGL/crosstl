"""
Code Generation Module.

This module provides code generators for translating CrossGL AST to various
target shader languages. It includes the backend registry system and individual
code generator implementations.

Supported backends:
    - CUDA: NVIDIA CUDA code generation
    - DirectX/HLSL: Microsoft HLSL code generation
    - OpenGL/GLSL: OpenGL GLSL code generation
    - HIP: AMD HIP code generation
    - Metal: Apple Metal Shading Language code generation
    - Mojo: Mojo code generation
    - Rust: Rust GPU code generation
    - Vulkan/SPIR-V: Vulkan SPIR-V assembly code generation
    - Slang: Slang shading language code generation (optional)

Example:
    >>> from crosstl.translator.codegen import get_codegen
    >>> metal_gen = get_codegen("metal")
    >>> metal_code = metal_gen.generate(ast)
"""

from .registry import (
    BackendSpec,
    register_backend,
    get_backend,
    get_codegen,
    get_backend_extension,
    normalize_backend_name,
    backend_names,
)
from .cuda_codegen import CudaCodeGen
from .directx_codegen import HLSLCodeGen
from .GLSL_codegen import GLSLCodeGen
from .hip_codegen import HipCodeGen
from .metal_codegen import MetalCodeGen
from .mojo_codegen import MojoCodeGen
from .rust_codegen import RustCodeGen
from .SPIRV_codegen import VulkanSPIRVCodeGen

register_backend(
    BackendSpec(
        name="cuda",
        codegen_class=CudaCodeGen,
        aliases=("cu",),
        file_extensions=(".cu",),
        format_backend="cuda",
    )
)
register_backend(
    BackendSpec(
        name="directx",
        codegen_class=HLSLCodeGen,
        aliases=("hlsl", "dx"),
        file_extensions=(".hlsl",),
        format_backend="directx",
    )
)
register_backend(
    BackendSpec(
        name="opengl",
        codegen_class=GLSLCodeGen,
        aliases=("glsl", "ogl"),
        file_extensions=(".glsl",),
        format_backend="opengl",
    )
)
register_backend(
    BackendSpec(
        name="hip",
        codegen_class=HipCodeGen,
        aliases=("hip",),
        file_extensions=(".hip",),
        format_backend="hip",
    )
)
register_backend(
    BackendSpec(
        name="metal",
        codegen_class=MetalCodeGen,
        aliases=("metal",),
        file_extensions=(".metal",),
        format_backend="metal",
    )
)
register_backend(
    BackendSpec(
        name="mojo",
        codegen_class=MojoCodeGen,
        aliases=("mojo",),
        file_extensions=(".mojo",),
        format_backend="mojo",
    )
)
register_backend(
    BackendSpec(
        name="rust",
        codegen_class=RustCodeGen,
        aliases=("rust", "rs"),
        file_extensions=(".rs",),
        format_backend="rust",
    )
)
register_backend(
    BackendSpec(
        name="vulkan",
        codegen_class=VulkanSPIRVCodeGen,
        aliases=("spirv", "spv"),
        file_extensions=(".spirv",),
        format_backend="vulkan",
    )
)

# Import slang_codegen only if available
try:
    from .slang_codegen import SlangCodeGen as _SlangCodeGen
except ImportError:
    _SlangCodeGen = None

if _SlangCodeGen is not None:
    SlangCodeGen = _SlangCodeGen
    register_backend(
        BackendSpec(
            name="slang",
            codegen_class=SlangCodeGen,
            aliases=("slang",),
            file_extensions=(".slang",),
            format_backend="slang",
        )
    )

__all__ = [
    "CudaCodeGen",
    "HLSLCodeGen",
    "GLSLCodeGen",
    "HipCodeGen",
    "MetalCodeGen",
    "MojoCodeGen",
    "RustCodeGen",
    "VulkanSPIRVCodeGen",
    "get_backend",
    "get_codegen",
    "get_backend_extension",
    "normalize_backend_name",
    "backend_names",
]

if "_SlangCodeGen" in globals() and _SlangCodeGen is not None:
    __all__.append("SlangCodeGen")
