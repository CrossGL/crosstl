"""Public registry helpers for CrossGL target code generation."""

from .cuda_codegen import CudaCodeGen
from .directx_codegen import HLSLCodeGen
from .GLSL_codegen import GLSLCodeGen
from .hip_codegen import HipCodeGen
from .metal_codegen import MetalCodeGen
from .mojo_codegen import MojoCodeGen
from .registry import (
    BackendSpec,
    backend_names,
    get_backend,
    get_backend_extension,
    get_codegen,
    normalize_backend_name,
    register_backend,
    source_backend_names,
    target_backend_names_with_source_frontends,
    target_profiles,
)
from .rust_codegen import RustCodeGen
from .SPIRV_codegen import VulkanSPIRVCodeGen
from .webgl_codegen import WebGLCodeGen
from .wgsl_codegen import WGSLCodeGen

register_backend(
    BackendSpec(
        name="cuda",
        codegen_class=CudaCodeGen,
        aliases=("cu",),
        file_extensions=(".cu", ".cuh", ".cuda"),
        format_backend="cuda",
    )
)
register_backend(
    BackendSpec(
        name="directx",
        codegen_class=HLSLCodeGen,
        aliases=("hlsl", "dx"),
        target_aliases=("dx11", "dx12", "d3d11", "d3d12"),
        target_profiles=("directx-11", "directx-12"),
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
        aliases=("msl",),
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
        file_extensions=(".rs", ".rust"),
        format_backend="rust",
    )
)
register_backend(
    BackendSpec(
        name="vulkan",
        codegen_class=VulkanSPIRVCodeGen,
        aliases=("spirv", "spv"),
        target_profiles=("vulkan-khr-cooperative-matrix",),
        file_extensions=(".spvasm",),
        format_backend="vulkan",
    )
)
register_backend(
    BackendSpec(
        name="webgl",
        codegen_class=WebGLCodeGen,
        aliases=("webgl2", "essl", "glsl-es"),
        file_extensions=(".webgl.glsl",),
        format_backend="opengl",
        has_source_frontend=False,
    )
)
register_backend(
    BackendSpec(
        name="wgsl",
        codegen_class=WGSLCodeGen,
        aliases=("webgpu",),
        file_extensions=(".wgsl",),
        format_backend="wgsl",
        has_source_frontend=False,
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
    "BackendSpec",
    "CudaCodeGen",
    "HLSLCodeGen",
    "GLSLCodeGen",
    "HipCodeGen",
    "MetalCodeGen",
    "MojoCodeGen",
    "RustCodeGen",
    "VulkanSPIRVCodeGen",
    "WebGLCodeGen",
    "WGSLCodeGen",
    "get_backend",
    "get_codegen",
    "get_backend_extension",
    "normalize_backend_name",
    "register_backend",
    "backend_names",
    "source_backend_names",
    "target_backend_names_with_source_frontends",
    "target_profiles",
]

if "_SlangCodeGen" in globals() and _SlangCodeGen is not None:
    __all__.append("SlangCodeGen")
