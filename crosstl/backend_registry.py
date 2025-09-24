"""
Backend Registry for CrossTL.
Centralized registration and management of all language backends.
"""

from typing import Dict, List, Type, Optional
from .backend.base_lexer import BaseLexer
from .backend.base_parser import BaseParser
from .backend.base_codegen import CrossGLToCrossGLConverter, CrossGLToTargetConverter


class BackendInfo:
    """Information about a language backend."""

    def __init__(
        self,
        name: str,
        file_extensions: List[str],
        lexer_class: Type[BaseLexer],
        parser_class: Type[BaseParser],
        to_crossgl_class: Type[CrossGLToCrossGLConverter],
        from_crossgl_class: Type[CrossGLToTargetConverter],
        description: str = "",
    ):
        self.name = name
        self.file_extensions = file_extensions
        self.lexer_class = lexer_class
        self.parser_class = parser_class
        self.to_crossgl_class = to_crossgl_class
        self.from_crossgl_class = from_crossgl_class
        self.description = description

    def create_lexer(self, code: str) -> BaseLexer:
        """Create a lexer instance for this backend."""
        return self.lexer_class(code)

    def create_parser(self, tokens) -> BaseParser:
        """Create a parser instance for this backend."""
        return self.parser_class(tokens)

    def create_to_crossgl_converter(self) -> CrossGLToCrossGLConverter:
        """Create a to-CrossGL converter for this backend."""
        return self.to_crossgl_class()

    def create_from_crossgl_converter(self) -> CrossGLToTargetConverter:
        """Create a from-CrossGL converter for this backend."""
        return self.from_crossgl_class()


class BackendRegistry:
    """Central registry for all CrossTL backends."""

    _backends: Dict[str, BackendInfo] = {}
    _extension_map: Dict[str, str] = {}

    @classmethod
    def register_backend(cls, backend_info: BackendInfo):
        """Register a new backend."""
        cls._backends[backend_info.name.lower()] = backend_info

        # Update extension mapping
        for ext in backend_info.file_extensions:
            cls._extension_map[ext.lower()] = backend_info.name.lower()

    @classmethod
    def get_backend(cls, name: str) -> Optional[BackendInfo]:
        """Get backend info by name."""
        return cls._backends.get(name.lower())

    @classmethod
    def get_backend_by_extension(cls, file_extension: str) -> Optional[BackendInfo]:
        """Get backend info by file extension."""
        if file_extension.startswith("."):
            file_extension = file_extension[1:]

        backend_name = cls._extension_map.get(file_extension.lower())
        if backend_name:
            return cls._backends.get(backend_name)
        return None

    @classmethod
    def get_backend_by_file(cls, file_path: str) -> Optional[BackendInfo]:
        """Get backend info by file path."""
        ext = file_path.split(".")[-1] if "." in file_path else ""
        return cls.get_backend_by_extension(ext)

    @classmethod
    def get_all_backends(cls) -> Dict[str, BackendInfo]:
        """Get all registered backends."""
        return cls._backends.copy()

    @classmethod
    def get_supported_languages(cls) -> List[str]:
        """Get list of supported language names."""
        return list(cls._backends.keys())

    @classmethod
    def get_supported_extensions(cls) -> List[str]:
        """Get list of supported file extensions."""
        return list(cls._extension_map.keys())

    @classmethod
    def is_supported_language(cls, name: str) -> bool:
        """Check if a language is supported."""
        return name.lower() in cls._backends

    @classmethod
    def is_supported_extension(cls, extension: str) -> bool:
        """Check if a file extension is supported."""
        if extension.startswith("."):
            extension = extension[1:]
        return extension.lower() in cls._extension_map


# Now let's register the existing backends
def initialize_builtin_backends():
    """Initialize all built-in backends."""

    # Import existing backend classes
    try:
        # CUDA
        from .backend.CUDA.CudaLexer import CudaLexer as CudaLexerOld
        from .backend.CUDA.CudaParser import CudaParser
        from .backend.CUDA.CudaCrossGLCodeGen import CudaToCrossGLConverter as CudaToGL
        from .translator.codegen.cuda_codegen import CudaCodeGen

        # Wrap old classes to match new interface
        class CudaLexerWrapper(BaseLexer):
            def __init__(self, code: str):
                self.old_lexer = CudaLexerOld(code)
                super().__init__(code)

            def get_token_patterns(self):
                return []

            def get_keywords(self):
                return {}

            def tokenize(self):
                old_tokens = self.old_lexer.tokenize()
                return [Token(TokenType.IDENTIFIER, text) for _, text in old_tokens]

        class CudaParserWrapper(BaseParser):
            def __init__(self, tokens):
                # Convert to old format if needed
                if tokens and hasattr(tokens[0], "type"):
                    old_tokens = [(token.type.name, token.value) for token in tokens]
                else:
                    old_tokens = tokens
                self.old_parser = CudaParser(old_tokens)
                super().__init__(tokens if hasattr(tokens[0], "type") else [])

            def parse(self):
                return self.old_parser.parse()

            def parse_statement(self):
                return None

            def parse_expression(self):
                return None

            def parse_type(self):
                return ""

            def parse_parameter(self):
                return None

            def parse_unary_expression(self):
                return None

        class CudaFromCrossGLWrapper(CrossGLToTargetConverter):
            def __init__(self):
                super().__init__("cuda")
                self.codegen = CudaCodeGen()

            def generate(self, ast_node):
                return self.codegen.generate(ast_node)

        cuda_backend = BackendInfo(
            name="cuda",
            file_extensions=["cu", "cuh", "cuda"],
            lexer_class=CudaLexerWrapper,
            parser_class=CudaParserWrapper,
            to_crossgl_class=CudaToGL,
            from_crossgl_class=CudaFromCrossGLWrapper,
            description="NVIDIA CUDA backend for GPU computing",
        )

        BackendRegistry.register_backend(cuda_backend)

    except ImportError as e:
        print(f"Warning: Could not register CUDA backend: {e}")

    # Register other backends similarly...
    _register_metal_backend()
    _register_directx_backend()
    _register_opengl_backend()
    _register_vulkan_backend()
    _register_rust_backend()
    _register_mojo_backend()
    _register_hip_backend()
    _register_slang_backend()


def _register_metal_backend():
    """Register Metal backend."""
    try:
        from .backend.Metal.MetalLexer import MetalLexer
        from .backend.Metal.MetalParser import MetalParser
        from .backend.Metal.MetalCrossGLCodeGen import MetalToCrossGLConverter
        from .translator.codegen.metal_codegen import MetalCodeGen

        class MetalLexerWrapper(BaseLexer):
            def __init__(self, code: str):
                self.old_lexer = MetalLexer(code)
                super().__init__(code)

            def get_token_patterns(self):
                return []

            def get_keywords(self):
                return {}

            def tokenize(self):
                old_tokens = self.old_lexer.tokenize()
                return [Token(TokenType.IDENTIFIER, text) for _, text in old_tokens]

        class MetalParserWrapper(BaseParser):
            def __init__(self, tokens):
                if tokens and hasattr(tokens[0], "type"):
                    old_tokens = [(token.type.name, token.value) for token in tokens]
                else:
                    old_tokens = tokens
                self.old_parser = MetalParser(old_tokens)
                super().__init__(tokens if hasattr(tokens[0], "type") else [])

            def parse(self):
                return self.old_parser.parse()

            def parse_statement(self):
                return None

            def parse_expression(self):
                return None

            def parse_type(self):
                return ""

            def parse_parameter(self):
                return None

            def parse_unary_expression(self):
                return None

        class MetalFromCrossGLWrapper(CrossGLToTargetConverter):
            def __init__(self):
                super().__init__("metal")
                self.codegen = MetalCodeGen()

            def generate(self, ast_node):
                return self.codegen.generate(ast_node)

        metal_backend = BackendInfo(
            name="metal",
            file_extensions=["metal"],
            lexer_class=MetalLexerWrapper,
            parser_class=MetalParserWrapper,
            to_crossgl_class=MetalToCrossGLConverter,
            from_crossgl_class=MetalFromCrossGLWrapper,
            description="Apple Metal backend for GPU programming",
        )

        BackendRegistry.register_backend(metal_backend)

    except ImportError as e:
        print(f"Warning: Could not register Metal backend: {e}")


def _register_directx_backend():
    """Register DirectX backend."""
    try:
        from .backend.DirectX.DirectxLexer import HLSLLexer
        from .backend.DirectX.DirectxParser import HLSLParser
        from .backend.DirectX.DirectxCrossGLCodeGen import HLSLToCrossGLConverter
        from .translator.codegen.directx_codegen import HLSLCodeGen

        class DirectXLexerWrapper(BaseLexer):
            def __init__(self, code: str):
                self.old_lexer = HLSLLexer(code)
                super().__init__(code)

            def get_token_patterns(self):
                return []

            def get_keywords(self):
                return {}

            def tokenize(self):
                old_tokens = self.old_lexer.tokenize()
                return [Token(TokenType.IDENTIFIER, text) for _, text in old_tokens]

        class DirectXParserWrapper(BaseParser):
            def __init__(self, tokens):
                if tokens and hasattr(tokens[0], "type"):
                    old_tokens = [(token.type.name, token.value) for token in tokens]
                else:
                    old_tokens = tokens
                self.old_parser = HLSLParser(old_tokens)
                super().__init__(tokens if hasattr(tokens[0], "type") else [])

            def parse(self):
                return self.old_parser.parse()

            def parse_statement(self):
                return None

            def parse_expression(self):
                return None

            def parse_type(self):
                return ""

            def parse_parameter(self):
                return None

            def parse_unary_expression(self):
                return None

        class DirectXFromCrossGLWrapper(CrossGLToTargetConverter):
            def __init__(self):
                super().__init__("directx")
                self.codegen = HLSLCodeGen()

            def generate(self, ast_node):
                return self.codegen.generate(ast_node)

        directx_backend = BackendInfo(
            name="directx",
            file_extensions=["hlsl"],
            lexer_class=DirectXLexerWrapper,
            parser_class=DirectXParserWrapper,
            to_crossgl_class=HLSLToCrossGLConverter,
            from_crossgl_class=DirectXFromCrossGLWrapper,
            description="Microsoft DirectX HLSL backend",
        )

        BackendRegistry.register_backend(directx_backend)

    except ImportError as e:
        print(f"Warning: Could not register DirectX backend: {e}")


def _register_opengl_backend():
    """Register OpenGL backend."""
    try:
        from .backend.GLSL.OpenglLexer import GLSLLexer
        from .backend.GLSL.OpenglParser import GLSLParser
        from .backend.GLSL.openglCrossglCodegen import GLSLToCrossGLConverter
        from .translator.codegen.GLSL_codegen import GLSLCodeGen

        class OpenGLLexerWrapper(BaseLexer):
            def __init__(self, code: str):
                self.old_lexer = GLSLLexer(code)
                super().__init__(code)

            def get_token_patterns(self):
                return []

            def get_keywords(self):
                return {}

            def tokenize(self):
                old_tokens = self.old_lexer.tokenize()
                return [Token(TokenType.IDENTIFIER, text) for _, text in old_tokens]

        class OpenGLParserWrapper(BaseParser):
            def __init__(self, tokens):
                if tokens and hasattr(tokens[0], "type"):
                    old_tokens = [(token.type.name, token.value) for token in tokens]
                else:
                    old_tokens = tokens
                self.old_parser = GLSLParser(old_tokens)
                super().__init__(tokens if hasattr(tokens[0], "type") else [])

            def parse(self):
                return self.old_parser.parse()

            def parse_statement(self):
                return None

            def parse_expression(self):
                return None

            def parse_type(self):
                return ""

            def parse_parameter(self):
                return None

            def parse_unary_expression(self):
                return None

        class OpenGLFromCrossGLWrapper(CrossGLToTargetConverter):
            def __init__(self):
                super().__init__("opengl")
                self.codegen = GLSLCodeGen()

            def generate(self, ast_node):
                return self.codegen.generate(ast_node)

        opengl_backend = BackendInfo(
            name="opengl",
            file_extensions=["glsl", "vert", "frag", "geom", "comp"],
            lexer_class=OpenGLLexerWrapper,
            parser_class=OpenGLParserWrapper,
            to_crossgl_class=GLSLToCrossGLConverter,
            from_crossgl_class=OpenGLFromCrossGLWrapper,
            description="OpenGL GLSL backend for graphics programming",
        )

        BackendRegistry.register_backend(opengl_backend)

    except ImportError as e:
        print(f"Warning: Could not register OpenGL backend: {e}")


def _register_vulkan_backend():
    """Register Vulkan backend."""
    try:
        from .backend.SPIRV.VulkanLexer import VulkanLexer
        from .backend.SPIRV.VulkanParser import VulkanParser
        from .backend.SPIRV.VulkanCrossGLCodeGen import VulkanToCrossGLConverter
        from .translator.codegen.SPIRV_codegen import VulkanSPIRVCodeGen

        class VulkanLexerWrapper(BaseLexer):
            def __init__(self, code: str):
                self.old_lexer = VulkanLexer(code)
                super().__init__(code)

            def get_token_patterns(self):
                return []

            def get_keywords(self):
                return {}

            def tokenize(self):
                old_tokens = self.old_lexer.tokenize()
                return [Token(TokenType.IDENTIFIER, text) for _, text in old_tokens]

        class VulkanParserWrapper(BaseParser):
            def __init__(self, tokens):
                if tokens and hasattr(tokens[0], "type"):
                    old_tokens = [(token.type.name, token.value) for token in tokens]
                else:
                    old_tokens = tokens
                self.old_parser = VulkanParser(old_tokens)
                super().__init__(tokens if hasattr(tokens[0], "type") else [])

            def parse(self):
                return self.old_parser.parse()

            def parse_statement(self):
                return None

            def parse_expression(self):
                return None

            def parse_type(self):
                return ""

            def parse_parameter(self):
                return None

            def parse_unary_expression(self):
                return None

        class VulkanFromCrossGLWrapper(CrossGLToTargetConverter):
            def __init__(self):
                super().__init__("vulkan")
                self.codegen = VulkanSPIRVCodeGen()

            def generate(self, ast_node):
                return self.codegen.generate(ast_node)

        vulkan_backend = BackendInfo(
            name="vulkan",
            file_extensions=["spv", "spirv"],
            lexer_class=VulkanLexerWrapper,
            parser_class=VulkanParserWrapper,
            to_crossgl_class=VulkanToCrossGLConverter,
            from_crossgl_class=VulkanFromCrossGLWrapper,
            description="Vulkan SPIR-V backend for graphics and compute",
        )

        BackendRegistry.register_backend(vulkan_backend)

    except ImportError as e:
        print(f"Warning: Could not register Vulkan backend: {e}")


def _register_rust_backend():
    """Register Rust backend."""
    try:
        from .backend.Rust.RustLexer import RustLexer
        from .backend.Rust.RustParser import RustParser
        from .backend.Rust.RustCrossGLCodeGen import RustToCrossGLConverter
        from .translator.codegen.rust_codegen import RustCodeGen

        class RustLexerWrapper(BaseLexer):
            def __init__(self, code: str):
                self.old_lexer = RustLexer(code)
                super().__init__(code)

            def get_token_patterns(self):
                return []

            def get_keywords(self):
                return {}

            def tokenize(self):
                old_tokens = self.old_lexer.tokenize()
                return [Token(TokenType.IDENTIFIER, text) for _, text in old_tokens]

        class RustParserWrapper(BaseParser):
            def __init__(self, tokens):
                if tokens and hasattr(tokens[0], "type"):
                    old_tokens = [(token.type.name, token.value) for token in tokens]
                else:
                    old_tokens = tokens
                self.old_parser = RustParser(old_tokens)
                super().__init__(tokens if hasattr(tokens[0], "type") else [])

            def parse(self):
                return self.old_parser.parse()

            def parse_statement(self):
                return None

            def parse_expression(self):
                return None

            def parse_type(self):
                return ""

            def parse_parameter(self):
                return None

            def parse_unary_expression(self):
                return None

        class RustFromCrossGLWrapper(CrossGLToTargetConverter):
            def __init__(self):
                super().__init__("rust")
                self.codegen = RustCodeGen()

            def generate(self, ast_node):
                return self.codegen.generate(ast_node)

        rust_backend = BackendInfo(
            name="rust",
            file_extensions=["rs", "rust"],
            lexer_class=RustLexerWrapper,
            parser_class=RustParserWrapper,
            to_crossgl_class=RustToCrossGLConverter,
            from_crossgl_class=RustFromCrossGLWrapper,
            description="Rust backend for systems programming",
        )

        BackendRegistry.register_backend(rust_backend)

    except ImportError as e:
        print(f"Warning: Could not register Rust backend: {e}")


def _register_mojo_backend():
    """Register Mojo backend."""
    try:
        from .backend.Mojo.MojoLexer import MojoLexer
        from .backend.Mojo.MojoParser import MojoParser
        from .backend.Mojo.MojoCrossGLCodeGen import MojoToCrossGLConverter
        from .translator.codegen.mojo_codegen import MojoCodeGen

        class MojoLexerWrapper(BaseLexer):
            def __init__(self, code: str):
                self.old_lexer = MojoLexer(code)
                super().__init__(code)

            def get_token_patterns(self):
                return []

            def get_keywords(self):
                return {}

            def tokenize(self):
                old_tokens = self.old_lexer.tokenize()
                return [Token(TokenType.IDENTIFIER, text) for _, text in old_tokens]

        class MojoParserWrapper(BaseParser):
            def __init__(self, tokens):
                if tokens and hasattr(tokens[0], "type"):
                    old_tokens = [(token.type.name, token.value) for token in tokens]
                else:
                    old_tokens = tokens
                self.old_parser = MojoParser(old_tokens)
                super().__init__(tokens if hasattr(tokens[0], "type") else [])

            def parse(self):
                return self.old_parser.parse()

            def parse_statement(self):
                return None

            def parse_expression(self):
                return None

            def parse_type(self):
                return ""

            def parse_parameter(self):
                return None

            def parse_unary_expression(self):
                return None

        class MojoFromCrossGLWrapper(CrossGLToTargetConverter):
            def __init__(self):
                super().__init__("mojo")
                self.codegen = MojoCodeGen()

            def generate(self, ast_node):
                return self.codegen.generate(ast_node)

        mojo_backend = BackendInfo(
            name="mojo",
            file_extensions=["mojo"],
            lexer_class=MojoLexerWrapper,
            parser_class=MojoParserWrapper,
            to_crossgl_class=MojoToCrossGLConverter,
            from_crossgl_class=MojoFromCrossGLWrapper,
            description="Mojo backend for AI and systems programming",
        )

        BackendRegistry.register_backend(mojo_backend)

    except ImportError as e:
        print(f"Warning: Could not register Mojo backend: {e}")


def _register_hip_backend():
    """Register HIP backend."""
    try:
        from .backend.HIP.HipLexer import HipLexer
        from .backend.HIP.HipParser import HipParser
        from .backend.HIP.HipCrossGLCodeGen import HipToCrossGLConverter
        from .translator.codegen.hip_codegen import HipCodeGen

        class HipLexerWrapper(BaseLexer):
            def __init__(self, code: str):
                self.old_lexer = HipLexer(code)
                super().__init__(code)

            def get_token_patterns(self):
                return []

            def get_keywords(self):
                return {}

            def tokenize(self):
                old_tokens = self.old_lexer.tokenize()
                return [
                    Token(
                        TokenType.IDENTIFIER,
                        text.value if hasattr(text, "value") else text,
                    )
                    for text in old_tokens
                ]

        class HipParserWrapper(BaseParser):
            def __init__(self, tokens):
                # HipParser expects Token objects, not tuples
                if tokens and hasattr(tokens[0], "type"):
                    # Already Token objects
                    old_tokens = tokens
                else:
                    # Convert tuples to Token objects
                    old_tokens = [
                        Token(type_name, value) for type_name, value in tokens
                    ]
                self.old_parser = HipParser(old_tokens)
                super().__init__(tokens if hasattr(tokens[0], "type") else [])

            def parse(self):
                return self.old_parser.parse()

            def parse_statement(self):
                return None

            def parse_expression(self):
                return None

            def parse_type(self):
                return ""

            def parse_parameter(self):
                return None

            def parse_unary_expression(self):
                return None

        class HipFromCrossGLWrapper(CrossGLToTargetConverter):
            def __init__(self):
                super().__init__("hip")
                self.codegen = HipCodeGen()

            def generate(self, ast_node):
                return self.codegen.generate(ast_node)

        hip_backend = BackendInfo(
            name="hip",
            file_extensions=["hip"],
            lexer_class=HipLexerWrapper,
            parser_class=HipParserWrapper,
            to_crossgl_class=HipToCrossGLConverter,
            from_crossgl_class=HipFromCrossGLWrapper,
            description="AMD HIP backend for GPU computing",
        )

        BackendRegistry.register_backend(hip_backend)

    except ImportError as e:
        print(f"Warning: Could not register HIP backend: {e}")


def _register_slang_backend():
    """Register Slang backend."""
    try:
        from .backend.slang.SlangLexer import SlangLexer
        from .backend.slang.SlangParser import SlangParser
        from .backend.slang.SlangCrossGLCodeGen import SlangToCrossGLConverter
        from .translator.codegen.slang_codegen import SlangCodeGen

        class SlangLexerWrapper(BaseLexer):
            def __init__(self, code: str):
                self.old_lexer = SlangLexer(code)
                super().__init__(code)

            def get_token_patterns(self):
                return []

            def get_keywords(self):
                return {}

            def tokenize(self):
                old_tokens = self.old_lexer.tokenize()
                return [Token(TokenType.IDENTIFIER, text) for _, text in old_tokens]

        class SlangParserWrapper(BaseParser):
            def __init__(self, tokens):
                if tokens and hasattr(tokens[0], "type"):
                    old_tokens = [(token.type.name, token.value) for token in tokens]
                else:
                    old_tokens = tokens
                self.old_parser = SlangParser(old_tokens)
                super().__init__(tokens if hasattr(tokens[0], "type") else [])

            def parse(self):
                return self.old_parser.parse()

            def parse_statement(self):
                return None

            def parse_expression(self):
                return None

            def parse_type(self):
                return ""

            def parse_parameter(self):
                return None

            def parse_unary_expression(self):
                return None

        class SlangFromCrossGLWrapper(CrossGLToTargetConverter):
            def __init__(self):
                super().__init__("slang")
                self.codegen = SlangCodeGen()

            def generate(self, ast_node):
                return self.codegen.generate(ast_node)

        slang_backend = BackendInfo(
            name="slang",
            file_extensions=["slang"],
            lexer_class=SlangLexerWrapper,
            parser_class=SlangParserWrapper,
            to_crossgl_class=SlangToCrossGLConverter,
            from_crossgl_class=SlangFromCrossGLWrapper,
            description="NVIDIA Slang backend for shader programming",
        )

        BackendRegistry.register_backend(slang_backend)

    except ImportError as e:
        print(f"Warning: Could not register Slang backend: {e}")


# Initialize all backends when module is imported
initialize_builtin_backends()
