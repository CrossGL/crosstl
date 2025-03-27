from .VulkanLexer import VulkanLexer
from .VulkanParser import VulkanParser
from .VulkanAst import ASTNode, ShaderNode, FunctionNode

# Add a stub converter class
class VulkanToCrossGLConverter:
    def __init__(self):
        self.name = "VulkanToCrossGLConverter"
        
    def generate(self, ast):
        return "# Vulkan to CrossGL conversion not yet implemented"
