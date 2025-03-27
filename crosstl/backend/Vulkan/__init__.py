from .VulkanLexer import *
from .VulkanParser import *
from .VulkanAst import *


# Add a stub converter class
class VulkanToCrossGLConverter:
    def __init__(self):
        self.name = "VulkanToCrossGLConverter"

    def generate(self, ast):
        return "# Vulkan to CrossGL conversion not yet implemented"
