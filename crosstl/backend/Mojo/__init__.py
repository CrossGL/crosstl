from .MojoAst import *
from .MojoLexer import *
from .MojoParser import *


# Add a stub converter class
class MojoToCrossGLConverter:
    def __init__(self):
        self.name = "MojoToCrossGLConverter"

    def generate(self, ast):
        return "# Mojo to CrossGL conversion not yet implemented"
