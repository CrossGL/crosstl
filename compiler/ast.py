class ASTNode:
    pass

class ShaderNode(ASTNode):
    def __init__(self, name, inputs, outputs, main_function):
        self.name = name
        self.inputs = inputs
        self.outputs = outputs
        self.main_function = main_function

class FunctionNode(ASTNode):
    def __init__(self, name, body):
        self.name = name
        self.body = body

class VariableNode(ASTNode):
    def __init__(self, vtype, name):
        self.vtype = vtype
        self.name = name

class AssignmentNode(ASTNode):
    def __init__(self, name, value):
        self.name = name
        self.value = value
