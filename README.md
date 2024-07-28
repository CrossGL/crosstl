CrossGL Translator
==================

The CrossGL Translator is a core component of our platform, enabling the conversion of CrossGL shader code directly into various graphics APIs, such as DirectX, Metal, and Vulkan.

# Key Features

- **Direct s2s conversion**: Translates shaders directly to graphical languages / IR, bypassing many needless intermediate translations.
- **Unified Language**: Supports multiple platforms with a single, consistent unified AST.
- **Optimized Performance**: Generates highly optimized code tailored to the target backend.

# How It Works

The translator takes CrossGL shader code and processes it through several stages:

1. **Parsing**: The code is parsed into an abstract syntax tree (AST).
2. **Intermediate Representation**: The AST is converted into an intermediate representation (IR) for optimization.
3. **Code Generation**: The IR is translated into the target backend code.
4. **Optimization**: Various optimization passes are applied to ensure maximum performance.
5. **Source output**: The final output is produced and ready for use.

# Examples 
https://github.com/CrossGL/demos/tree/main

 |Backends                           |
| ----------------------------------- |
| Vulkan                           |
 | Metal |
| DirectX                  |
| OpenGL                          |

