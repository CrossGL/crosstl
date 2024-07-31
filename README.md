
# CrossGL Translator

The CrossGL Translator is a core component of our platform, enabling the conversion of CrossGL shader code directly into various graphics APIs, such as DirectX, Metal, Vulkan, and OpenGL. This translator simplifies shader development by allowing a single, unified shader language to be used across multiple platforms.

## 🌟 CrossGL: Revolutionizing Shader Development

### The Universal Shader Language

In the ever-evolving world of graphics programming, **CrossGL** emerges as a game-changing solution, bridging the gap between diverse graphics APIs.

#### 🚀 Write Once, Run Everywhere

Imagine writing a shader _once_ and deploying it across:

- 🍎 **Metal**
- 🎮 **DirectX**
- 🖥️ **OpenGL**
- 🖥️ **Vulkan**

...all without changing a single line of code!

## 💡 Key Benefits

1. **⏱️ Time-Saving**: Slash development time by eliminating the need for multiple shader versions.
2. **🛠️ Consistency**: Ensure uniform behavior across all platforms.
3. **🧠 Simplified Learning Curve**: Master one language instead of many.
4. **🔍 Enhanced Debugging**: Develop universal tools for shader analysis.
5. **🔮 Future-Proof**: Easily adapt to new graphics APIs as they emerge.

## How It Works

The translator takes CrossGL shader code and processes it through several stages:

1. **Parsing**: The code is parsed into an abstract syntax tree (AST).
2. **Intermediate Representation**: The AST is converted into an intermediate representation (IR) for optimization.
3. **Code Generation**: The IR is translated into the target backend code.
4. **Optimization**: Various optimization passes are applied to ensure maximum performance.
5. **Source Output**: The final output is produced and ready for use.

## 🔄 Two-Way Translation: From Platform-Specific to CrossGL

CrossGL doesn't just translate from a universal language to platform-specific shaders - it also works in reverse! This powerful feature allows developers to convert existing shaders from various platforms into CrossGL.

## 🌈 CrossGL Shader Example

```cpp
shader main {
    vertex {
        input vec3 position;
        output vec2 vUV;

        void main() {
            vUV = position.xy * 10.0;
            gl_Position = vec4(position, 1.0);
        }
    }

    float perlinNoise(vec2 p) {
        return fract(sin(dot(p, vec2(12.9898, 78.233))) * 43758.5453);
    }

    fragment {
        input vec2 vUV;
        output vec4 fragColor;

        void main() {
            float noise = perlinNoise(vUV);
            float height = noise * 10.0;
            vec3 color = vec3(height / 10.0, 1.0 - height / 10.0, 0.0);
            fragColor = vec4(color, 1.0);
        }
    }
}

```

## Getting Started

To get started with the CrossGL Translator, check out our [Getting Started Notebook](https://colab.research.google.com/drive/1reF8usj2CA5R6M5JSrBKOQBtU4WW-si2?usp=sharing#scrollTo=D7qkQrpcQ7zF).

## Supported Backends

- Vulkan
- Metal
- DirectX
- OpenGL

## Examples

Explore example projects and demos showcasing the CrossGL Translator's capabilities: [CrossGL Demos](https://github.com/CrossGL/demos/tree/main).

## 📚 Documentation

Comprehensive documentation is available to help you get started and master CrossGL:

- [CrossGL Documentation](https://crossgl.github.io/translator.html)


## Contribution Guidelines

We welcome contributions to the CrossGL Translator. To get started, please read our [Contribution Guidelines](https://crossgl.github.io/contribution.html).

### Steps to Contribute:

1. Fork the repository.
2. Create a new branch for your feature or bugfix.
3. Write your code and tests.
4. Ensure all tests pass.
5. Submit a pull request with a detailed description of your changes.

For more detailed information, visit the [Contribution Guidelines](https://crossgl.github.io/contribution.html).

## License

The CrossGL Translator is open-source and licensed under the [License](https://github.com/CrossGL/crosstl/blob/main/LICENSE).

---

Stay connected and follow our latest updates and announcements:

- [Twitter](https://x.com/crossGL_)
- [LinkedIn](https://www.linkedin.com/company/crossgl/?viewAsMember=true)
- [Discord Channel](https://discord.gg/mYH45zZ9)
- [YouTube](https://www.youtube.com/channel/UCxv7_flRCHp7p0fjMxVSuVQ)

---

Thank you for using the CrossGL Translator!

The CrossGL Team
