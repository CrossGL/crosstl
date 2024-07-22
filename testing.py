from crosstl import transpiler

code = """shader main {
                            input vec3 position;
                            input vec2 texCoord;
                            output vec4 fragColor;
                            vec3 customFunction(vec3 random, float factor) {
                                return random * factor;
                            }

                            void main() {
                                vec3 color = vec3(position.x,position.y, 0.0);
                                float factor = 1.0;

                                if (texCoord.x > 0.5) {
                                    color = vec3(1.0, 0.0, 0.0);
                                } else {
                                    color = vec3(0.0, 1.0, 0.0);
                                }

                                for (int i = 0; i < 3; i = i + 1) {
                                    factor = factor * 0.5;
                                    color = customFunction(color, factor);
                                }

                                if (length(color) > 1.0) {
                                    color = normalize(color);
                                }

                                fragColor = vec4(color, 1.0);
                            }
                        }"""

backend = "metal"

if __name__ == "__main__":
    metal_transpiler = transpiler(code, backend)
    print("############ metal ############")
    print(metal_transpiler)
    directx_transpiler = transpiler(code, "directx")
    print("############ directx ############")
    print(directx_transpiler)
    opengl_transpiler = transpiler(code, "opengl")
    print("############ opengl ############ ")
    print(opengl_transpiler)
