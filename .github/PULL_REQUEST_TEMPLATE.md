### PR Description
<!-- Provide a brief summary of the changes you have made. Explain the purpose and motivation behind these changes. -->

### Related Issue
<!-- Link to the related issue(s) that this PR addresses. Example: #123 -->

### shader Sample
<!-- Provide a shader sample or snippet on which you have tested your changes. like

```crossgl
shader PerlinNoise {
    vertex {
        input vec3 position;
        output vec2 vUV;

        void main() {
            vUV = position.xy * 10.0;
            gl_Position = vec4(position, 1.0);
        }
    }

    // Perlin Noise Function
    float perlinNoise(vec2 p) {
        return fract(sin(dot(p, vec2(12.9898, 78.233))) * 43758.5453);
    }

    // Fragment Shader
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
```-->


### Checklist
- [ ] Have you added the necessary tests?
- [ ] Only modified the files mentioned in the related issue(s)?
- [ ] Are all tests passing?


