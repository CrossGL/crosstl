#version 400
/* CrossGL fallback: OpenGL source validation cannot preserve specialization constant id 16 for 'arraySize'; using the default literal. */
const int arraySize = 5;
/* CrossGL fallback: OpenGL source validation cannot preserve specialization constant id 17 for 'spBool'; using the default literal. */
const bool spBool = true;
/* CrossGL fallback: OpenGL source validation cannot preserve specialization constant id 18 for 'spFloat'; using the default literal. */
const float spFloat = 3.14;
/* CrossGL fallback: OpenGL source validation cannot preserve specialization constant id 19 for 'spDouble'; using the default literal. */
const double spDouble = 3.141592653589793;
/* CrossGL fallback: OpenGL source validation cannot preserve specialization constant id 22 for 'scale'; using the default literal. */
const uint scale = 2;
/* CrossGL fallback: OpenGL source validation cannot preserve specialization constant id 116 for 'dupArraySize'; using the default literal. */
const int dupArraySize = 12;
/* CrossGL fallback: OpenGL source validation cannot preserve specialization constant id 117 for 'spDupBool'; using the default literal. */
const bool spDupBool = true;
/* CrossGL fallback: OpenGL source validation cannot preserve specialization constant id 118 for 'spDupFloat'; using the default literal. */
const float spDupFloat = 3.14;
/* CrossGL fallback: OpenGL source validation cannot preserve specialization constant id 119 for 'spDupDouble'; using the default literal. */
const double spDupDouble = 3.141592653589793;
/* CrossGL fallback: OpenGL source validation cannot preserve specialization constant id 122 for 'dupScale'; using the default literal. */
const uint dupScale = 2;

in vec4 ucol[arraySize];
in vec4 dupUcol[dupArraySize];
out vec4 color;
flat out int size;
// Vertex Shader
void foo(vec4 p[arraySize]) {
    color += dupUcol[2];
    size += dupArraySize;
    if (spDupBool) {
        color *= dupScale;
    }
    color += float((spDupDouble / spDupFloat));
}

int builtin_spec_constant() {
    int result = gl_MaxImageUnits;
    return result;
}

void main() {
    color = ucol[2];
    size = arraySize;
    if (spBool) {
        color *= scale;
    }
    color += float((spDouble / spFloat));
    foo(ucol);
    return;
}
