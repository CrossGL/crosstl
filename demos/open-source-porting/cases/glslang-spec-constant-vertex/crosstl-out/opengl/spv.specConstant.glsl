#version 400
layout(constant_id = 16) const int arraySize = 5;
layout(constant_id = 17) const bool spBool = true;
layout(constant_id = 18) const float spFloat = 3.14;
layout(constant_id = 19) const double spDouble = double(3.141592653589793);
layout(constant_id = 22) const uint scale = 2;
layout(constant_id = 116) const int dupArraySize = 12;
layout(constant_id = 117) const bool spDupBool = true;
layout(constant_id = 118) const float spDupFloat = 3.14;
layout(constant_id = 119) const double spDupDouble = double(3.141592653589793);
layout(constant_id = 122) const uint dupScale = 2;
layout(constant_id = 24) gl_MaxImageUnits;

in vec4 ucol[arraySize];
in vec4 dupUcol[dupArraySize];
out vec4 color;
flat out int size;
// Vertex Shader
void foo(vec4 p[arraySize]);

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
