
groupshared int a;
// Compute Shader
[numthreads(64, 1, 1)]
void CSMain() {
    a = 123;
    int4 x = int4(a, a, a, a);
}
