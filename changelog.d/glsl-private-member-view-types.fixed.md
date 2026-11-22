GLSL output types a private pointer to a struct member (`bump(&box.bits)`) through the struct the root value is
declared with. It used to look the member up by name in every struct, so a same-named array member of an unrelated
struct made it refuse the call with `missing-view-backing`.
