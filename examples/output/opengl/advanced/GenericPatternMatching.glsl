
#version 450 core
struct RenderState {};
layout(std140, binding = 0) OptionType variant;
layout(std140, binding = 1) ResultType variant;
layout(std140, binding = 2) T y;
layout(std140, binding = 3) T z;
layout(std140, binding = 4) Vec3 row1;
layout(std140, binding = 5) Vec3 row2;
