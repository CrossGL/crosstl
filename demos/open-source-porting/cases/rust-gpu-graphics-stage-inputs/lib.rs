use spirv_std::glam::{vec3, Vec2, Vec3, Vec4};
use spirv_std::spirv;

fn tonemap(color: Vec3) -> Vec3 {
    color
}

#[spirv(fragment)]
pub fn main_fs(output: &mut Vec4) {
    let color = vec3(1.0, 0.5, 0.25);
    *output = tonemap(color).extend(1.0);
}

#[spirv(vertex)]
pub fn main_vs(pos: Vec2, #[spirv(position)] builtin_pos: &mut Vec4) {
    *builtin_pos = pos.extend(0.0).extend(1.0);
}
