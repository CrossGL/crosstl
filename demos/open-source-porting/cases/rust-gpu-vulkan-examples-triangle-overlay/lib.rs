#![no_std]

use spirv_std::glam::Vec4;
use spirv_std::spirv;

#[spirv(fragment)]
pub fn main_fs(out_frag_color: &mut Vec4) {
    *out_frag_color = Vec4::new(1.0, 1.0, 1.0, 1.0);
}
