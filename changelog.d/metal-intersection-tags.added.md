Metal input imports intersection functions tagged `instancing`, `triangle_data`,
`world_space_data`, `primitive_motion`, `instance_motion`, `extended_limits`,
`curve_data` or `max_levels<N>`, and keeps `[[primitive_id]]`, `[[instance_id]]`,
`[[front_facing]]` and `[[barycentric_coord]]` under their Metal names. Metal
output writes the tags an intersection entry states plus the tags its inputs
need, and accepts `[[user_instance_id]]`, `[[time]]`, and triangle
`[[distance]]` and `[[front_facing]]` inputs.
