OpenCL C input imports work-item queries whose dimension is written with an
integer suffix or an integer cast (`get_global_id(1u)`, `get_global_id((uint)1)`).
The dimension used to be accepted only as a bare `0`, `1` or `2`, so the
suffixed and cast forms were copied unchanged and the CrossGL parser then
refused them as undeclared functions. All six queries (`get_global_id`,
`get_local_id`, `get_group_id`, `get_local_size`, `get_num_groups`,
`get_global_size`) now map a constant dimension 0, 1 or 2 to the `.x`, `.y`
or `.z` component in either spelling. A dimension of 3 or more, a float cast
or a runtime value keeps the previous behaviour.
