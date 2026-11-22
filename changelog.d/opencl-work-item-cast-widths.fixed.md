OpenCL C input converts a constant work-item dimension under an integer cast to the cast type's width, as C does.
`get_local_size((uchar)256)` used to import as the out-of-range size `1u`, though C passes dimension 0, and
`get_group_id((ushort)65537)` as `0u`, though C passes dimension 1; both now keep the call, which the translation
refuses like any other cast dimension 0, 1 or 2 (#4819). A cast that wraps out of range still folds to the
out-of-range value: `get_local_size((uchar)259)` is `1u`.
