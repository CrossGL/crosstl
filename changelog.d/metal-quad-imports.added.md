Metal input imports the quad-group functions: `quad_all` and `quad_any` as `QuadAll` and `QuadAny`,
`quad_product`, `quad_min`, `quad_max`, `quad_and`, `quad_or` and `quad_xor` as clustered reductions of four lanes,
and `quad_broadcast`, `quad_shuffle`, `quad_shuffle_xor`, `quad_shuffle_up`, `quad_shuffle_down` and the
`quad_shuffle_rotate_*` functions as `QuadReadLaneAt` of the lane each reads (`QuadReadAcrossX`, `Y` and `Diagonal`
for a constant xor of 1, 2 and 3). Other output languages used to receive the Metal names, which they do not define.
