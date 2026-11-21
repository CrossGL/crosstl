Metal output reads a `QuadReadLaneAt` lane chosen at run time with `quad_shuffle`. It used to write
`quad_broadcast`, whose lane must be the same in every thread of the quad-group.
