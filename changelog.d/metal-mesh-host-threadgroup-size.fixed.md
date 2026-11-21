Metal output translates a mesh or object stage whose threadgroup size the host passes when it draws, as Metal input
marks every imported mesh and object function. It writes no `max_total_threads_per_threadgroup` for such a stage, and
still writes the limit that a `threadgroup_size` source option or a CrossGL size gives. It used to refuse every such
stage whose result depends on the threadgroup size.
