Metal input lowers member calls on receivers declared through a struct's member type alias of a Metal library
template again, such as `typedef typename Frag_t::mat_type mat_type; mat_type m; m.thread_elements()[i] = x;`
where the fragment declares `typedef metal::simdgroup_matrix<T, 8, 8> mat_type;`. Since namespace-qualified
receiver types were recognised, such receivers failed with `project.translate.metal-struct-method`.
