HLSL output writes three CrossGL brace-list forms that DXC used to reject. A structure that a
brace list leaves partly zero, as `S s = {};`, zeroes its array members element by element
instead of assigning a brace list (#3734). A short brace list for a structure constant,
`const W C = {{1.0, 2.0}};`, becomes an initializer list with the omitted members zero (#3716).
A matrix brace list of column vectors, `mat2 m = {vec2(1.0, 0.0), vec2(0.0, x)}`, takes one
column per vector instead of padding each vector with zeros (#3718).
