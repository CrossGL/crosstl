Mojo output keeps the size of a local array whose size is a lowercase integer constant or a constant expression:
`float a[count]` and `float b[N * 2]` are declared as `StaticTuple[Float32, count]` and `StaticTuple[Float32, 8]`.
The first used to be written with the CrossGL type `float[count]` and the second as an empty `List[Float32]()`.
