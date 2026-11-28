GLSL output lowers a pointer view of a shared array that changes the element width, such as
``reinterpret_cast<threadgroup float*>(tile)`` over a ``threadgroup half`` array, to one ``shared uint`` array of
32-bit words, with 16- and 8-bit elements packed as lanes and read or written through bit extraction and
insertion. A view that keeps the 32-bit width but changes the type is still refused with a typed diagnostic
(#4823).
