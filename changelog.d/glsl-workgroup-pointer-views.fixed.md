GLSL output lowers a pointer view of a shared array that keeps the array's
element type, such as `reinterpret_cast<threadgroup float*>(tile)` over a
`threadgroup float` array, to the array plus an offset. It used to refuse the
view as having no shared-array backing object. A view that changes the element
type is refused with a typed diagnostic (#4823).
