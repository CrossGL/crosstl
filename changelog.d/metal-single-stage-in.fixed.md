Metal output gives a vertex or fragment entry that reads several stage inputs
one `[[stage_in]]` structure. Each structure parameter becomes a nested member
of the entry's generated `<entry>_Input` structure, beside the loose inputs, so
the `[[attribute(n)]]` indices the host binds and the fragment links to vertex
outputs stay the same. It used to give every structure parameter its own
`[[stage_in]]`, which Apple's compiler rejects. Overlapping vertex attribute
indices are refused.
