Metal input lowers the methods of structs declared in namespaces when a call
names the receiver's struct with its namespace (`ops::Reduce::apply(r, x)` or a
receiver of type `ops::Reduce`), and gives structs of the same name in
different namespaces (`ops::P`, `alt::P`) distinct CrossGL names following C++
name lookup. Both used to fail or merge the structs; an unqualified name that
finds both is refused as ambiguous.
