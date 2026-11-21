Mojo and Rust output refuse a struct with a `@union_layout` instead of writing it as a struct whose overlapping
members have separate storage, so a program that writes one member and reads another is no longer translated
into one that reads a field that was never written (#4774).
