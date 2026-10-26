# Changelog

Notable changes to CrossTL for people who install and use it, newest first. Internal refactors, test and CI changes are left out; the commit history has them.

## [Unreleased]

- Mojo output uses `&` and `|` for component-wise `&&` and `||` on boolean vectors (`bvecN`). Scalar `bool` operands keep `and` and `or` so short-circuit evaluation is preserved. (Fixes #3137)

### Breaking Changes
