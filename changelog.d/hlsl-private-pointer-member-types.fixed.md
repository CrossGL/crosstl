HLSL output types a private-pointer argument `&local.member` by the declared structure of the local. When another
structure declares a member of the same name with another type (`uint bits` and `uint bits[2]`), the pre-pass that
classifies pointer arguments used to guess the member's type by name and refused the call as a mixed scalar and
array backing.
