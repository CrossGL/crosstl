#pragma once

#include <metal_stdlib>

using namespace metal;

struct Payload {
  float value;
};

inline bool choose(thread uint& calls) {
  calls += 1u;
  return (calls & 1u) != 0u;
}

inline Payload make_payload(thread uint& calls, float value) {
  calls += 10u;
  return Payload{value};
}

inline Payload consume(Payload value) {
  return value;
}

struct SelectPayload {
  Payload operator()(
      thread uint& calls,
      Payload left,
      Payload right) const {
    Payload initialized =
        choose(calls) ? make_payload(calls, left.value)
                      : make_payload(calls, right.value);
    initialized = choose(calls) ? left : right;
    Payload argument = consume(
        choose(calls) ? make_payload(calls, left.value)
                      : make_payload(calls, right.value));
    return choose(calls) ? initialized : argument;
  }
};

template <typename T, typename Op>
[[kernel]] void materialized_select(
    device T* output [[buffer(0)]],
    uint index [[thread_position_in_grid]]) {
  Op op;
  uint calls = 0u;
  T left{1.0f};
  T right{2.0f};
  output[index] = op(calls, left, right);
}
