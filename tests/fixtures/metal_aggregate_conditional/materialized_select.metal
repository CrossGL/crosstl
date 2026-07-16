#include "materialized_select.h"

template [[host_name("materialized_select_payload")]] [[kernel]]
decltype(materialized_select<Payload, SelectPayload>)
materialized_select<Payload, SelectPayload>;
