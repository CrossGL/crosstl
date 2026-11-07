"""Constrained free-function calls that bind over several rounds, or in functions no entry point reaches.

``_materialize_constrained_template_calls_in_scope`` binds a call to a constrained free-function template from the
types of its arguments. Two cases need more than its single strict pass.

Unreachable helpers. The importer translates every concrete function of a Metal source, including ``inline``
helpers that no kernel calls. A helper such as::

    template <typename T>
    enable_if_t<is_floating_point_v<T>, T> max(T a, T b);
    template <> inline bfloat max(bfloat a, bfloat b);
    inline float larger(float a, float b) { return max(a, b); }

would otherwise keep an unbound ``max(float, float)`` call. Its only remaining source candidate is then the
``bfloat`` explicit specialization, and the whole translation is refused even though no kernel executes
``larger``. Unreachable bodies are therefore resolved too. A call there that cannot be resolved is no error of the
program, because no entry point evaluates it, so that body is left as it was and the pass is repeated without it.

Arguments typed by an earlier binding. In::

    auto a = max(x, y);
    auto q = div(b, a);

``a`` has a type only once ``max`` is bound to a concrete instantiation, so ``div`` cannot be bound in the same
pass. Such a call is deferred, and its templates stay declared for the next round, which sees the instantiation's
return type. Rounds repeat while they make progress. A round reuses the instantiations that earlier rounds
appended, so a specialization is materialized once. A call that is still deferred when no further round makes
progress gets a final strict pass, which refuses it exactly as the single pass did.

Calls in reachable bodies are always resolved fail-closed.
"""

from typing import Any, Callable, Dict, FrozenSet, List, Optional, Set, Tuple

# Each soft retry repeats the whole pass; past this many failures the reachable-only pass decides.
MAX_SOFT_RETRIES = 4
# Each round binds the calls whose argument types the previous round's instantiations made known.
MAX_ROUNDS = 4

Span = Tuple[int, int]


class UnreachableCallerScope:
    """State shared with the constrained-call pass while this module drives it."""

    def __init__(self) -> None:
        self.active = False
        self.spans: Optional[FrozenSet[Span]] = None
        self.failure_offset: Optional[int] = None
        self.deferring = False
        self.deferred: List[int] = []
        # Instantiations earlier rounds appended, by template and arguments, so a later round reuses them.
        self.instantiations: Dict[Tuple[str, str, Tuple[str, ...]], str] = {}

    def reusable_instantiation(
        self, template: Any, arguments: Tuple[str, ...], present: Set[str]
    ) -> Optional[str]:
        """The helper an earlier round materialized for ``template<arguments>``, when the source still has it."""

        if not self.active:
            return None
        name = self.instantiations.get((template.name, template.source, arguments))
        return name if name is not None and name in present else None

    def record_instantiation(
        self, template: Any, arguments: Tuple[str, ...], name: str
    ) -> None:
        if self.active:
            self.instantiations[(template.name, template.source, arguments)] = name

    def extend(self, reachable_spans: Optional[List[Span]]) -> Optional[List[Span]]:
        """Return ``reachable_spans`` with the active unreachable spans added."""

        if reachable_spans is None or not self.spans:
            return reachable_spans
        return sorted(set(reachable_spans) | self.spans)

    def defer(self, offset: int) -> bool:
        """Record a call whose argument types are not known yet; whether it may wait for another round."""

        if not self.deferring:
            return False
        self.deferred.append(offset)
        return True


def unreachable_caller_scope(preprocessor: Any) -> UnreachableCallerScope:
    """The preprocessor's scope, created on first use."""

    scope = getattr(preprocessor, "_unreachable_caller_scope", None)
    if scope is None:
        scope = preprocessor._unreachable_caller_scope = UnreachableCallerScope()
    return scope


def _containing(offset: Optional[int], spans: FrozenSet[Span]) -> Optional[Span]:
    if offset is None:
        return None
    for start, end in sorted(spans):
        if start <= offset < end:
            return (start, end)
    return None


def _run(
    scope: UnreachableCallerScope,
    materialize: Callable[[str], str],
    code: str,
    spans: Optional[FrozenSet[Span]],
    defer: bool,
) -> Tuple[str, List[int]]:
    scope.spans = spans
    scope.failure_offset = None
    scope.deferring = defer
    scope.deferred = []
    try:
        return materialize(code), list(scope.deferred)
    finally:
        scope.spans = None
        scope.failure_offset = None
        scope.deferring = False
        scope.deferred = []


def _soft_pass(
    preprocessor: Any,
    scope: UnreachableCallerScope,
    materialize: Callable[[str], str],
    code: str,
    defer: bool,
) -> Tuple[str, List[int]]:
    """One pass over reachable and unreachable bodies; failures in unreachable bodies drop those bodies."""

    template_spans = preprocessor._find_template_declaration_spans(code)
    active = frozenset(preprocessor._unreachable_function_spans(code, template_spans))
    for _ in range(MAX_SOFT_RETRIES + 1):
        if not active:
            break
        try:
            return _run(scope, materialize, code, active, defer)
        except ValueError:
            failed = _containing(scope.failure_offset, active)
            if failed is None:
                # A reachable call, or a failure without a call: the reachable-only pass reports it.
                break
            active = active - {failed}
    return _run(scope, materialize, code, None, defer)


def materialize_with_unreachable_callers(
    preprocessor: Any, code: str, materialize: Callable[[str], str]
) -> str:
    """Run ``materialize`` in rounds over reachable and unreachable bodies (see the module docstring)."""

    scope = unreachable_caller_scope(preprocessor)
    if scope.active:
        # A nested pass keeps the outer pass's scope.
        return materialize(code)
    scope.active = True
    scope.instantiations = {}
    try:
        for _ in range(MAX_ROUNDS):
            result, deferred = _soft_pass(preprocessor, scope, materialize, code, True)
            if not deferred:
                return result
            if result == code:
                break
            code = result
        # Deferred calls the last round could not bind: resolve them strictly.
        return _soft_pass(preprocessor, scope, materialize, code, False)[0]
    finally:
        scope.active = False
        scope.instantiations = {}
