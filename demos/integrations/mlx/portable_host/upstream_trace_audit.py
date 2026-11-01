"""Check native dispatch records one at a time, as the host writes them.

A long upstream group dispatches tens of gigabytes of retained values. The
native worker therefore runs every per-dispatch check on each record before it
is written and keeps only the record's compact form on disk; the verifier ties
the compact records to their packages afterwards.
"""

import json

from demos.integrations.mlx.portable_host.trace_records import (
    TraceDigest,
    compact_trace_values,
)


class TraceAudit:
    """Run per-dispatch checks and keep what the package audit needs.

    ``audits`` are ``(selects, family, check, role)`` tuples in audit order,
    as ``verify_upstream_ops.event_audits`` builds them. For each record,
    ``owners`` counts the checks that own it and ``identities`` says whether
    its native identity is still unchecked. ``digest`` counts and hashes the
    full lines the host would have written.
    """

    def __init__(self, audits, target):
        self.audits = audits
        self.target = target
        self.owners = []
        self.identities = []
        self.indexing = 0
        self.error = None
        self.digest = TraceDigest()

    def check(self, event):
        """Run every check that selects ``event``; return its compact form."""
        entry, owned, identity = event["entry"], 0, True
        for selects, family, check, role in self.audits:
            if not selects(entry, event):
                continue
            if family is not None and event["target"] != self.target:
                raise ValueError(f"Upstream op {family} targets another backend")
            covered = check(event)
            identity = identity and role == "scan" and not covered
            owned += role != "random"
            self.indexing += role == "indexing"
        self.owners.append(owned)
        self.identities.append(identity)
        return compact_trace_values(event)

    def __call__(self, event, line):
        """The host's ``trace_audit``: check ``event``, return its compact line.

        The first failure is kept and refuses every later record, so a test
        that expects an exception cannot hide it.
        """
        if self.error is not None:
            raise RuntimeError(f"The native trace audit failed: {self.error}")
        self.digest.add(line.encode("utf-8"))
        try:
            compact = self.check(event)
        except Exception as error:
            self.error = f"{event.get('entry')}: {error}"
            raise
        return json.dumps(compact) + "\n"

    def summary(self):
        """The audit record the verifier reads beside the compact trace."""
        return {
            **self.digest.summary(),
            "owners": self.owners,
            "identities": self.identities,
            "indexing": self.indexing,
            "error": self.error,
        }
