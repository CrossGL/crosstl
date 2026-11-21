OpenCL C input preserves program-scope `__global` variables as CrossGL device
memory. Targets without equivalent mutable device globals now refuse them instead
of emitting invalid code (#4563).
