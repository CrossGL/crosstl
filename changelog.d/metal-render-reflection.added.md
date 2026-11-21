Coverage checks the host interface of every Metal input round trip: the
entry points of the Metal CrossTL writes back must keep the argument-table
indices, argument-buffer layouts, vertex attributes, stage inputs and render
targets of the native probe, read from the AIR metadata the Metal compiler
writes (`tools/coverage/metal_reflection.py`). A render probe, which never runs,
used to count as supported when its translation moved a binding.
