SPHINXBUILD ?= sphinx-build
SPHINXOPTS ?= -W --keep-going
SOURCEDIR = source
BUILDDIR = _build

.PHONY: html clean linkcheck

html:
	$(SPHINXBUILD) -M html "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS)

linkcheck:
	$(SPHINXBUILD) -M linkcheck "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS)

clean:
	rm -rf "$(BUILDDIR)"
