# CrossGL Paper

LaTeX source for the CrossGL paper.

## Files

- `crossgl.tex` — main LaTeX source
- `crossgl.pdf` — compiled output
- `svproc.cls` — Springer proceedings document class
- `aliascnt.sty`, `remreset.sty` — auxiliary style packages
- `svind.ist`, `svindd.ist` — index style files
- `bibtex/` — BibTeX bibliography styles

## Building

```bash
pdflatex crossgl.tex
bibtex crossgl
pdflatex crossgl.tex
pdflatex crossgl.tex
```
