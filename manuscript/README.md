# Manuscript

Final written outputs of this project, plus the LaTeX sources needed to
reproduce them.

## Contents

```
report/
  REPORT.pdf          Final manuscript (chlorophyll Case Study 1 + oxygen Case Study 2)
  main.tex             LaTeX entry point
  sections/             Section source files (01-07, appendix)
  references.bib        Bibliography
  figures/               Every figure the manuscript cites (self-contained)

presentation/
  PRESENTATION.pdf     Final slide deck
  main.tex             LaTeX (beamer) entry point
  figures/               Every figure the deck cites (reused from the report where applicable)
```

## Reproducing the PDFs

Both documents compile with [Tectonic](https://tectonic-typesetting.github.io/)
(a self-contained LaTeX engine — no separate TeX Live install needed) or any
standard TeX Live distribution:

```bash
cd manuscript/report
tectonic main.tex        # -> main.pdf

cd ../presentation
tectonic main.tex        # -> main.pdf
```

Both `main.tex` files use only standard packages (`article`/`beamer`,
`graphicx`, `booktabs`, `siunitx`, `natbib`, `tikz`, etc.) — no custom
`.cls`/`.sty` files or external data lookups are required to compile.

## Relationship to the rest of this repository

The manuscript is the narrative synthesis of the results in
`results_public/` and `data_public/` — see `docs/evidence_hierarchy.md` for
how to weigh claims made in the text against the underlying validation
tables. The manuscript is the authoritative source for terminology; where
prose in this `docs/` folder differs, the manuscript prevails.
