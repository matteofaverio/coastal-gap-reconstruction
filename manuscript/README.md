# Manuscript

Final written outputs of this project, plus the LaTeX sources needed to
reproduce them.

## Contents

| Document | Audience/language | PDF | Source entry point |
|---|---|---|---|
| Final report | English, technical | `report/coastal_gap_reconstruction_report.pdf` | `report/main.tex` |
| Academic presentation | English, technical | `presentation/coastal_gap_reconstruction_presentation_en.pdf` | `presentation/main.tex` |
| Colleague presentation | Spanish, non-specialist | `presentation_colleagues_es/coastal_gap_reconstruction_presentation_es.pdf` | `presentation_colleagues_es/main.tex` |
| Scientific poster | English, technical | `poster/coastal_gap_reconstruction_poster.pdf` | `poster/main.tex` |

```
report/
  coastal_gap_reconstruction_report.pdf   Final manuscript (chlorophyll Case Study 1 + oxygen Case Study 2)
  main.tex             LaTeX entry point
  sections/             Section source files (01-07, appendix)
  references.bib        Bibliography
  figures/               Every figure the manuscript cites (self-contained)

presentation/
  coastal_gap_reconstruction_presentation_en.pdf   Final English slide deck
  main.tex             LaTeX (beamer) entry point
  figures/               Every figure the deck cites (reused from the report where applicable)

presentation_colleagues_es/
  coastal_gap_reconstruction_presentation_es.pdf   Spanish, non-specialist colleague deck
  main.tex             LaTeX (beamer) entry point
  figures/               Self-contained figure set (some reused/translated from presentation/)

poster/
  coastal_gap_reconstruction_poster.pdf   Scientific poster (single case study: chlorophyll)
  main.tex             LaTeX entry point
  assets/, figures/     Self-contained logo/figure set
```

## Reproducing the PDFs

All four documents compile with [Tectonic](https://tectonic-typesetting.github.io/)
(a self-contained LaTeX engine — no separate TeX Live install needed) or any
standard TeX Live distribution:

```bash
cd manuscript/report && tectonic main.tex
cd ../presentation && tectonic main.tex
cd ../presentation_colleagues_es && tectonic main.tex
cd ../poster && tectonic main.tex
```

Each `main.tex` uses only standard packages (`article`/`beamer`, `graphicx`,
`booktabs`, `siunitx`, `natbib`, `tikz`, `tcolorbox`, etc.) — no custom
`.cls`/`.sty` files or external data lookups are required to compile.

## Relationship to the rest of this repository

The manuscript is the narrative synthesis of the results in
`results_public/` and `data_public/` — see `docs/evidence_hierarchy.md` for
how to weigh claims made in the text against the underlying validation
tables. The manuscript is the authoritative source for terminology; where
prose in this `docs/` folder differs, the manuscript prevails.
