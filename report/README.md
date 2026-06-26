# Report (LNCS)

M.Sc. report: *Quantifying Process Model Generalization — A Generative N-gram Metric and a Cross-Paradigm Benchmark*. Restructured 2026-06-12 to the required 7-section report layout (~21 pages).

## Files
- `main.tex` — the report (Springer LNCS class `llncs`)
- `references.bib` — BibTeX references (style: `splncs04`)

## Structure (matches the supervisor's required layout)
1. Introduction (problem, solution, results, positioning, structure)
2. Background (logs, Petri nets, token replay's two readings, N-gram/Katz/Good–Turing)
3. Related Work (thematic groups + explicit gap statement)
4. Method (construct, framework, formal defs, Algorithm 1 pseudocode, running example, **time & space complexity**, scope)
5. Implementation (architecture, design decisions, tool-in-use listings, baseline realities, deviations)
6. Evaluation (setup, feasibility, cross-paradigm, version study, acceptance reading, runtime, discussion, threats)
7. Conclusion (contribution, limitations, future work)

## Build
```bash
pdflatex -enable-installer main && bibtex main && pdflatex -enable-installer main && pdflatex -enable-installer main
```
Uses `algorithm`/`algpseudocode` (pseudocode), `listings` (tool output), `pgfplots` (Figs. 2–3). All auto-install on first MiKTeX compile.

## Before submission (checklist)
- [ ] Real e-mail address(es) in `\institute` (currently `...`).
- [ ] **Hilfsmittel/KI declaration** per the chair's template — required regardless of prose rewrites (code base was AI-assisted, both partners). Ask the supervisor which template applies.
- [ ] Confirm author order with Tianhao (currently Geng, Krengel) and supervisor naming if required.
- [ ] Optional: rename HybridGen → GenShadow/ShadowGen (find-replace + drop the historical-name footnote) once the team decides.
- [ ] Optional: replace Listing 1.3 / the listings with real terminal screenshots if the template prefers them.

## Provenance
Numbers derive from `benchmark/results/configs/` (external methods, D1) and `benchmark/results/configs_v2/` (M1 family, D1+D2), seed 42. Re-running cells means regenerating the tables.
