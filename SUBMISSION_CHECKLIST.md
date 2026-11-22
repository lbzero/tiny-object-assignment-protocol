# Submission checklist (IEEE JSTARS)

Ordered by whether it blocks submission. Nothing here is optional; the two items in
§1 must be closed before the manuscript leaves the machine.

## 1. Blocking

- [ ] **Zenodo DOI exists** and the manuscript's Code Availability paragraph cites it.
      Currently the paragraph contains literal placeholders: `[repository URL]` and
      `[Zenodo DOI]`. A PDF containing them must not be submitted.
- [x] **GitHub repository is public** and the tag `v1.0.0` matches the release.
      https://github.com/lbzero/tiny-object-assignment-protocol (tag moved to the
      licence commit on 2026-11-22; GitHub reports AGPL-3.0).
- [x] **Licence decided**: AGPL-3.0, because the code imports Ultralytics YOLO.
      Declared in `LICENSE`, `CITATION.cff`, `.zenodo.json` and Section VIII.
- [ ] **Funding and Acknowledgements** filled in, or the paragraphs removed.
      Currently marked `[to be completed]`.
- [ ] **No placeholder text anywhere** in `jstars/main.pdf`:
      `grep -n "TODO\|to be completed\|repository URL\|Zenodo DOI" 60-paper/latex/*.tex 60-paper/latex/sections/*.tex`

- [ ] **No `\pending{}` left.** Every placeholder is wrapped in one macro that emits a
      build warning, so the log names each one:
      `grep -c "PLACEHOLDER still in text" jstars/main.log` must be 0
      (it is 4 while the repository URL, the Zenodo DOI, funding and acknowledgements
      are outstanding).
- [ ] **Same-node check is cited.** The supplementary table of the epoch-60 EMA
      evaluation exists and the main text says which node each number comes from.

## 2. Numbers that must agree across abstract, text, tables and figures

- [ ] 2,804 images / 75,091 official boxes / 75,065 training-visible boxes.
      The three numbers appear in §3 and S5; nowhere may 75,091 and 75,065 be
      combined in one ratio.
- [ ] Subset A: 500 images, 12,184 boxes, 1,700 very tiny.
      Subset B: 500 images, 14,089 boxes, 2,002 very tiny.
- [ ] Diagonal cells 4.15% / 4.16% on A and 5.53% / 5.86% on B.
- [ ] Interface effect −1.38 (A) / −1.41 (B); weight effect +1.82 → +2.88 (CIoU)
      and +1.40 → +1.74 (NWD); interaction −0.43 → −1.15.
- [ ] Abstract quotes +1.82 and +2.88 and states direction consistency; it does not
      call the pair a confidence interval or a range of a population effect.
- [ ] Static proxy: 8.88% vs 5.13%, 1,221 flags / 343 correct, 8.4% gate route,
      91.6% competition route, +3.50 (A) / +5.66 (B) / +3.75 (full).
- [ ] AP75 −4.00, AP50 unchanged, AP −1.69.
- [ ] Every number above regenerates from `40-experiments/` or the archived traces.

## 3. References

- [ ] 32 entries, all verified against DOI/Crossref records; volume, pages and year
      match the publisher record, not a secondary source.
- [ ] `aitodpycocotools` cited as software with its version, not as a paper.
- [ ] No claim about another work that its own text does not support (NWD ablation
      12.8 → 20.2, APvt 0.0 → 3.3 is \citet{xu2022nwd}, quoted as such).

## 4. Format

- [ ] Two-column build compiles with zero errors and no undefined references.
      Current: 19 pages, one 2.33 pt overfull hbox in an inline formula.
- [ ] Page budget decided. JSTARS charges for pages above the free allowance; the
      three compression candidates are (i) move Table III to the supplement,
      (ii) trim §2.1 by a third, (iii) merge two single-column figures. §4 and §6
      evidence chains are not to be cut.
- [ ] Figures are vector PDFs, colour-safe, and legible in greyscale (redundant
      hatching where colour carries meaning).
- [ ] Title page: author, affiliation, e-mail, ORCID correct; no second author.

## 5. Declarations

- [ ] Data availability: imagery not redistributed, derived artefacts released.
- [ ] Code availability: repository + DOI (see §1).
- [ ] Competing interests: none.
- [ ] Funding: filled or removed.
- [ ] Acknowledgements: filled or removed.

## 6. Final independent check

- [ ] Regenerate all seven supplementary tables from the archive and diff against
      the submitted `supp_tables/*.tex`. At v1.0.0 the diff is empty.
- [ ] Re-run one diagnostic cell from scratch (one checkpoint, one subset) and
      confirm the trace row count and the very-tiny zero-positive rate match the
      archived file.
- [ ] Read the PDF once as a reviewer would: look for sentences that claim more than
      the design supports, starting with any sentence containing "proves",
      "demonstrates that", "causes" or "confirms".
