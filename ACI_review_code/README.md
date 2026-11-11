# Code for "Aerosol–Cloud Interactions and Their Radiative Impacts" (literature review, v1–v4)

Everything needed to regenerate `ACI_Radiative_Impacts_Literature_Review_v4.docx`.

## Layout
| File | What it does |
|---|---|
| `build_review_v4.js` | Node (docx-js) script that builds the whole document: text, tables, references. Writes `..._v4_raw.docx`. |
| `section7.js` | Section 7 ("The physics supporting the results in ACI research"): text, Word-native equations (7.1–7.28), figures and Box 1. Called from `build_review_v4.js`. |
| `finalize_v4.py` | Post-processes the raw .docx: turns stand-alone equations into display equations and removes ZIP folder entries. Writes the final `..._v4.docx`. |
| `render_v4.sh` | Runs build → finalize → validate → PDF → page images, for checking. |
| `patch_v4.py` | How `build_review_v4.js` was produced from v3 (renumbering of old Sections 7–11 to 8–12, cross-references, new references). Kept as a record; not needed to rebuild. |
| `figs/physics.py` | Physics helpers, each formula tied to its source (Rogers & Yau equations and tables, Bohren & Clothiaux two-stream, Twomey activation, κ-Köhler). `python3 physics.py` runs self-checks against the source tables. |
| `figs/make_figs.py` | Draws all Section 7 figures from `physics.py` and source tables. `python3 make_figs.py` regenerates the PNGs and prints check values. |
| `figs/fig*.png` | The figures as embedded in v4. |
| `history/` | Build scripts for v1–v3 and the patch scripts that produced v2 and v3. |

## Figure file → figure number in the document
Files are numbered in drawing order; the document numbers them in order of appearance:
fig01→1, fig02→2, fig03→3, fig04→4, fig05→5, fig06→6, fig07→7, fig08→8, **fig10_mie→9, fig09_dots→10**, fig11→11, fig12→12, fig13→13, **fig16_absorb→14, fig14_evap→15, fig15_net→16**.

## Rebuilding
Requirements: Node with the `docx` npm package; Python 3 with numpy, matplotlib, miepython, Pillow; the Carlito font (falls back otherwise); LibreOffice (with the Math component) and poppler only for rendering checks.

```bash
cd figs && python3 make_figs.py && cd ..   # optional: regenerate figures
node build_review_v4.js                      # -> ACI_Radiative_Impacts_Literature_Review_v4_raw.docx
python3 finalize_v4.py                       # -> ACI_Radiative_Impacts_Literature_Review_v4.docx
```
`render_v4.sh` additionally validates and renders; it expects the docx skill's `office/` scripts (set `DOCX_SCRIPTS` to their folder).

## Caveats
- The parcel model in `physics.py` (Figure 5) is a simplified integration written for this review; it differs from Twomey's formulas by 8–17%.
- `make_figs.py` hard-codes source table values (Rogers & Yau Tables 6.2, 7.2, 7.3, 8.1, 8.2; Dedrick et al., 2026 EPCAPE values) transcribed from the textbooks.
