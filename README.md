# disagg

Disaggregation & heterogeneity explorer for datacenter LLM inference.

Forked from `~/transformer_math.html` (preserved at `reference/transformer_math.html`).
The goal: pick a model + batch + **disaggregation axis** (prefill/decode, attention/expert, …),
sweep heterogeneous chip splits, and trace the **throughput × interactivity × $/token** Pareto
frontier — the thing the original tool explicitly listed as "not modeled."

## Layout
- `src/engine/core.js` — analytical engine extracted **verbatim** (ref 8179–8756): chip catalogue
  (`chipPerfSpecs`), model presets (`modelPresets`), FLOPs/bytes kernel (`getStepCost`), roofline +
  capacity + interconnect model (`computeChipSummary`), parallelism planner (`deriveDeploymentMode`).
  Only change vs original: `export` block at the bottom.
- `reference/` — the original 11k-line HTML, untouched.
- `test/anchors.mjs` — reproduces published benchmark points to sanity-check the math.
  Run: `npm run test:anchors`
- `audit/AUDIT.md` — full first-pass audit (math + benchmark data). **Read this first.**

## Status
- ✅ Engine forked, audited, catalogue corrected w/ provenance (`audit/AUDIT.md`).
- ✅ Sustained-effective convention (per-chip MFU + BW-eff).
- ✅ **MoE low-batch fix (M1)** — DeepSeek-V3 B=1 now ~51 tok/s (was ~770).
- ✅ **Two-tier memory** (fast SRAM/HBM + cold LPDDR/CXL) — `mem_*_cold`, `weight_tier`. Drives d-Matrix.
- ✅ **Disaggregation axes** (`src/engine/disagg.js`) — three, all sharing one point shape + Pareto/UI:
  - **Prefill / Decode** — phase split; KV shipped prefill→decode once per request.
  - **Attention / Expert** (MoE) — per-layer hidden-state transfer between bandwidth-attention and
    capacity-expert pools (transfer is per-layer, both directions — the make-or-break cost).
  - **Speculative decoding** — draft/target split; accepted tokens = (1-α^(K+1))/(1-α); ×2–3 speedup.
- ✅ **UI** (`index.html`) — axis selector, chip-pool pickers w/ provenance tags, throughput × interactivity
  × $/Mtok Pareto chart (◆ hetero / ○ homo), hetero-vs-homo callout. Self-contained ES modules, no build.

## Run
- `npm run ui` → http://localhost:5173 (needs a server, not file://, for ES-module imports)
- `npm test` → validate (catalogue + closed-form + reality bands) and axis regression
- `npm run test:disagg` → frontier demo

## Next
- Calibrate d-Matrix with a real tok/s anchor (over-predicts until then).
- Embedding/encoder axis (needs multimodal model presets).
- Per-chip MFU calibration beyond the global 0.50/0.70 default.

## ⚠ Confidentiality
Catalogue contains vendor-proprietary data (d-Matrix slide marked "Proprietary"; Tensordyne deck-derived
numbers). **Do not make the repo public / CDN-serve it** without scrubbing those rows.
