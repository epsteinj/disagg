# Engine Audit — forked from `transformer_math.html`

Status: first full pass, 2026-06-06. Covers (1) the math/methodology and (2) the
benchmark catalogue. Engine extracted verbatim to `src/engine/core.js` (ref lines
8179–8756); catalogue split out to `src/engine/chips.js` with provenance.

### Applied 2026-06-06
- **Sustained-effective convention** added to `computeChipSummary` (`comp_eff` 0.50 / `bw_eff` 0.70
  GPU-ASIC default; Groq & Cerebras = 1.0 to avoid double-derate). Anchors now land in range:
  H100 70B decode 48→**33.5** tok/s, prefill 0.43→**0.77 s**, B200 fp8 70B 114→**80**, 405B 45→**32**.
  Groq **282** / Cerebras **1,856** unchanged (already calibrated).
- **Group A factual fixes applied:** Tenstorrent cost 15k→**1,399**, fp8 1490→**664**, fp16 745→**664**,
  interconnect 1e12→**0.4e12**; Meta MTIA fp16 177→**354**; Trainium2 fp16 658→**667**;
  Rubin label→**H2 2026**, rack_size 144→**72**, fp4 20000→**25000**; Groq fp16 20→**188**, ic 5e12→**0.33e12**.
- **rack_size = "largest practical single-job domain":** Huawei 8→**384**, TPU stays 256, Rubin 72.
  Huawei also fp16 500→**780**, mem_bw 1.6→**3.2** (per-package). Maia power 700→**500**.
- **Provenance tags** on every chip; hyperscaler ASIC costs marked `cost imputed`.
- **Retained per user:** Tensordyne, Positron Atlas/Asimov (tagged `vendor-claimed`, numbers untouched).
- **d-Matrix (from user, 2026-06-06):** Corsair modeled SRAM-resident (mem_bw 150 TB/s, mem_cap 2 GB fast
  tier, MXINT compute 0.6/2.4/9.6 PF, switched-PCIe interconnect, rack_size 72). Gen-2 Raptor added (roadmap).
  Verified-spec but **NOT calibrated** — over-predicts at B=1 (35k tok/s) until a tok/s anchor pins comp_eff/bw_eff;
  flips to interconnect-bound at batch. Still need: tok/s anchor, cost, power. Note: "SpecDec + Expert
  disaggregation ready today" — on-thesis for the disagg axes.
- **Pending:** d-Matrix calibration anchor; MoE low-batch bug **M1 still open** (DeepSeek-V3 B=1 now 770 tok/s —
  better but still ~10× high; needs the dedicated fix, not just bw_eff). Two-tier (SRAM/LPDDR) memory model also
  open — affects d-Matrix, Positron, Cerebras.

---

## Part 1 — Math & methodology

### What's correct (verified by reading + anchor tests)
- **Capacity / scale-out math is sound.** Llama-70B fp16 → 141 GB → engine correctly
  requires ≥2× H100; 405B → 11× H100; DeepSeek-V3 671B fp8 → 4× B200. KV-cache sizing
  (`2·d_kv·L·B·T_max`), I:O-ratio-driven `T_max`, and independent KV precision are right.
- **Roofline bind classification is sane** (compute vs weights vs kv vs interconnect).
- **Dense FFN/attention FLOPs are correct.** SwiGLU = 6·B·T·d·d_ffn (up+gate+down); attention
  Q/K/V/O + scores + AV all check out. Decode reads full KV (`T_seq`), prefill processes `T_input`.
- **Ring all-reduce (TP) and all-to-all (EP) volumes** use the standard `2·(N-1)/N` form.

### Anchor-test results (engine output vs published reality)
| Case | Engine | Reality (approx) | Verdict |
|---|---|---|---|
| H100 · 70B · decode B=1 | 47.9 tok/s/user | ~25–35 | optimistic ~1.4–1.9× |
| H100 · 70B · prefill 4k | 0.43 s TTFT | ~0.6–1.0 s @ 45% MFU | optimistic ~1.5× |
| B200 · 70B fp8 · decode B=1 | 114 tok/s/user | ~100–130 | **good** |
| H100 · 405B · decode B=1 | 45 tok/s/user | ~15–20 | optimistic ~2.5× (worse at scale) |
| B200 · DeepSeek-V3 · decode **B=1** | **1100 tok/s/user** | ~20–90 | **~15–50× too high** ⚠ |
| B200 · DeepSeek-V3 · decode B=64 | 43 tok/s/user | plausible | OK |
| Groq · 70B · decode B=1 | 282 tok/s/user | ~280 (calibrated) | matches by construction |
| Cerebras · 70B · decode B=1 | 1856 tok/s/user | ~2100 | closer than expected (N_chips=4 multiplies the 70 TB/s) |

### Issues found (ranked)

**M1 — MoE low-batch throughput is wildly optimistic. [CRITICAL]**
`getStepCost` reads only `experts_touched = min(E, max(K, K·B·T))` experts' weights. At
decode B=1 that's K=8 of 256 experts, so per-token weight traffic collapses to ~active-param
size and DeepSeek-V3 "runs" at 1100 tok/s/user. Real low-batch MoE decode can't exploit this
(routing irregularity, tiny per-expert matmuls at low BW efficiency, expert-load imbalance).
The heuristic also assumes **zero routing collisions** (every token hits distinct experts).
Net: MoE looks far too good at low batch — exactly the regime an interactivity/Pareto tool
lives in. **This is the most consequential math bug for the thesis.** Fix: add an MoE decode
efficiency derate and/or a collision model; floor per-token cost.

**M2 — No MFU / bandwidth-utilization derate. [HIGH, systematic]**
`compute_t = flops/(peak·N)` and `mem_t = bytes/(memBW·N)` assume 100% utilization. Real
prefill MFU ~40–55%; real decode BW util ~70–85%. This makes every absolute number optimistic
(~1.3–2×) and — critically — **biases the prefill↔decode and compute-chip↔bandwidth-chip
comparisons**, because the optimism is larger on the compute axis. Interacts with D-tier data
issue below. Fix: per-precision MFU and BW-efficiency factors (and they differ by chip class).

**M3 — Optimism grows with chip count. [MED]**
405B (11 chips) is ~2.5× off vs 70B (2 chips) ~1.5×. Interconnect/all-reduce latency is
under-biting at scale, or TP sync overhead is too light. The `interconnectBW` selection
(M4) is implicated. Fix: validate all-reduce term against a known 8×/16× TP latency.

**M4 — Intra/inter-rack bandwidth selection may be inverted. [MED, verify]**
`core.js`: `intraRackBW = chip.xc_interconnect_bw || chip.interconnect_bw`. But `xc_` is
documented as **cross-chassis** in the catalogue comments (e.g. Groq 5e12 intra vs xc 100e9).
Using the *cross*-chassis number as the *intra*-rack BW looks backwards and would mis-cost
TP-heavy configs on chips where the two differ (Groq, d-Matrix). Confirm intended semantics.

**M5 — Prefill attention not causal-halved; no FlashAttention. [LOW]**
Prefill scores use full `T²` not `T²/2`. ~2× over-count on attention-compute-bound long context.

**M6 — MLA modeled as low-H_kv GQA. [LOW, acknowledged]**
DeepSeek/Kimi MLA KV compression is approximated via small `H_kv`. Understates MLA's KV win
somewhat; the code comments already flag this. Fine for v1, revisit for the disagg layer
(KV-transfer cost between prefill/decode pools depends heavily on this).

---

## Part 2 — Benchmark catalogue (verified mid-2026, 4 parallel research agents)

### Cross-cutting: peak-vs-sustained is inconsistent [CRITICAL methodology]
The catalogue **derates Groq & Cerebras to sustained-effective** (calibrated to real tok/s) but
uses **raw peak** for NVIDIA/AMD/etc. Combined with M2 (no MFU), every cross-chip comparison is
**biased toward the peak-spec'd chips**. Pick one convention: either sustained-effective for all
(with per-chip MFU) or peak for all. This single choice can flip Pareto winners.

### Straight factual corrections (errors, not modeling choices)
| Chip | Field | Catalogue | Correct | Severity |
|---|---|---|---|---|
| **Tenstorrent Blackhole** | cost_usd | $15,000 | **$1,399** | CRITICAL (10× — flips its $/tok story) |
| Tenstorrent | peak_fp8 | 1490e12 | 664e12 (no FP8 doubling) | HIGH |
| Tenstorrent | peak_fp16 | 745e12 | 664e12 (120-core cut) | MED |
| Tenstorrent | interconnect_bw | 1e12 | ~0.4e12 (4×800GbE) | MED |
| **Huawei 910C** | rack_size | 8 | **384** (CloudMatrix optical domain) | HIGH |
| Huawei 910C | peak_fp16 | 500e12 | ~752–780e12 (dual-die) | MED |
| Huawei 910C | mem_bw | 1.6e12 | ~3.2e12 (per-package) | MED |
| **Google TPU v7** | rack_size | 256 | **9216** (superpod ICI domain) | HIGH |
| **Meta MTIA v2** | peak_fp16 | 177e12 | ~354e12 dense BF16 | MED |
| **Rubin R100** | ship date | "est 2027" | **H2 2026** (2027 = Rubin *Ultra*) | MED |
| Rubin R100 | rack_size | 144 (dies) | 72 (packages) — convention clash w/ 288GB/pkg | HIGH |
| Rubin R100 | peak_fp4 | 20000e12 | ~25000e12 dense (headline 3.6EF is sparse) | MED |
| AWS Trainium2 | peak_fp16 | 658e12 | 667e12 | LOW |

### Derates / fabricated effective-specs to revisit
| Chip | Field | Issue |
|---|---|---|
| **d-Matrix Corsair** | mem_bw 4e12 | ~10× too high for out-of-SRAM 70B. Real tiers: 150 TB/s SRAM (2 GB) / **0.4 TB/s LPDDR** (256 GB). Model as two tiers. |
| Corsair | peak_fp8/fp16 | Mislabeled — it's MXINT (block), not IEEE. MXINT8=2.4e15 (not 3e15), MXINT16=0.6e15 (not 1.5e15). |
| **Positron Asimov** | mem_bw 8e12 | Vendor realizable = **2.76e12**; the 8 TB/s tiering figure is unsupported. |
| Asimov | mem_cap 2e12 | Native fast = **864 GB**; the rest is cold CXL tier (~256 GB/s). 2 TB-as-fast overstates ~2.3×. |
| **Tensordyne** | peak (2.1e15 ×3) | **No public per-chip FLOPS exists** — fabricated. 2.1 PF @ 350 W ⇒ ~6 PF/W, a ~5–10× efficiency outlier. Tag estimate-only. |
| Tensordyne | mem_cap 96e9 | Reported **144 GB** HBM3e. Reconcile. |
| **Cerebras WSE-3** | provenance | "derated from 125 PF" → should be **12.5 PF dense** (125 is 10× sparse). 3 PF value itself ≈ OK. |
| Groq LPU | mem_bw 0.065e12 | Calibration constant (fit to 70B ~280 tok/s), **not transferable** across model sizes; also peak_fp16 (20→188) and interconnect (5e12→0.33e12) are wrong for this gen. |

### "Not a price" flag — imputed costs [HIGH for a $/tok thesis tool]
TPU v7, Trainium2, Maia 100, MTIA v2 are **never sold** — cloud/internal only. Their `cost_usd`
are modeling proxies, not market prices. Cerebras/Positron/Tensordyne/d-Matrix have **no public
price** either. For a $/token thesis instrument, these must be surfaced as assumptions (and ideally
the cost layer should support a rental/$-per-hour mode for the hyperscaler ASICs).

### Confirmed correct (don't "fix")
- H100 / B200 all FLOPS fields use **dense** numbers (avoided the 2× sparse trap). ✓
- H100 FP4 = null (Hopper has no FP4). ✓ · B200 FP4 = 9000e12 dense ✓
- **Gaudi 3 fp16 == fp8 == 1835e12 is REAL** (shared MAC array, no FP8 doubling) — not a bug. ✓
- AMD MI355X, Trainium2, Ironwood correctly use dense where vendors headline sparse. ✓
- Cerebras 44 GB cap, Groq 230 MB cap, MTIA LPDDR 204 GB/s — all correct & physically important.

---

## Part 3 — Prioritized fix queue
1. **[M1]** MoE low-batch decode derate (biggest thesis risk).
2. **[Methodology]** Decide peak-vs-sustained convention; add per-chip-class MFU + BW-eff (M2).
3. **[Data]** Tenstorrent cost 10×; Corsair/Asimov/Tensordyne effective-spec corrections.
4. **[Data]** rack_size fixes (Huawei 384, TPU 9216, Rubin 72) — large impact on scale-up modeling.
5. **[M4]** Verify intra/inter-rack BW selection isn't inverted.
6. **[Cost layer]** Separate "imputed/rental" vs "street price"; support $/hr mode for ASICs.
7. **[M3/M5/M6]** Interconnect-at-scale, causal halving, MLA — refine when building the disagg layer.
