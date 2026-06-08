// Chip catalogue — forked from transformer_math.html (ref lines 8226–8278) and corrected
// against mid-2026 public sources (see audit/AUDIT.md). Separated from engine logic so every
// correction carries provenance and the thesis output can be honest about confidence.
//
// Convention: SUSTAINED-EFFECTIVE for ALL chips (per design decision 2026-06-06).
//   comp_eff = realized fraction of peak FLOP/s (MFU) on compute-bound steps
//   bw_eff   = realized fraction of peak HBM bandwidth on memory-bound steps
//   GPU/ASIC default: comp_eff 0.50, bw_eff 0.70 (calibrated so H100 70B decode ≈ 34 tok/s/user,
//     4k prefill ≈ 0.86 s — both in the published range).
//   Groq & Cerebras: comp_eff = bw_eff = 1.0 because their mem_bw is ALREADY a calibrated
//     effective constant (reproduces real ~280 / ~2,100 tok/s on Llama-70B). Applying the default
//     derate on top would double-count.
//
// prov tags: 'verified' | 'verified-med' | 'roadmap-estimate' | 'calibrated-effective'
//            | 'vendor-claimed' | 'pending-user-data'
//
// Units: peak_* FLOP/s · mem_bw byte/s · mem_cap bytes · *interconnect_bw byte/s · cost_usd/chip · power_w/chip
// rack_size = largest PRACTICAL single-job fast-fabric domain (not theoretical max).
// xc_interconnect_bw: cross-chassis bandwidth once you scale past one fabric domain.

export const chipPerfSpecs = {
  // ── NVIDIA ──────────────────────────────────────────────────────────────────
  'nvidia-h100':       { peak_fp16: 989e12,    peak_fp8: 1979e12,  peak_fp4: null,      mem_bw: 3.35e12,  mem_cap: 80e9,    interconnect_bw: 900e9,    xc_interconnect_bw: 900e9,    rack_size: 8,    scaleout_bw: 100e9,  cost_usd: 30000,   power_w: 700,    comp_eff: 0.50, bw_eff: 0.70, prov: 'verified',            name: 'NVIDIA H100',           ic_name: 'NVLink 4' },
  // NVL72 rack product = 72-GPU NVLink domain. (Standalone HGX B200 baseboard is only 8-GPU.)
  'nvidia-b200':       { peak_fp16: 2250e12,   peak_fp8: 4500e12,  peak_fp4: 9000e12,   mem_bw: 8.0e12,   mem_cap: 192e9,   interconnect_bw: 1.8e12,   xc_interconnect_bw: 1.8e12,   rack_size: 72,   scaleout_bw: 100e9,  cost_usd: 40000,   power_w: 1000,   comp_eff: 0.50, bw_eff: 0.70, prov: 'verified',            name: 'NVIDIA B200',           ic_name: 'NVLink 5 / NVL72' },
  // CORRECTED: ship date 2027→H2 2026 (2027 = Rubin *Ultra*, a bigger part); rack_size 144→72
  //   (NVL144 = 144 *dies* = 72 *packages*; mem_cap 288e9 is per-package, so the domain is 72 packages);
  //   peak_fp4 20000e12→25000e12 (NVIDIA's 3.6 EF/rack headline is SPARSE; dense ≈ 25 PF/package).
  //   peak_fp8 10000e12 left as-is but low-confidence (disclosures imply ~16 PF/package dense).
  'nvidia-rubin-r100': { peak_fp16: 5000e12,   peak_fp8: 10000e12, peak_fp4: 25000e12,  mem_bw: 14e12,    mem_cap: 288e9,   interconnect_bw: 3.6e12,   xc_interconnect_bw: 3.6e12,   rack_size: 72,   scaleout_bw: 200e9,  cost_usd: 60000,   power_w: 1800,   comp_eff: 0.50, bw_eff: 0.70, prov: 'roadmap-estimate',    name: 'NVIDIA Rubin R100 (est H2 2026)', ic_name: 'NVLink 6 / NVL144' },

  // ── AMD / Intel ─────────────────────────────────────────────────────────────
  'amd-mi355x':        { peak_fp16: 1310e12,   peak_fp8: 2622e12,  peak_fp4: 5245e12,   mem_bw: 8e12,     mem_cap: 288e9,   interconnect_bw: 896e9,    xc_interconnect_bw: 896e9,    rack_size: 8,    scaleout_bw: 100e9,  cost_usd: 25000,   power_w: 1400,   comp_eff: 0.50, bw_eff: 0.70, prov: 'verified',            name: 'AMD MI355X',            ic_name: 'Infinity Fabric' },
  // peak_fp16 == peak_fp8 is CORRECT (intentional): Gaudi 3 runs BF16 and FP8 through the same MME array, no FP8 doubling.
  'intel-gaudi3':      { peak_fp16: 1835e12,   peak_fp8: 1835e12,  peak_fp4: null,      mem_bw: 3.7e12,   mem_cap: 128e9,   interconnect_bw: 1.2e12,   xc_interconnect_bw: 1.2e12,   rack_size: 8,    scaleout_bw: 200e9,  cost_usd: 15000,   power_w: 900,    comp_eff: 0.50, bw_eff: 0.70, prov: 'verified',            name: 'Intel Gaudi 3',         ic_name: '24× 200 GbE RoCE' },

  // ── China / Hyperscaler ASICs ───────────────────────────────────────────────
  // CORRECTED: fp16 500e12→780e12 (dual-die package; CloudMatrix 300PF/384≈781TF/chip); mem_bw 1.6e12→3.2e12
  //   (per-package, not per-die); rack_size 8→384 (CloudMatrix 384 optical all-to-all is Huawei's scale-up unit).
  'huawei-ascend910c': { peak_fp16: 780e12,    peak_fp8: null,     peak_fp4: null,      mem_bw: 3.2e12,   mem_cap: 96e9,    interconnect_bw: 392e9,    xc_interconnect_bw: 392e9,    rack_size: 384,  scaleout_bw: 100e9,  cost_usd: 20000,   power_w: 700,    comp_eff: 0.50, bw_eff: 0.70, prov: 'verified-med',        name: 'Huawei Ascend 910C',    ic_name: 'HCCS / CloudMatrix' },
  // cost_usd imputed: TPUs are never sold (GCP HaaS). rack_size 256 = practical slice (full superpod = 9216).
  'google-tpu7':       { peak_fp16: 2307e12,   peak_fp8: 4614e12,  peak_fp4: null,      mem_bw: 7.37e12,  mem_cap: 192e9,   interconnect_bw: 1.2e12,   xc_interconnect_bw: 1.2e12,   rack_size: 256,  scaleout_bw: 400e9,  cost_usd: 30000,   power_w: 1100,   comp_eff: 0.50, bw_eff: 0.70, prov: 'verified; cost imputed', name: 'Google TPU Ironwood',  ic_name: 'ICI 3D torus + OCS' },
  // CORRECTED: fp16 658e12→667e12. cost_usd imputed (AWS-internal, cloud-only).
  'aws-trainium2':     { peak_fp16: 667e12,    peak_fp8: 1316e12,  peak_fp4: null,      mem_bw: 2.9e12,   mem_cap: 96e9,    interconnect_bw: 1e12,     xc_interconnect_bw: 1e12,     rack_size: 64,   scaleout_bw: 200e9,  cost_usd: 20000,   power_w: 500,    comp_eff: 0.50, bw_eff: 0.70, prov: 'verified; cost imputed', name: 'AWS Trainium 2',       ic_name: 'NeuronLink UltraServer' },
  // CORRECTED: power_w 700→500 (provisioned; 700 is max TDP). fp8 is really MX9/MX6, not IEEE FP8. cost imputed.
  'microsoft-maia100': { peak_fp16: 800e12,    peak_fp8: 1600e12,  peak_fp4: null,      mem_bw: 1.8e12,   mem_cap: 64e9,    interconnect_bw: 600e9,    xc_interconnect_bw: 600e9,    rack_size: 8,    scaleout_bw: 100e9,  cost_usd: 12000,   power_w: 500,    comp_eff: 0.50, bw_eff: 0.70, prov: 'verified; cost imputed', name: 'Microsoft Maia 100',   ic_name: 'RoCE-like' },
  // CORRECTED: fp16 177e12→354e12 (dense BF16; 177 was a half/sparse confusion). cost imputed (Meta-internal).
  'meta-mtia2':        { peak_fp16: 354e12,    peak_fp8: null,     peak_fp4: null,      mem_bw: 0.2048e12, mem_cap: 128e9,  interconnect_bw: 100e9,    xc_interconnect_bw: 100e9,    rack_size: 8,    scaleout_bw: 100e9,  cost_usd: 8000,    power_w: 90,     comp_eff: 0.50, bw_eff: 0.70, prov: 'verified; cost imputed', name: 'Meta MTIA v2',         ic_name: 'PCIe + Ethernet' },

  // ── Inference incumbents ────────────────────────────────────────────────────
  // CORRECTED (biggest factual errors in catalogue): cost 15000→1399 (p150a MSRP, 10×!); fp8 1490e12→664e12
  //   (no FP8 doubling — FP8==BF16 on Blackhole); fp16 745e12→664e12 (120-core cut); interconnect 1e12→0.4e12 (4×800GbE).
  'tt-blackhole':      { peak_fp16: 664e12,    peak_fp8: 664e12,   peak_fp4: null,      mem_bw: 0.512e12, mem_cap: 32e9,    interconnect_bw: 0.4e12,   xc_interconnect_bw: 0.4e12,   rack_size: 8,    scaleout_bw: 200e9,  cost_usd: 1399,    power_w: 300,    comp_eff: 0.50, bw_eff: 0.70, prov: 'verified',            name: 'Tenstorrent Blackhole p150', ic_name: '4× 800GbE QSFP-DD' },
  // mem_bw 0.065e12 is a CALIBRATED EFFECTIVE constant (reproduces Groq's ~280 tok/s/user on Llama-70B,
  //   ArtificialAnalysis). Real SRAM rate is ~80 TB/s but Groq's pipeline-parallelism makes per-user rate
  //   nearly flat in batch — NOT roofline-shaped. This constant only holds near the 70B operating point.
  //   comp_eff=bw_eff=1.0 to avoid double-derate. CORRECTED for correctness (unused in calibrated path):
  //   peak_fp16 20e12→188e12; interconnect 5e12→0.33e12 (GroqChip-1 RealScale, was a next-gen number).
  'groq-lpu':          { peak_fp16: 188e12,    peak_fp8: null,     peak_fp4: null,      mem_bw: 0.065e12, mem_cap: 0.23e9,  interconnect_bw: 0.33e12,  xc_interconnect_bw: 100e9,    rack_size: 72,   scaleout_bw: 50e9,   cost_usd: 20000,   power_w: 215,    comp_eff: 1.0,  bw_eff: 1.0,  prov: 'calibrated-effective', name: 'Groq LPU',             ic_name: 'RealScale (pipeline)' },
  // mem_bw 70e12 and peak 3e15 are CALIBRATED EFFECTIVE. Note: 3 PF is ~24% MFU of the honest 12.5 PF DENSE
  //   peak (NOT a derate of the 125 PF *sparse* marketing figure, as the original comment claimed).
  //   With autofit N_chips=4 for 70B, effective BW ×4 lands ≈ 1,856 tok/s vs published ~2,100 — good.
  //   comp_eff=bw_eff=1.0 (already calibrated).
  'cerebras-wse3':     { peak_fp16: 3e15,      peak_fp8: null,     peak_fp4: null,      mem_bw: 70e12,    mem_cap: 44e9,    interconnect_bw: 70e12,    xc_interconnect_bw: 100e9,    rack_size: 16,   scaleout_bw: 50e9,   cost_usd: 2500000, power_w: 23000,  comp_eff: 1.0,  bw_eff: 1.0,  prov: 'calibrated-effective', name: 'Cerebras WSE-3',       ic_name: 'wafer mesh / SwarmX' },

  // ── Venture-stage (RETAINED per user 2026-06-06 — numbers untouched, flagged unverified) ──
  // Atlas = 8× Altera Agilex-7M FPGA server. mem_bw/cap/power solid; peak_fp16 & cost are estimates.
  'positron-atlas':    { peak_fp16: 480e12,    peak_fp8: null,     peak_fp4: null,      mem_bw: 6.5e12,   mem_cap: 256e9,   interconnect_bw: 64e9,     xc_interconnect_bw: 64e9,     rack_size: 8,    scaleout_bw: 50e9,   cost_usd: 175000,  power_w: 2000,   comp_eff: 0.50, bw_eff: 0.70, prov: 'vendor-claimed',      name: 'Positron Atlas',        ic_name: 'PCIe Gen5' },
  // NOTE (unapplied, per user): public data suggests mem_bw ~2.76e12 (not 8e12) and 864 GB native fast (not 2 TB).
  'positron-asimov':   { peak_fp16: 512e12,    peak_fp8: 1024e12,  peak_fp4: null,      mem_bw: 8e12,     mem_cap: 2e12,    interconnect_bw: 2e12,     xc_interconnect_bw: 2e12,     rack_size: 4,    scaleout_bw: 100e9,  cost_usd: 20000,   power_w: 400,    comp_eff: 0.50, bw_eff: 0.70, prov: 'vendor-claimed',      name: 'Positron Asimov',       ic_name: 'Credo 16 Tb/s · EWM+LPDDR5x tiered' },
  // NOTE (unapplied, per user): 2.1 PFLOPS (×3 precisions) has no public source; mem_cap may be 144 GB (not 96).
  'tensordyne':        { peak_fp16: 2.1e15,    peak_fp8: 2.1e15,   peak_fp4: 2.1e15,    mem_bw: 4.2e12,   mem_cap: 96e9,    interconnect_bw: 500e9,    xc_interconnect_bw: 500e9,    rack_size: 144,  scaleout_bw: 100e9,  cost_usd: 25000,   power_w: 350,    comp_eff: 0.50, bw_eff: 0.70, prov: 'vendor-claimed',      name: 'Tensordyne',            ic_name: 'Juniper cell fabric' },

  // ── d-Matrix Corsair (Gen 1) — CALIBRATED to vendor anchor (2026 research) ──
  // ANCHOR (d-Matrix SC24, Nov 2024; EE Times): Llama3-70B at ~500 tok/s/user (2 ms/token) and ~30,000 tok/s
  //   AGGREGATE on a 64-CARD RACK at batch 48–64, MXINT8. (Llama3-8B: 60,000 tok/s / 1 ms/token on an 8-card
  //   server.) Vendor-claimed, no MLPerf — but internally consistent: 500 tok/s/user × ~60 streams ≈ 30k.
  // REGIME: SRAM-RESIDENT — weights AND KV pinned in 2 GB/card SRAM, fanned across the rack; the batch ceiling
  //   (~48–64) IS the SRAM-capacity limit. So mem_cap = 2e9 gates cards (weights+KV must fit SRAM) and KV uses
  //   the SRAM tier — the 256 GB LPDDR cold tier is REMOVED here: "capacity mode" (weights/KV in LPDDR) is a
  //   separate ~10–80× slower path (~5–45 tok/s/user), not the competitive operating point we model.
  // bw_eff = 0.0078 is a CALIBRATED-EFFECTIVE constant (Groq/Cerebras-style): the realized ~500 tok/s/user is
  //   gated by interconnect + pipeline depth across the ~36–80 sharded cards, NOT the raw 150 TB/s SRAM.
  //   Reproduces ~520 tok/s/user and ~26–33k aggregate for 70B at batch 48–64 (and ~720/user, ~43k for 8B).
  //   Intra-rack fabric set to DMX 1 TB/s with rack_size 128 so the operating deployment stays one domain.
  //   Compute = MXINT (0.6/2.4/9.6 PF). NOTE: this takes the vendor headline at face value — The Register
  //   argues the real JetStream/Ethernet scale-out (~100 GB/s/node) would degrade per-user further at scale.
  // STILL ESTIMATE: cost_usd (no public price); power 550 W (real 275 W @ 0.8 GHz – 550 W @ 1.2 GHz).
  'dmatrix-corsair':   { peak_fp16: 0.6e15,    peak_fp8: 2.4e15,   peak_fp4: 9.6e15,    mem_bw: 150e12,   mem_cap: 2e9,     weight_tier: 'fast', interconnect_bw: 1e12, xc_interconnect_bw: 1e12, rack_size: 128,  scaleout_bw: 100e9,  cost_usd: 38000,   power_w: 550,    comp_eff: 1.0, bw_eff: 0.0078, prov: 'calibrated-effective (vendor, no MLPerf); cost est', name: 'd-Matrix Corsair', ic_name: 'DMX die-to-die (1 TB/s) + ESUN scale-out' },
  // ── d-Matrix Raptor (Gen 2, NEAR-TERM, 3D-DRAM) ── roadmap. Slide: "32 GB/card · ~100 TB/s via 3D-DRAM",
  //   "2T+ capacity single deployment". No Gen-2 anchor — bw_eff extrapolated from Corsair's calibration (the
  //   per-user effective rate is interconnect/pipeline-gated, so similar); 32 GB SRAM cuts card count vs Corsair.
  'dmatrix-raptor':    { peak_fp16: 0.6e15,    peak_fp8: 2.4e15,   peak_fp4: 9.6e15,    mem_bw: 100e12,   mem_cap: 32e9,    weight_tier: 'fast', interconnect_bw: 0.128e12, xc_interconnect_bw: 0.128e12, rack_size: 72,   scaleout_bw: 100e9,  cost_usd: 38000,   power_w: 600,    comp_eff: 1.0, bw_eff: 0.04, prov: 'roadmap; bw_eff extrapolated from Corsair', name: 'd-Matrix Raptor (3D-DRAM, est)', ic_name: '3D-DRAM + switched PCIe + ESUN' },
};
