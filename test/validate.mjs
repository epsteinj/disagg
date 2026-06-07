// Validation / double-check pass. Three independent checks:
//  (1) Catalogue consistency lint (units, monotonic precision, tier consistency, sane ranges).
//  (2) Closed-form cross-check: for weight-bandwidth-bound decode, tok/s MUST ≈ mem_bw·bw_eff·N / model_bytes.
//      This is first-principles and INDEPENDENT of the engine's per-step loop — if they disagree, a bug exists.
//  (3) Reality-band anchors: engine output vs published benchmark ranges.
// Run: node test/validate.mjs
import { chipPerfSpecs, modelPresets, computeChipSummary, _bytesPerValOf } from '../src/engine/core.js';

let warns = 0, fails = 0, passes = 0;
const W = (m) => { console.log('  ⚠ ' + m); warns++; };
const FAIL = (m) => { console.log('  ✗ ' + m); fails++; };
const OK = (m) => { console.log('  ✓ ' + m); passes++; };

// ── (1) Catalogue consistency ──
console.log('\n(1) CATALOGUE CONSISTENCY');
const EQUAL_OK = new Set(['intel-gaudi3']);          // fp16==fp8 is real here
const CALIB = new Set(['groq-lpu', 'cerebras-wse3']); // mem_bw is a calibrated effective constant
for (const [k, c] of Object.entries(chipPerfSpecs)) {
  if (!(c.mem_bw > 0) || !(c.mem_cap > 0)) FAIL(`${k}: mem_bw/mem_cap must be >0`);
  if (!(c.cost_usd >= 1e3 && c.cost_usd <= 3e6)) W(`${k}: cost_usd ${c.cost_usd} outside [1e3,3e6]`);
  if (!(c.power_w >= 50 && c.power_w <= 30000)) W(`${k}: power_w ${c.power_w} outside [50,30000]`);
  if (!(c.comp_eff > 0 && c.comp_eff <= 1)) FAIL(`${k}: comp_eff ${c.comp_eff} not in (0,1]`);
  if (!(c.bw_eff > 0 && c.bw_eff <= 1)) FAIL(`${k}: bw_eff ${c.bw_eff} not in (0,1]`);
  if (c.peak_fp8 && c.peak_fp16 && c.peak_fp8 < c.peak_fp16 && !EQUAL_OK.has(k))
    W(`${k}: peak_fp8 < peak_fp16 (unusual)`);
  if (c.peak_fp4 && c.peak_fp8 && c.peak_fp4 < c.peak_fp8) W(`${k}: peak_fp4 < peak_fp8 (unusual)`);
  const hasColdBW = !!c.mem_bw_cold, hasColdCap = !!c.mem_cap_cold;
  if (hasColdBW !== hasColdCap) W(`${k}: cold tier half-specified (bw_cold=${c.mem_bw_cold} cap_cold=${c.mem_cap_cold})`);
  if (c.weight_tier && c.weight_tier !== 'fast') W(`${k}: unknown weight_tier '${c.weight_tier}'`);
  const ai = c.peak_fp16 / c.mem_bw; // roofline crossover (FLOP/byte)
  if (!CALIB.has(k) && (ai < 50 || ai > 2000)) W(`${k}: roofline AI ${ai.toFixed(0)} FLOP/byte unusual (typical 150–700)`);
}
if (!fails) OK(`${Object.keys(chipPerfSpecs).length} chips passed structural checks (${warns} soft warnings)`);

// ── (2) Closed-form decode cross-check (independent of the per-step loop) ──
console.log('\n(2) CLOSED-FORM CROSS-CHECK  (decode tok/s ≈ mem_bw·bw_eff·N / model_bytes, B=1)');
const xcheck = [
  ['nvidia-h100',  'llama-3-70b',  'fp16'],
  ['nvidia-b200',  'llama-3-70b',  'fp8'],
  ['nvidia-h100',  'llama-3-405b', 'fp16'],
  ['amd-mi355x',   'llama-3-70b',  'fp8'],
  ['groq-lpu',     'llama-3-70b',  'fp16'],
  ['cerebras-wse3','llama-3-70b',  'fp16'],
];
for (const [ck, mk, prec] of xcheck) {
  const c = chipPerfSpecs[ck], m = modelPresets[mk];
  const r = computeChipSummary(ck, m, 1, 4096, prec, 'llm', 'decode', 50, null, 'auto', 1);
  const bpv = _bytesPerValOf(r.effPrec);
  const closedForm = (c.mem_bw * c.bw_eff * r.N_chips) / (m.N_params * bpv); // tok/s, weight-bound ideal
  const engine = 1 / r.totalLayerTime;
  const ratio = engine / closedForm;
  const msg = `${ck.padEnd(15)} ${mk.padEnd(13)} engine ${engine.toFixed(1)}  closed-form ${closedForm.toFixed(1)}  (×${ratio.toFixed(2)})`;
  // engine should be ≤ closed-form (KV/attention/interconnect only ADD time) and within ~25%.
  if (ratio > 1.05) FAIL(msg + '  [engine FASTER than weight-bound limit — impossible]');
  else if (ratio < 0.6) W(msg + '  [>40% below limit — KV/interconnect heavy, expected for long ctx]');
  else OK(msg);
}

// ── (3) Reality-band anchors ──
console.log('\n(3) REALITY-BAND ANCHORS  (engine vs published ranges)');
const bands = [
  // chip, model, workload, phase, B, prec, [lo, hi] expected, metric
  ['nvidia-h100',  'llama-3-70b',  'llm', 'decode',  1, 'fp16', [20, 45],   'per-user tok/s'],
  ['nvidia-b200',  'llama-3-70b',  'llm', 'decode',  1, 'fp8',  [60, 140],  'per-user tok/s'],
  ['nvidia-h100',  'llama-3-405b', 'llm', 'decode',  1, 'fp16', [10, 35],   'per-user tok/s'],
  ['nvidia-b200',  'deepseek-v3',  'moe', 'decode',  1, 'fp8',  [15, 100],  'per-user tok/s (MoE B=1)'],
  ['nvidia-b200',  'deepseek-v3',  'moe', 'decode', 64, 'fp8',  [15, 70],   'per-user tok/s (MoE B=64)'],
  ['groq-lpu',     'llama-3-70b',  'llm', 'decode',  1, 'fp16', [200, 350], 'per-user tok/s'],
  ['cerebras-wse3','llama-3-70b',  'llm', 'decode',  1, 'fp16', [1500, 2600],'per-user tok/s'],
  ['nvidia-h100',  'llama-3-70b',  'llm', 'prefill', 1, 'fp16', [0.4, 1.2], 'TTFT s (4k)'],
];
for (const [ck, mk, wl, ph, B, prec, [lo, hi], label] of bands) {
  const r = computeChipSummary(ck, modelPresets[mk], B, 4096, prec, wl, ph, 50, null, 'auto', 1);
  const v = ph === 'prefill' ? r.totalLayerTime : 1 / r.totalLayerTime;
  const inBand = v >= lo && v <= hi;
  const msg = `${ck.padEnd(15)} ${mk.padEnd(13)} ${label.padEnd(24)} = ${v.toFixed(ph==='prefill'?3:1)}  [${lo}–${hi}]`;
  inBand ? OK(msg) : W(msg + '  OUT OF BAND');
}

console.log(`\n── ${passes} passed · ${warns} warnings · ${fails} failures ──`);
process.exit(fails ? 1 : 0);
