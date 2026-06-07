// Anchor tests: does the forked engine reproduce known, published benchmark points?
// If a formula is wrong, the absolute numbers will be implausible vs reality.
// Run: node test/anchors.mjs
import { chipPerfSpecs, modelPresets, computeChipSummary } from '../src/engine/core.js';

const AMORT = 3 * 365 * 24 * 3600; // 3-year capex amortization window (s)

function costPerMtok(r) {
  const sysCost = r.N_chips * r.chip.cost_usd;
  const dollarsPerSec = sysCost / AMORT;
  return r.topValue > 0 ? (dollarsPerSec / r.topValue) * 1e6 : Infinity;
}

// (chipKey, modelKey, workload, phase, B, T, precision, kvPrec, ioRatio)
const CASES = [
  ['nvidia-h100',  'llama-3-70b',  'llm',    'decode',   1, 4096, 'fp16', 'auto', 1],
  ['nvidia-h100',  'llama-3-70b',  'llm',    'decode',  32, 4096, 'fp16', 'auto', 1],
  ['nvidia-h100',  'llama-3-70b',  'llm',    'prefill',  1, 4096, 'fp16', 'auto', 1],
  ['nvidia-b200',  'llama-3-70b',  'llm',    'decode',   1, 4096, 'fp8',  'auto', 1],
  ['nvidia-b200',  'llama-3-70b',  'llm',    'decode',  64, 4096, 'fp8',  'auto', 1],
  ['nvidia-h100',  'llama-3-405b', 'llm',    'decode',   1, 4096, 'fp16', 'auto', 1],
  ['nvidia-b200',  'deepseek-v3',  'moe',    'decode',   1, 4096, 'fp8',  'auto', 1],
  ['nvidia-b200',  'deepseek-v3',  'moe',    'decode',  64, 4096, 'fp8',  'auto', 1],
  ['groq-lpu',     'llama-3-70b',  'llm',    'decode',   1, 4096, 'fp16', 'auto', 1],
  ['cerebras-wse3','llama-3-70b',  'llm',    'decode',   1, 4096, 'fp16', 'auto', 1],
  // d-Matrix: specs verified, NOT yet calibrated — expect implausible (over-predicted) tok/s until anchored.
  ['dmatrix-corsair','llama-3-70b','llm',    'decode',   1, 4096, 'fp4',  'auto', 1],
  ['dmatrix-corsair','llama-3-70b','llm',    'decode',  32, 4096, 'fp4',  'auto', 1],
  ['dmatrix-raptor', 'llama-3-70b','llm',    'decode',   1, 4096, 'fp4',  'auto', 1],
];

const pad = (s, n) => String(s).padEnd(n);
const padL = (s, n) => String(s).padStart(n);
console.log(pad('chip', 16), pad('model', 14), pad('phase', 8), padL('B', 4),
  padL('Nchip', 6), pad('bind', 12), padL('per-user t/s', 13), padL('aggr t/s', 12),
  padL('TTFT', 9), padL('$/Mtok', 9));
console.log('-'.repeat(118));

for (const [ck, mk, wl, ph, B, T, prec, kvp, io] of CASES) {
  const model = modelPresets[mk];
  const r = computeChipSummary(ck, model, B, T, prec, wl, ph, 50, null, kvp, io);
  if (!r) { console.log('NULL for', ck, mk); continue; }
  const perUser = ph === 'decode' ? (1 / r.totalLayerTime) : '';
  const aggr = ph === 'decode' ? r.topValue : '';
  const ttft = ph === 'prefill' ? r.topValue : '';
  console.log(
    pad(ck, 16), pad(mk, 14), pad(ph, 8), padL(B, 4),
    padL(r.N_chips, 6), pad(r.overallBinds, 12),
    padL(perUser === '' ? '' : perUser.toFixed(1), 13),
    padL(aggr === '' ? '' : aggr.toFixed(0), 12),
    padL(ttft === '' ? '' : ttft.toFixed(3) + 's', 9),
    padL(ph === 'decode' ? costPerMtok(r).toFixed(3) : '', 9)
  );
}

// Hard capacity assertions (these are physics, not benchmarks).
console.log('\n--- capacity sanity ---');
const l70 = modelPresets['llama-3-70b'];
const w70_fp16 = l70.N_params * 2 / 1e9;
console.log(`Llama-70B fp16 weights = ${w70_fp16.toFixed(0)} GB; H100=80GB -> expect >=2 chips`);
const r2 = computeChipSummary('nvidia-h100', l70, 1, 4096, 'fp16', 'llm', 'decode', 50, null, 'auto', 1);
console.log(`  engine N_chips @ B=1: ${r2.N_chips}  (${r2.N_chips >= 2 ? 'OK' : 'FAIL'})`);
