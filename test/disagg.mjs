// Disaggregation sweep demo: does heterogeneous prefill/decode beat a homogeneous fleet?
// Run: node test/disagg.mjs
import { sweepDisagg, paretoFrontier, computeDisaggPoint } from '../src/engine/disagg.js';

const MODEL = 'llama-3-70b';
const PREFILL = ['nvidia-b200', 'nvidia-h100', 'amd-mi355x'];           // compute-dense candidates
const DECODE  = ['nvidia-b200', 'nvidia-h100', 'groq-lpu', 'dmatrix-corsair']; // bandwidth/latency-dense
const BATCHES = [1, 4, 16, 64, 256];
const T_in = 4096, T_out = 512;

const pts = sweepDisagg({ modelKey: MODEL, prefillChips: PREFILL, decodeChips: DECODE, batches: BATCHES, T_in, T_out, precision: 'fp8' });

const pad = (s, n) => String(s).padEnd(n);
const padL = (s, n) => String(s).padStart(n);
const row = p => [
  pad(p.heterogeneous ? 'HET' : 'hom', 4),
  pad(p.prefillChip.replace(/^(nvidia|amd|dmatrix|groq)-/, ''), 9),
  pad('→ ' + p.decodeChip.replace(/^(nvidia|amd|dmatrix|groq)-/, ''), 11),
  padL(p.Bd, 4),
  padL(p.interactivity.toFixed(1), 9),
  padL(p.throughput.toFixed(0), 10),
  padL('$' + p.costPerMtok.toFixed(3), 10),
  padL(p.ttft.toFixed(3) + 's', 8),
  pad(p.decodeBind, 7),
].join(' ');

console.log(`\n=== ${MODEL} · disaggregated prefill/decode · T_in=${T_in} T_out=${T_out} ===`);
console.log(`Swept ${pts.length} (prefill × decode × batch) configs\n`);
console.log(pad('', 4), pad('prefill', 9), pad('decode', 11), padL('Bd', 4), padL('per-user', 9), padL('throughput', 10), padL('$/Mtok', 10), padL('TTFT', 8), 'bind');
console.log('-'.repeat(80));

// Pareto frontier on throughput × interactivity.
const frontier = paretoFrontier(pts, { xKey: 'throughput', yKey: 'interactivity' })
  .sort((a, b) => b.interactivity - a.interactivity);
console.log('PARETO FRONTIER (throughput × interactivity):');
for (const p of frontier) console.log(row(p));

// Thesis question: at a chat-grade interactivity target (~30 tok/s/user), what's the cheapest config,
// and is it heterogeneous?
console.log('\nCheapest config at ≥30 tok/s/user interactivity (chat-grade):');
const chat = pts.filter(p => p.interactivity >= 30).sort((a, b) => a.costPerMtok - b.costPerMtok);
for (const p of chat.slice(0, 5)) console.log(row(p));

// Homogeneous baseline vs best heterogeneous at the same interactivity band.
const band = pts.filter(p => p.interactivity >= 25 && p.interactivity <= 45);
const bestHom = band.filter(p => !p.heterogeneous).sort((a, b) => a.costPerMtok - b.costPerMtok)[0];
const bestHet = band.filter(p => p.heterogeneous).sort((a, b) => a.costPerMtok - b.costPerMtok)[0];
console.log('\nHomogeneous vs heterogeneous in the 25–45 tok/s/user band (by $/Mtok):');
if (bestHom) console.log('  best homogeneous:  ', row(bestHom));
if (bestHet) console.log('  best heterogeneous:', row(bestHet));
if (bestHom && bestHet) {
  const d = (1 - bestHet.costPerMtok / bestHom.costPerMtok) * 100;
  console.log(`  → heterogeneous is ${d > 0 ? d.toFixed(0) + '% cheaper' : (-d).toFixed(0) + '% more expensive'} at matched interactivity`);
}
