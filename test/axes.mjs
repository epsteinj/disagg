// Regression + sanity for the disaggregation axes. Run: node test/axes.mjs
import { computeAttnExpertPoint, computeSpecDecodePoint, computeWideEPPoint, computeEPDPoint, sweep } from '../src/engine/disagg.js';
import { computeChipSummary, modelPresets } from '../src/engine/core.js';

let fails = 0;
const ok = (c, m) => { console.log((c ? '  ✓ ' : '  ✗ ') + m); if (!c) fails++; };

console.log('\nATTENTION / EXPERT (MoE)');
// Homogeneous attn+expert on one chip must ≈ plain MoE decode on that chip (+ small transfer overhead),
// and never be FASTER than plain (disaggregation only adds cost).
const plain = 1 / computeChipSummary('nvidia-b200', modelPresets['deepseek-v3'], 32, 4096, 'fp8', 'moe', 'decode', 50, null, 'auto', 8).totalLayerTime;
const ae = computeAttnExpertPoint({ modelKey: 'deepseek-v3', attnChip: 'nvidia-b200', expertChip: 'nvidia-b200', Bd: 32, T_in: 4096, T_out: 512, precision: 'fp8' });
ok(ae && ae.interactivity <= plain * 1.02, `homogeneous attn/expert (${ae.interactivity.toFixed(1)}) ≤ plain decode (${plain.toFixed(1)})`);
ok(ae && ae.interactivity > plain * 0.4, `homogeneous attn/expert within 2.5× of plain (not absurdly slow)`);
ok(ae && ae.transferFrac >= 0 && ae.transferFrac < 1, `transfer fraction in [0,1): ${(ae.transferFrac*100).toFixed(1)}%`);
ok(computeAttnExpertPoint({ modelKey: 'llama-3-70b', attnChip: 'nvidia-b200', expertChip: 'nvidia-b200', Bd: 8 }) === null, 'returns null for dense (non-MoE) model');

console.log('\nSPECULATIVE DECODING');
const sd = computeSpecDecodePoint({ modelKey: 'llama-3-70b', draftChip: 'nvidia-b200', targetChip: 'nvidia-b200', draftModelKey: 'llama-3-2-1b', Bd: 16, K: 4, alpha: 0.7, precision: 'fp8' });
const nAcc = (1 - Math.pow(0.7, 5)) / (1 - 0.7); // theoretical ceiling for α=0.7,K=4
ok(sd && sd.speedup > 1, `speedup > 1× (${sd.speedup.toFixed(2)}×)`);
ok(sd && sd.speedup <= nAcc + 1e-9, `speedup ≤ acceptance ceiling ${nAcc.toFixed(2)}× (${sd.speedup.toFixed(2)}×)`);
// higher acceptance ⇒ more speedup
const sdHi = computeSpecDecodePoint({ modelKey: 'llama-3-70b', draftChip: 'nvidia-b200', targetChip: 'nvidia-b200', draftModelKey: 'llama-3-2-1b', Bd: 16, K: 4, alpha: 0.9, precision: 'fp8' });
ok(sdHi.speedup > sd.speedup, `α=0.9 (${sdHi.speedup.toFixed(2)}×) faster than α=0.7 (${sd.speedup.toFixed(2)}×)`);

console.log('\nWIDE EXPERT PARALLELISM (MoE)');
const ep32 = computeWideEPPoint({ modelKey: 'deepseek-v3', chip: 'nvidia-b200', EP: 32, Bd: 64, precision: 'fp8' });
const ep128 = computeWideEPPoint({ modelKey: 'deepseek-v3', chip: 'nvidia-b200', EP: 128, Bd: 64, precision: 'fp8' });
ok(ep32 && ep128 && ep128.detail.perDeviceExperts < ep32.detail.perDeviceExperts, `wider EP → fewer experts/device (${ep32.detail.perDeviceExperts.toFixed(1)} → ${ep128.detail.perDeviceExperts.toFixed(1)})`);
ok(ep32.interFabric !== 'scale-out' && ep128.interFabric === 'scale-out', `EP=32 stays on-fabric, EP=128 (>72) crosses to scale-out`);
const epDie = computeWideEPPoint({ modelKey: 'deepseek-v3', chip: 'nvidia-b200', EP: 256, Bd: 64, precision: 'fp8' });
ok(epDie && epDie.detail.onePerDie, `EP = #experts (256) flagged one-expert-per-die`);
ok(computeWideEPPoint({ modelKey: 'llama-3-70b', chip: 'nvidia-b200', EP: 8, Bd: 16 }) === null, 'returns null for dense (non-MoE) model');

console.log('\nEPD / ENCODER (multimodal)');
const epd1 = computeEPDPoint({ modelKey: 'llama-3-70b', encoderChip: 'nvidia-b200', llmChip: 'nvidia-b200', Bd: 16, imagesPerReq: 1, precision: 'fp8' });
const epd4 = computeEPDPoint({ modelKey: 'llama-3-70b', encoderChip: 'nvidia-b200', llmChip: 'nvidia-b200', Bd: 16, imagesPerReq: 4, precision: 'fp8' });
ok(epd1 && epd1.ttft > 0 && epd1.detail.encTime > 0, `encoder adds to TTFT (encode ${(epd1.detail.encTime * 1e3).toFixed(0)}ms, TTFT ${(epd1.ttft * 1e3).toFixed(0)}ms)`);
ok(epd4.ttft > epd1.ttft, `more images → higher TTFT (${(epd1.ttft * 1e3).toFixed(0)} → ${(epd4.ttft * 1e3).toFixed(0)}ms)`);

console.log('\nDISPATCH');
for (const axis of ['prefill-decode', 'afd', 'spec-decode', 'wide-ep', 'epd']) {
  const pts = sweep({ axis, modelKey: 'deepseek-v3', chipsA: ['nvidia-b200', 'nvidia-h100'], chipsB: ['nvidia-b200'], batches: [1, 16, 64], precision: 'fp8' });
  ok(pts.length > 0, `axis '${axis}' → ${pts.length} feasible points`);
}

console.log(`\n── ${fails ? fails + ' FAILURES' : 'all axis checks passed'} ──`);
process.exit(fails ? 1 : 0);
