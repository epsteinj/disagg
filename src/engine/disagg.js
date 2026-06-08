// Disaggregation layer — the part transformer_math.html explicitly did NOT model.
// Runs prefill and decode on SEPARATE, possibly DIFFERENT chips, charges the KV-cache transfer
// between the pools, and traces the throughput × interactivity × $/token frontier so we can see
// whether HETEROGENEOUS prefill/decode beats a homogeneous fleet — and which silicon wins each phase.
//
// Deployment model (DistServe-style goodput):
//   • Decode pool: chip D serving Bd concurrent users. Interactivity = 1/TPOT (per-user tok/s);
//     system output throughput = Bd/TPOT (aggregate tok/s).
//   • Prefill pool: chip P, provisioned with enough replicas to keep the decode pool fed — i.e. to
//     sustain the prompt rate = (output throughput / T_out).
//   • KV transfer: each request's prompt KV is computed on P and shipped to D over the slower of the
//     two pools' scale-out links; that latency is added to TTFT.
//   • $/Mtok = (prefill chips + decode chips) capex, amortized 3yr/24-7, over the output token rate.
//
// v1 assumptions (documented, not hidden): prefill batch Bp=1 (prefill is compute-bound, low-batch);
// KV transfer charged as latency only (not yet as a bandwidth contention term); pools sized to min-fit.

import { chipPerfSpecs, modelPresets, computeChipSummary, estimateNonExpertBytes, _bytesPerValOf, getStepCost as getStepCostRef } from './core.js';

const AMORT_S = 3 * 365 * 24 * 3600; // 3-year capex amortization window (s)

// Inter-pool transfer bandwidth: scale-UP (each vendor's own intra-rack fabric — NVLink, Infinity Fabric,
// ICI, NeuronLink, Credo, Juniper…) vs scale-OUT (cross-rack IB/Ethernet). A scale-up fabric is
// vendor-specific and only spans ONE chip type's domain, so it's available only when both pools are the
// SAME chip AND fit that chip's own fabric-domain size (rack_size). 'auto' enforces this; mixed-vendor
// disagg is therefore forced onto scale-out. BW and domain are read per-chip (not assumed = NVL72).
function _interPoolBW(aKey, bKey, Na, Nb, topology = 'auto') {
  const ca = chipPerfSpecs[aKey], cb = chipPerfSpecs[bKey];
  const sameChip = aKey === bKey;
  const scaleUpBW = Math.min(ca.interconnect_bw || 1e9, cb.interconnect_bw || 1e9); // each chip's own fabric
  const scaleOut = Math.min(ca.scaleout_bw || 100e9, cb.scaleout_bw || 100e9);
  const fabric = sameChip ? (ca.ic_name || 'scale-up fabric') : 'mixed-vendor (no shared fabric)';
  if (topology === 'scale-up') return { bw: scaleUpBW, mode: sameChip ? fabric : 'forced scale-up (hypothetical)' };
  if (topology === 'scale-out') return { bw: scaleOut, mode: 'scale-out' };
  const fitsFabric = sameChip && (Na + Nb) <= (ca.rack_size || 8);
  return fitsFabric ? { bw: scaleUpBW, mode: fabric } : { bw: scaleOut, mode: 'scale-out' };
}

export function computeDisaggPoint({ modelKey, prefillChip, decodeChip, Bd, T_in = 4096, T_out = 512, precision = 'fp8', kvPrec = 'auto', topology = 'auto' }) {
  const model = modelPresets[modelKey];
  if (!model) return null;
  const workload = model.is_moe ? 'moe' : 'llm';
  const ioRatio = T_in / T_out; // input:output → T_max = T_in + T_out for KV sizing

  // ── Decode pool ──
  const rd = computeChipSummary(decodeChip, model, Bd, T_in, precision, workload, 'decode', 50, null, kvPrec, ioRatio);
  if (!rd) return null;
  const TPOT = rd.totalLayerTime;        // s per output token, per user
  const interactivity = 1 / TPOT;        // tok/s/user
  const throughput = rd.topValue;        // Bd / TPOT, aggregate output tok/s
  const Nd = rd.N_chips;
  const cd = chipPerfSpecs[decodeChip];
  const decodeCost = Nd * cd.cost_usd;

  // ── Prefill pool — provisioned to feed Bd decoders ──
  const rp = computeChipSummary(prefillChip, model, 1, T_in, precision, workload, 'prefill', 50, null, kvPrec, ioRatio);
  if (!rp) return null;
  const prefillTime = rp.totalLayerTime; // s to prefill one T_in prompt (Bp=1)
  const cp = chipPerfSpecs[prefillChip];
  const promptRate = throughput / T_out; // prompts/s the system completes
  const prefillReplicas = Math.max(1, Math.ceil(promptRate * prefillTime));
  const Np = prefillReplicas * rp.N_chips;
  const prefillCost = Np * cp.cost_usd;

  // ── KV transfer prefill → decode (scale-up NVLink vs scale-out, per topology) ──
  const dkv = (model.H_kv && model.d_k) ? model.H_kv * model.d_k : model.d;
  const kvBytesPerReq = 2 * dkv * model.L * T_in * rd.kv_bytes_per_val;
  const xfer = _interPoolBW(prefillChip, decodeChip, Np, Nd, topology);
  const kvTransferTime = kvBytesPerReq / xfer.bw;
  const ttft = prefillTime + kvTransferTime;

  // ── Economics ──
  const totalCost = prefillCost + decodeCost;
  const costPerMtok = throughput > 0 ? (totalCost / AMORT_S) / throughput * 1e6 : Infinity;

  return {
    prefillChip, decodeChip, Bd,
    interactivity, throughput, costPerMtok, ttft, kvTransferTime,
    Np, Nd, prefillCost, decodeCost, totalCost,
    decodeBind: rd.overallBinds, interFabric: xfer.mode,
    heterogeneous: prefillChip !== decodeChip,
    detail: {
      axis: 'prefill-decode',
      poolA: { role: 'Prefill', chipKey: prefillChip, total: Np, replicas: prefillReplicas, perReplica: rp.N_chips, time: prefillTime },
      poolB: { role: 'Decode', chipKey: decodeChip, total: Nd },
      link: { what: 'KV cache (prompt)', bytes: kvBytesPerReq, mode: xfer.mode, bw: xfer.bw, time: kvTransferTime, freq: 'once per request' },
    },
  };
}

export function sweepDisagg({ modelKey, prefillChips, decodeChips, batches, T_in = 4096, T_out = 512, precision = 'fp8' }) {
  const pts = [];
  for (const pc of prefillChips)
    for (const dc of decodeChips)
      for (const Bd of batches) {
        const p = computeDisaggPoint({ modelKey, prefillChip: pc, decodeChip: dc, Bd, T_in, T_out, precision });
        if (p && isFinite(p.throughput) && isFinite(p.costPerMtok)) pts.push(p);
      }
  return pts;
}

// Pareto frontier: keep points not dominated on BOTH objectives (default: maximize throughput &
// interactivity). For a cost frontier pass yKey='costPerMtok' with maximize=false on that axis.
export function paretoFrontier(points, { xKey = 'throughput', yKey = 'interactivity', xMax = true, yMax = true } = {}) {
  const better = (a, b, key, max) => (max ? a[key] >= b[key] : a[key] <= b[key]);
  const strictly = (a, b, key, max) => (max ? a[key] > b[key] : a[key] < b[key]);
  return points.filter(p =>
    !points.some(q => q !== p &&
      better(q, p, xKey, xMax) && better(q, p, yKey, yMax) &&
      (strictly(q, p, xKey, xMax) || strictly(q, p, yKey, yMax)))
  );
}

// ══════════════════════════════════════════════════════════════════════════════
// Additional disaggregation axes. All points share the same shape as prefill/decode
// (prefillChip/decodeChip/Bd/interactivity/throughput/costPerMtok/ttft/Np/Nd/heterogeneous/decodeBind)
// so the UI renders any axis uniformly. pool A → "prefillChip" slot, pool B → "decodeChip" slot.
// ══════════════════════════════════════════════════════════════════════════════

function _resolvePrec(chip, prec) {
  if (chip['peak_' + prec]) return prec;
  if (prec === 'fp4' && chip.peak_fp8) return 'fp8';
  return 'fp16';
}

// Per-LAYER roofline time for a SUBSET of steps on one chip. Mirrors computeChipSummary's verified
// inner loop (max(compute, memory) per step, two-tier BW, MoE expert-weight penalty) so the axis
// models stay consistent with the audited engine. Returns seconds for one layer.
function _rooflineForSteps(chip, model, steps, { B, T_seq, workload, phase, effPrec, kvPrec, N_chips, moeDecodeEff = 1 }) {
  const peak = chip['peak_' + effPrec] || chip.peak_fp16 || chip.peak_fp8;
  const bpv = _bytesPerValOf(effPrec), kvbpv = _bytesPerValOf(kvPrec);
  const compEff = chip.comp_eff ?? 0.5, bwEff = chip.bw_eff ?? 0.7;
  const memBW = chip.mem_bw, kvBW = chip.mem_bw_cold || chip.mem_bw;
  let layer = 0;
  for (const step of steps) {
    const cost = getStepCostRef(step, workload, phase, model, B, T_seq);
    const isExpertW = workload === 'moe' && (step === 14 || step === 16);
    const wBW = isExpertW ? memBW * moeDecodeEff : memBW;
    const t_w = wBW > 0 ? (cost.bytes_weight * bpv) / (wBW * bwEff * N_chips) : 0;
    const t_kv = kvBW > 0 ? (cost.bytes_kv * kvbpv) / (kvBW * bwEff * N_chips) : 0;
    const t_a = memBW > 0 ? (cost.bytes_act * bpv) / (memBW * bwEff * N_chips) : 0;
    const comp = peak > 0 ? cost.flops / (peak * compEff * N_chips) : 0;
    layer += Math.max(comp, t_w + t_kv + t_a);
  }
  return layer;
}

// ── AXIS 2 · Attention / Expert disaggregation (MoE only) ──
// Run attention (KV-bound, wants bandwidth) and the MoE experts (capacity/compute-bound) on DIFFERENT
// pools. The cost that makes-or-breaks it: the hidden state (B·d) crosses between pools EVERY layer,
// both directions — a far higher-frequency transfer than prefill/decode's once-per-request KV ship.
export function computeAttnExpertPoint({ modelKey, attnChip, expertChip, Bd, T_in = 4096, T_out = 512, precision = 'fp8', kvPrec = 'auto', topology = 'auto' }) {
  const model = modelPresets[modelKey];
  if (!model || !model.is_moe) return null; // axis is MoE-specific
  const ca = chipPerfSpecs[attnChip], ce = chipPerfSpecs[expertChip];
  if (!ca || !ce) return null;
  const ioRatio = T_in / T_out, T_max = T_in * (1 + 1 / ioRatio);
  const effA = _resolvePrec(ca, precision), effE = _resolvePrec(ce, precision);
  const bpvA = _bytesPerValOf(effA), bpvE = _bytesPerValOf(effE);
  const kvA = kvPrec === 'auto' ? effA : kvPrec;

  // Per-role capacity sizing (attn pool holds attn+embed weights + KV; expert pool holds experts).
  const nonExpertA = estimateNonExpertBytes(model, bpvA);
  const dkv = (model.H_kv && model.d_k) ? model.H_kv * model.d_k : model.d;
  const kvBytes = Bd * T_max * 2 * dkv * model.L * _bytesPerValOf(kvA);
  const attnCap = ca.mem_cap + (ca.mem_cap_cold || 0);
  let Na = Math.max(1, Math.ceil((nonExpertA + kvBytes) / attnCap));
  if (ca.weight_tier === 'fast') Na = Math.max(Na, Math.ceil(nonExpertA / ca.mem_cap));
  const expertBytes = Math.max(0, model.N_params * bpvE - estimateNonExpertBytes(model, bpvE));
  const expertCap = ce.mem_cap + (ce.mem_cap_cold || 0);
  let Ne = Math.max(1, Math.ceil(expertBytes / expertCap));
  if (ce.weight_tier === 'fast') Ne = Math.max(Ne, Math.ceil(expertBytes / ce.mem_cap));

  const E = model.E || 64, K = model.K || 8;
  const expertsTouched = Math.min(E, Math.max(K, K * Bd));
  const moeEff = Math.max(0.05, Math.min(1, expertsTouched / E));
  const attnLayer = _rooflineForSteps(ca, model, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    { B: Bd, T_seq: T_max, workload: 'moe', phase: 'decode', effPrec: effA, kvPrec: kvA, N_chips: Na });
  const expertLayer = _rooflineForSteps(ce, model, [11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
    { B: Bd, T_seq: T_max, workload: 'moe', phase: 'decode', effPrec: effE, kvPrec: effE, N_chips: Ne, moeDecodeEff: moeEff });

  const xfer = _interPoolBW(attnChip, expertChip, Na, Ne, topology);
  const transferLayer = (2 * Bd * model.d * bpvA) / xfer.bw; // hidden state both directions, per layer
  const tPerToken = model.L * (attnLayer + expertLayer + transferLayer);
  const interactivity = 1 / tPerToken, throughput = Bd / tPerToken;
  const totalCost = Na * ca.cost_usd + Ne * ce.cost_usd;
  const costPerMtok = throughput > 0 ? (totalCost / AMORT_S) / throughput * 1e6 : Infinity;
  const transferFrac = (model.L * transferLayer) / tPerToken;

  return {
    axis: 'afd', prefillChip: attnChip, decodeChip: expertChip, Bd,
    interactivity, throughput, costPerMtok, ttft: 0, kvTransferTime: 0,
    Np: Na, Nd: Ne, totalCost,
    decodeBind: transferFrac > 0.4 ? 'xfer-bound' : (expertLayer > attnLayer ? 'expert' : 'attention'),
    interFabric: xfer.mode, heterogeneous: attnChip !== expertChip, transferFrac,
    detail: {
      axis: 'afd',
      poolA: { role: 'Attention', chipKey: attnChip, total: Na },
      poolB: { role: 'Expert', chipKey: expertChip, total: Ne },
      link: { what: 'Hidden-state activations', bytesPerLayer: 2 * Bd * model.d * bpvA, mode: xfer.mode, bw: xfer.bw, freq: `every layer × 2 directions (${model.L} layers)`, transferFrac },
    },
  };
}

// ── AXIS 3 · Speculative decoding (draft / target) ──
// A small DRAFT model proposes K tokens; the big TARGET verifies them in one batched pass. Disaggregate
// draft (latency-cheap chip) from target (big chip). Accepted tokens per cycle follows the standard
// E[accept] = (1-α^(K+1))/(1-α). Assumption: target verify ≈ one decode pass (weight-bound — the K extra
// tokens ride along on the same weight read until compute-bound); documented, holds for small K.
export function computeSpecDecodePoint({ modelKey, targetChip, draftChip, draftModelKey = 'llama-3-2-1b', Bd, K = 4, alpha = 0.7, T_in = 4096, T_out = 512, precision = 'fp8' }) {
  const target = modelPresets[modelKey], draft = modelPresets[draftModelKey];
  if (!target || !draft) return null;
  const ct = chipPerfSpecs[targetChip], cd = chipPerfSpecs[draftChip];
  if (!ct || !cd) return null;
  const ioRatio = T_in / T_out;
  const rt = computeChipSummary(targetChip, target, Bd, T_in, precision, target.is_moe ? 'moe' : 'llm', 'decode', 50, null, 'auto', ioRatio);
  const rd = computeChipSummary(draftChip, draft, Bd, T_in, precision, draft.is_moe ? 'moe' : 'llm', 'decode', 50, null, 'auto', ioRatio);
  if (!rt || !rd) return null;
  const tTarget = rt.totalLayerTime, tDraft = rd.totalLayerTime;
  const nAcc = (1 - Math.pow(alpha, K + 1)) / (1 - alpha); // expected tokens emitted per cycle
  const cycle = K * tDraft + tTarget;                       // draft proposes K, target verifies in ~1 pass
  const interactivity = nAcc / cycle, throughput = Bd * interactivity;
  const speedup = interactivity * tTarget;                  // vs target-alone (1/tTarget)
  const Nt = rt.N_chips, Nd = rd.N_chips;
  const totalCost = Nt * ct.cost_usd + Nd * cd.cost_usd;
  const costPerMtok = throughput > 0 ? (totalCost / AMORT_S) / throughput * 1e6 : Infinity;
  return {
    axis: 'spec-decode', prefillChip: draftChip, decodeChip: targetChip, Bd, K, alpha,
    interactivity, throughput, costPerMtok, ttft: 0,
    Np: Nd, Nd: Nt, totalCost, speedup,
    decodeBind: `spec ×${speedup.toFixed(1)}`,
    heterogeneous: draftChip !== targetChip,
    detail: {
      axis: 'spec-decode',
      poolA: { role: 'Draft', chipKey: draftChip, total: Nd, model: draft.name },
      poolB: { role: 'Target', chipKey: targetChip, total: Nt },
      link: { what: 'Draft proposals + verification', note: 'K token-ids each way — negligible bytes', freq: 'per spec cycle' },
      K, alpha, nAcc, speedup, tDraft, tTarget,
    },
  };
}

// ── AXIS 4 · Wide Expert Parallelism (MoE only) — FIRST-ORDER MODEL ──
// DISTINCT from AFD: AFD splits attention *from* the FFN onto different pools. Wide-EP scales the EXPERT
// operator across many devices (EP degree), with attention data-parallel on the SAME devices. As EP grows,
// each device holds & reads fewer experts (faster, cheaper per device) but the all-to-all dispatch/combine
// grows and — past the chip's fabric domain (rack_size) — falls onto slow scale-out. Optimum is often
// "one expert per die" (EP = E) IF that fits inside the scale-up fabric (NVL72 / Huawei UB).
export function computeWideEPPoint({ modelKey, chip, EP, Bd, T_in = 4096, T_out = 512, precision = 'fp8', topology = 'auto' }) {
  const model = modelPresets[modelKey];
  if (!model || !model.is_moe || EP < 1) return null;
  const c = chipPerfSpecs[chip];
  if (!c) return null;
  const ioRatio = T_in / T_out, T_max = T_in * (1 + 1 / ioRatio);
  const effPrec = _resolvePrec(c, precision), bpv = _bytesPerValOf(effPrec);
  const E = model.E || 64, K = model.K || 8;
  const nonExpert = estimateNonExpertBytes(model, bpv);
  const expertBytes = Math.max(0, model.N_params * bpv - nonExpert);
  const dkv = (model.H_kv && model.d_k) ? model.H_kv * model.d_k : model.d;
  const kvBytes = Bd * T_max * 2 * dkv * model.L * bpv;
  // Per-device capacity: replicated attention/embeddings (DP) + expert shard (E/EP) + KV shard.
  const cap = c.mem_cap + (c.mem_cap_cold || 0);
  if (nonExpert + expertBytes / EP + kvBytes / EP > cap * 0.95) return null; // infeasible at this EP/batch
  const memBW = c.mem_bw, bwEff = c.bw_eff ?? 0.7;
  // Per-token decode reads (all layers): attention weights (replicated) + KV shard + per-device expert read.
  const expertsTouched = Math.min(E, Math.max(K, K * Bd));
  const perDeviceExpertRead = (expertsTouched / E) * expertBytes / EP;
  const memT = (nonExpert + kvBytes / EP + perDeviceExpertRead) / (memBW * bwEff);
  // All-to-all dispatch+combine per layer, over scale-up fabric if EP fits the domain, else scale-out.
  const inDomain = EP <= (c.rack_size || 8);
  const fabricBW = topology === 'scale-out' ? (c.scaleout_bw || 100e9)
                 : topology === 'scale-up' ? (c.interconnect_bw || 1e9)
                 : (inDomain ? (c.interconnect_bw || 1e9) : (c.scaleout_bw || 100e9));
  const commT = model.L * 2 * (Bd * K * model.d * bpv / EP) / fabricBW;
  const decodeT = memT + commT;
  const interactivity = 1 / decodeT, throughput = Bd / decodeT;
  const totalCost = EP * c.cost_usd;
  const costPerMtok = throughput > 0 ? (totalCost / AMORT_S) / throughput * 1e6 : Infinity;
  const commFrac = commT / decodeT;
  const fabricMode = (inDomain || topology === 'scale-up') ? (c.ic_name || 'scale-up') : 'scale-out';
  return {
    axis: 'wide-ep', prefillChip: chip, decodeChip: chip, Bd, EP,
    interactivity, throughput, costPerMtok, ttft: 0,
    Np: EP, Nd: EP, totalCost, heterogeneous: false, interFabric: fabricMode,
    decodeBind: commFrac > 0.4 ? 'all-to-all' : (EP === E ? '1 expert/die' : 'expert read'),
    detail: {
      axis: 'wide-ep',
      poolA: { role: 'Attention (DP)', chipKey: chip, total: EP },
      poolB: { role: `Experts (EP=${EP}, ${(E / EP).toFixed(1)}/die)`, chipKey: chip, total: EP },
      link: { what: 'expert all-to-all (dispatch + combine)', mode: fabricMode, freq: `every layer (${model.L}) · ${commFrac > 0.4 ? 'comm-bound' : 'comm-light'}` },
      EP, E, perDeviceExperts: E / EP, commFrac, onePerDie: EP === E,
    },
  };
}

// ── AXIS 5 · EPD / Encoder disaggregation (multimodal) — FIRST-ORDER MODEL ──
// Splits the modality ENCODER (vision/audio → embeddings; compute-bound, saturates tensor cores) from the
// LLM (prefill + decode). Encoder runs on its own pool; image embeddings cross to the LLM pool once per
// request. Encoder is parameterized by GFLOPs/image, tokens/image, images/request (no VLM preset needed —
// any model is treated as the LLM, with image tokens appended to the prompt).
export function computeEPDPoint({ modelKey, encoderChip, llmChip, Bd, T_in = 4096, T_out = 512, precision = 'fp8', imagesPerReq = 1, encGflopsPerImage = 200, imgTokensPerImage = 256, topology = 'auto' }) {
  const model = modelPresets[modelKey];
  if (!model) return null;
  const ce = chipPerfSpecs[encoderChip], cl = chipPerfSpecs[llmChip];
  if (!ce || !cl) return null;
  const workload = model.is_moe ? 'moe' : 'llm';
  const imgTokens = Math.round(imgTokensPerImage * imagesPerReq);
  const T_in_eff = T_in + imgTokens;       // image tokens lengthen the prompt the LLM digests
  const ioRatio = T_in_eff / T_out;
  // LLM pool: decode (interactivity / throughput) + prefill (TTFT) on llmChip.
  const rd = computeChipSummary(llmChip, model, Bd, T_in_eff, precision, workload, 'decode', 50, null, 'auto', ioRatio);
  const rp = computeChipSummary(llmChip, model, 1, T_in_eff, precision, workload, 'prefill', 50, null, 'auto', ioRatio);
  if (!rd || !rp) return null;
  const interactivity = 1 / rd.totalLayerTime, throughput = rd.topValue;
  const Nllm = rd.N_chips, llmCost = Nllm * cl.cost_usd;
  // Encoder pool: compute-bound; sized (replicas) to feed the request rate.
  const effEnc = _resolvePrec(ce, precision);
  const peakEnc = ce['peak_' + effEnc] || ce.peak_fp16;
  const encTimePerReq = (encGflopsPerImage * 1e9 * imagesPerReq) / (peakEnc * (ce.comp_eff ?? 0.5));
  const promptRate = throughput / T_out;
  const Nenc = Math.max(1, Math.ceil(promptRate * encTimePerReq));
  const encCost = Nenc * ce.cost_usd;
  // Transfer: image embeddings encoder → LLM, once per request.
  const embBytes = imgTokens * model.d * _bytesPerValOf(effEnc);
  const xfer = _interPoolBW(encoderChip, llmChip, Nenc, Nllm, topology);
  const xferTime = embBytes / xfer.bw;
  const ttft = encTimePerReq + xferTime + rp.totalLayerTime;
  const totalCost = encCost + llmCost;
  const costPerMtok = throughput > 0 ? (totalCost / AMORT_S) / throughput * 1e6 : Infinity;
  return {
    axis: 'epd', prefillChip: encoderChip, decodeChip: llmChip, Bd,
    interactivity, throughput, costPerMtok, ttft,
    Np: Nenc, Nd: Nllm, totalCost, heterogeneous: encoderChip !== llmChip,
    interFabric: xfer.mode, decodeBind: rd.overallBinds,
    detail: {
      axis: 'epd',
      poolA: { role: 'Encoder', chipKey: encoderChip, total: Nenc },
      poolB: { role: 'LLM (prefill+decode)', chipKey: llmChip, total: Nllm },
      link: { what: 'image embeddings', bytes: embBytes, mode: xfer.mode, bw: xfer.bw, time: xferTime, freq: `per request (${imagesPerReq} img × ${imgTokensPerImage} tok)` },
      encTime: encTimePerReq, imgTokens,
    },
  };
}

// Unified sweep dispatcher. chipsA → pool A (prefill / attention / draft / encoder), chipsB → pool B (decode / FFN / target / LLM / EP chip).
export function sweep({ axis = 'prefill-decode', modelKey, chipsA, chipsB, batches, T_in = 4096, T_out = 512, precision = 'fp8', draftModelKey = 'llama-3-2-1b', K = 4, alpha = 0.7, topology = 'auto', epDegrees = [8, 16, 32, 64, 128, 256], imagesPerReq = 1, encGflopsPerImage = 200, imgTokensPerImage = 256 }) {
  const pts = [];
  const keep = p => { if (p && isFinite(p.throughput) && isFinite(p.costPerMtok) && p.throughput > 0) pts.push(p); };
  if (axis === 'wide-ep') {
    for (const b of chipsB) for (const Bd of batches) for (const EP of epDegrees)
      keep(computeWideEPPoint({ modelKey, chip: b, EP, Bd, T_in, T_out, precision, topology }));
  } else if (axis === 'epd') {
    for (const a of chipsA) for (const b of chipsB) for (const Bd of batches)
      keep(computeEPDPoint({ modelKey, encoderChip: a, llmChip: b, Bd, T_in, T_out, precision, imagesPerReq, encGflopsPerImage, imgTokensPerImage, topology }));
  } else {
    for (const a of chipsA) for (const b of chipsB) for (const Bd of batches) {
      if (axis === 'afd') keep(computeAttnExpertPoint({ modelKey, attnChip: a, expertChip: b, Bd, T_in, T_out, precision, topology }));
      else if (axis === 'spec-decode') keep(computeSpecDecodePoint({ modelKey, targetChip: b, draftChip: a, draftModelKey, Bd, K, alpha, T_in, T_out, precision }));
      else keep(computeDisaggPoint({ modelKey, prefillChip: a, decodeChip: b, Bd, T_in, T_out, precision, topology }));
    }
  }
  return pts;
}
