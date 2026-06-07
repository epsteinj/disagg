import { chipPerfSpecs } from './chips.js';

const modelPresets = {
  // ─────────────── Frontier LLMs (proprietary or >100B active) ───────────────
  // GPT-5.5: shipped Apr 23, 2026. Analyst consensus ~1.8T total / ~110B active / ~6% sparsity / K≈2-4.
  // OpenAI does not disclose. Inferred from latency, throughput, GB300-NVL72 sizing. Treat as estimate.
  'gpt-5-5':            { name: 'GPT-5.5 (~1.8T MoE, ~110B active, est)', L: 120, d: 14336, d_k: 128, H_q: 112, H_kv: 14,  d_ffn: 7168,  vocab: 200000, N_params: 1.8e12, N_active: 110e9, is_diffusion: false, is_moe: true,  E: 64,  K: 4 },
  // Theoretical 2T MoE — matches the colleague's chassis-scale SAM analysis (slide 8 of 20260526 PDF).
  // MLA-style attention (H_kv=4 approximates compressed latent dim) so non-expert weights stay small enough
  // to enable pure EP across chassis. ~6.5% sparsity / top-8 of 256 experts — the architectural sweet spot
  // for TDN's chassis-scale pure-EP pitch.
  'theoretical-2t-moe': { name: 'Theoretical 2T MoE (MLA, 130B active)',  L: 130, d: 16384, d_k: 128, H_q: 128, H_kv: 4,   d_ffn: 8192,  vocab: 200000, N_params: 2e12,   N_active: 130e9, is_diffusion: false, is_moe: true,  E: 256, K: 8 },
  // Claude Opus 4.7: Anthropic does NOT disclose architecture. Analyst consensus is "dense or near-dense, NOT MoE."
  // Treat dimensions as estimates calibrated against pricing tier ($5/$25 per M) and 1M-context support.
  'claude-opus-4-7':    { name: 'Claude Opus 4.7 (~500B dense, est)',  L: 120, d: 16384, d_k: 128, H_q: 128, H_kv: 16,  d_ffn: 53248, vocab: 200000, N_params: 500e9,                   is_diffusion: false, is_moe: false },
  // Kimi K2 uses MLA-style KV compression (Moonshot's "MoBA"). Effective d_kv ≈ 512.
  'kimi-k2':            { name: 'Kimi K2 (1T MoE, 32B active)',        L: 61,  d: 7168,  d_k: 128, H_q: 64,  H_kv: 4,   d_ffn: 2048,  vocab: 163840, N_params: 1e12,   N_active: 32e9,  is_diffusion: false, is_moe: true,  E: 384, K: 8 },
  'qwen3-235b-a22b':    { name: 'Qwen3-235B-A22B (MoE)',               L: 94,  d: 4096,  d_k: 128, H_q: 64,  H_kv: 4,   d_ffn: 1536,  vocab: 151936, N_params: 235e9,  N_active: 22e9,  is_diffusion: false, is_moe: true,  E: 128, K: 8 },
  // DeepSeek V3 uses MLA (Multi-Latent Attention) — KV is compressed to a latent dim ~512, not full 16384 MHA.
  // Setting H_kv=4 to approximate the compressed effective KV dimension (4 × d_k=128 = 512).
  'deepseek-v3':        { name: 'DeepSeek V3 (671B MoE, 37B active, MLA)', L: 61,  d: 7168,  d_k: 128, H_q: 128, H_kv: 4,   d_ffn: 2048,  vocab: 129280, N_params: 671e9,  N_active: 37e9,  is_diffusion: false, is_moe: true,  E: 256, K: 8 },
  'llama-3-405b':       { name: 'Llama 3.1 405B',                      L: 126, d: 16384, d_k: 128, H_q: 128, H_kv: 8,   d_ffn: 53248, vocab: 128256, N_params: 405e9,                   is_diffusion: false, is_moe: false },
  'gpt4-class':         { name: 'GPT-4 class (est, dense)',            L: 120, d: 16384, d_k: 128, H_q: 128, H_kv: 16,  d_ffn: 65536, vocab: 200000, N_params: 1.8e12,                  is_diffusion: false, is_moe: false },

  // ─────────────── Mid-size LLMs (10-100B) ───────────────
  'llama-3-70b':        { name: 'Llama 3.1 70B',                       L: 80,  d: 8192,  d_k: 128, H_q: 64,  H_kv: 8,   d_ffn: 28672, vocab: 128256, N_params: 70.6e9,                  is_diffusion: false, is_moe: false },
  'mixtral-8x22b':      { name: 'Mixtral 8×22B (MoE)',                 L: 56,  d: 6144,  d_k: 128, H_q: 48,  H_kv: 8,   d_ffn: 16384, vocab: 32000,  N_params: 141e9,  N_active: 39e9,  is_diffusion: false, is_moe: true,  E: 8,   K: 2 },
  'mistral-small-3':    { name: 'Mistral Small 3 (24B)',               L: 40,  d: 5120,  d_k: 128, H_q: 32,  H_kv: 8,   d_ffn: 14336, vocab: 131072, N_params: 23.5e9,                  is_diffusion: false, is_moe: false },
  'mixtral-8x7b':       { name: 'Mixtral 8×7B (MoE)',                  L: 32,  d: 4096,  d_k: 128, H_q: 32,  H_kv: 8,   d_ffn: 14336, vocab: 32000,  N_params: 47e9,   N_active: 13e9,  is_diffusion: false, is_moe: true,  E: 8,   K: 2 },
  'gemma-2-9b':         { name: 'Gemma 2 9B',                          L: 42,  d: 3584,  d_k: 256, H_q: 16,  H_kv: 8,   d_ffn: 14336, vocab: 256000, N_params: 9e9,                     is_diffusion: false, is_moe: false },
  'qwen3-8b':           { name: 'Qwen3 8B',                            L: 36,  d: 4096,  d_k: 128, H_q: 32,  H_kv: 8,   d_ffn: 12288, vocab: 151936, N_params: 8e9,                     is_diffusion: false, is_moe: false },
  'llama-3-8b':         { name: 'Llama 3.1 8B',                        L: 32,  d: 4096,  d_k: 128, H_q: 32,  H_kv: 8,   d_ffn: 14336, vocab: 128256, N_params: 8.0e9,                   is_diffusion: false, is_moe: false },

  // ─────────────── SLMs (<10B, edge-deployable) ───────────────
  'phi-3-5-mini':       { name: 'Phi-3.5 mini (3.8B)',                 L: 32,  d: 3072,  d_k: 96,  H_q: 32,  H_kv: 32,  d_ffn: 8192,  vocab: 32064,  N_params: 3.8e9,                   is_diffusion: false, is_moe: false },
  'llama-3-2-3b':       { name: 'Llama 3.2 3B',                        L: 28,  d: 3072,  d_k: 128, H_q: 24,  H_kv: 8,   d_ffn: 8192,  vocab: 128256, N_params: 3.2e9,                   is_diffusion: false, is_moe: false },
  'gemma-2-2b':         { name: 'Gemma 2 2B',                          L: 26,  d: 2304,  d_k: 256, H_q: 8,   H_kv: 4,   d_ffn: 9216,  vocab: 256000, N_params: 2.6e9,                   is_diffusion: false, is_moe: false },
  'llama-3-2-1b':       { name: 'Llama 3.2 1B',                        L: 16,  d: 2048,  d_k: 64,  H_q: 32,  H_kv: 8,   d_ffn: 8192,  vocab: 128256, N_params: 1.2e9,                   is_diffusion: false, is_moe: false },
  'qwen3-0-6b':         { name: 'Qwen3 0.6B (edge)',                   L: 28,  d: 1024,  d_k: 128, H_q: 16,  H_kv: 8,   d_ffn: 3072,  vocab: 151936, N_params: 600e6,                   is_diffusion: false, is_moe: false },

  // ─────────────── Diffusion (DiT) ───────────────
  'sd3-medium':         { name: 'SD3 Medium',                          L: 24,  d: 1536,  d_k: 64,  H_q: 24,  H_kv: 24,  d_ffn: 6144,                 N_params: 2.0e9,                   is_diffusion: true,  is_moe: false },
  'flux-dev':           { name: 'Flux.1 Dev',                          L: 28,  d: 3072,  d_k: 128, H_q: 24,  H_kv: 24,  d_ffn: 12288,                N_params: 12e9,                    is_diffusion: true,  is_moe: false },
  'sora-class':         { name: 'Sora-class (est)',                    L: 60,  d: 4096,  d_k: 128, H_q: 32,  H_kv: 32,  d_ffn: 16384,                N_params: 50e9,                    is_diffusion: true,  is_moe: false },
};

// chipPerfSpecs imported from ./chips.js — catalogue corrected vs mid-2026 sources (see audit/AUDIT.md).

// Per-step FLOPs and bytes (approximate). Returns { flops, bytes_weight, bytes_act, bytes_kv }.
function getStepCost(step, workload, phase, model, B, T) {
  const d = model.d, d_k = model.d_k, H_q = model.H_q, H_kv = model.H_kv, d_kv = H_kv * d_k, d_ffn = model.d_ffn;
  const is_decode = (((workload === 'llm' || workload === 'moe') && phase === 'decode'));
  const T_new = is_decode ? 1 : T;  // tokens/patches processed THIS step
  const T_seq = T;                  // total sequence length (for KV cache reads at decode)
  // Training overlay: full T tokens always (no decode optimization).
  // Forward: same as prefill. Backward: ~2x forward FLOPs + writes gradient bytes.
  // Optimizer: reads weights + gradients + 2 moments; writes weights + 2 moments. Compute trivial.
  const isTraining = (workload === 'training');
  const trainingPhase = isTraining ? phase : null;  // 'forward' | 'backward' | 'optimizer'
  let f = 0, bw = 0, ba = 0, bk = 0;

  // MoE workload: 21-step expansion (router, top-K, dispatch, K-active expert FFN, combine, gather)
  if (workload === 'moe') {
    const E = model.E || 64, K = model.K || 4;
    switch (step) {
      // Steps 0-10: identical to LLM (input, embed, Q/K/V, attention, output proj, residual)
      case 0: ba = B * T_new * 4; break;
      case 1: f = B * T_new * d; bw = T_new * d; ba = B * T_new * d; break;
      case 2: f = 2 * B * T_new * d * d; bw = d * d; ba = B * T_new * d * 2; break;
      case 3: f = 4 * B * T_new * d * d_kv; bw = 2 * d * d_kv; ba = 2 * B * T_new * d_kv; break;
      case 4:
        if (is_decode) { f = 2 * B * H_q * T_new * T_seq * d_k; bk = B * H_kv * T_seq * d_k; ba = B * H_q * T_seq; }
        else { f = 2 * B * H_q * T_seq * T_seq * d_k; ba = B * H_q * T_seq * T_seq; }
        break;
      case 5: case 7:
        if (is_decode) { f = 5 * B * H_q * T_seq; ba = B * H_q * T_seq; }
        else { f = 5 * B * H_q * T_seq * T_seq; ba = B * H_q * T_seq * T_seq; }
        break;
      case 6:
        if (is_decode) break;
        f = B * H_q * T_seq * T_seq; ba = B * H_q * T_seq * T_seq;
        break;
      case 8:
        if (is_decode) { f = 2 * B * H_q * T_new * T_seq * d_k; bk = B * H_kv * T_seq * d_k; }
        else { f = 2 * B * H_q * T_seq * T_seq * d_k; ba = B * H_q * T_seq * T_seq + B * T_seq * d_kv; }
        break;
      case 9: f = 2 * B * T_new * d * d; bw = d * d; ba = B * T_new * d; break;
      case 10: f = B * T_new * d; ba = B * T_new * d * 2; break;
      // Step 11: Router gating (tiny matmul d×E)
      case 11: f = 2 * B * T_new * d * E; bw = d * E; ba = B * T_new * E; break;
      // Step 12: Top-K + softmax (cheap)
      case 12: f = B * T_new * E + B * T_new * K * 5; ba = B * T_new * K; break;
      // Step 13: Token dispatch (all-to-all). Compute negligible; activations cross fabric.
      case 13: ba = B * T_new * d; break;
      // Step 14: Expert W_up + W_gate (K active per token)
      case 14: {
        f = 4 * B * T_new * d * d_ffn * K;
        const experts_touched = Math.min(E, Math.max(K, K * B * T_new));
        bw = 2 * d * d_ffn * experts_touched;
        ba = B * T_new * d_ffn * 2 * K;
        break;
      }
      // Step 15: Expert activation (SwiGLU)
      case 15: f = 4 * B * T_new * d_ffn * K; ba = B * T_new * d_ffn * K; break;
      // Step 16: Expert W_down (K active)
      case 16: {
        f = 2 * B * T_new * d_ffn * d * K;
        const experts_touched = Math.min(E, Math.max(K, K * B * T_new));
        bw = d_ffn * d * experts_touched;
        ba = B * T_new * d * K;
        break;
      }
      // Step 17: Combine (weighted sum of K expert outputs)
      case 17: f = B * T_new * d * K; ba = B * T_new * d * K; break;
      // Step 18: Token gather (all-to-all reverse)
      case 18: ba = B * T_new * d; break;
      // Step 19: Residual + LayerNorm after MoE FFN
      case 19: f = B * T_new * d; ba = B * T_new * d * 2; break;
      // Step 20: Cross-chip / next layer
      case 20: f = B * T_new * d; bw = d; break;
    }
    return { flops: f, bytes_weight: bw, bytes_act: ba, bytes_kv: bk };
  }

  switch (step) {
    case 0: ba = B * T_new * 4; break;
    case 1: f = B * T_new * d; bw = T_new * d; ba = B * T_new * d; break;
    case 2: f = 2 * B * T_new * d * d; bw = d * d; ba = B * T_new * d * 2; break;
    case 3: f = 4 * B * T_new * d * d_kv; bw = 2 * d * d_kv; ba = 2 * B * T_new * d_kv; break;
    case 4:
      if (is_decode) { f = 2 * B * H_q * T_new * T_seq * d_k; bk = B * H_kv * T_seq * d_k; ba = B * H_q * T_seq; }
      else { f = 2 * B * H_q * T_seq * T_seq * d_k; ba = B * H_q * T_seq * T_seq; }
      break;
    case 5: case 7:
      if (is_decode) { f = 5 * B * H_q * T_seq; ba = B * H_q * T_seq; }
      else { f = 5 * B * H_q * T_seq * T_seq; ba = B * H_q * T_seq * T_seq; }
      break;
    case 6:
      if (workload === 'diffusion') break;  // mask skipped in diffusion
      if (is_decode) break;                 // mask skipped at decode time
      f = B * H_q * T_seq * T_seq; ba = B * H_q * T_seq * T_seq;
      break;
    case 8:
      if (is_decode) { f = 2 * B * H_q * T_new * T_seq * d_k; bk = B * H_kv * T_seq * d_k; }
      else { f = 2 * B * H_q * T_seq * T_seq * d_k; ba = B * H_q * T_seq * T_seq + B * T_seq * d_kv; }
      break;
    case 9: f = 2 * B * T_new * d * d; bw = d * d; ba = B * T_new * d; break;
    case 10: case 14: f = B * T_new * d; ba = B * T_new * d * 2; break;
    case 11: {  // FFN up — SwiGLU has W_up + W_gate
      if (model.is_moe) {
        const K = model.K, E = model.E;
        // Compute: each of B·T tokens activates K experts, each doing full FFN's worth of work
        f = 4 * B * T_new * d * d_ffn * K;
        // Weight bytes: at low B·T, only ~K·B·T experts get touched. At high B·T, all E.
        const experts_touched = Math.min(E, Math.max(K, K * B * T_new));
        bw = 2 * d * d_ffn * experts_touched;
        ba = B * T_new * d_ffn * 2 * K;  // K experts' activations
      } else {
        f = 4 * B * T_new * d * d_ffn;
        bw = 2 * d * d_ffn;
        ba = B * T_new * d_ffn * 2;
      }
      break;
    }
    case 12: {  // Activation
      if (model.is_moe) {
        const K = model.K;
        f = 4 * B * T_new * d_ffn * K;
        ba = B * T_new * d_ffn * K;
      } else {
        f = 4 * B * T_new * d_ffn;
        ba = B * T_new * d_ffn;
      }
      break;
    }
    case 13: {  // FFN down
      if (model.is_moe) {
        const K = model.K, E = model.E;
        f = 2 * B * T_new * d_ffn * d * K;
        const experts_touched = Math.min(E, Math.max(K, K * B * T_new));
        bw = d_ffn * d * experts_touched;
        ba = B * T_new * d * K;
      } else {
        f = 2 * B * T_new * d_ffn * d;
        bw = d_ffn * d;
        ba = B * T_new * d;
      }
      break;
    }
    case 15: f = B * T_new * d; bw = d; break;
  }
  // Training overlay: scale compute and add gradient/optimizer memory traffic.
  if (isTraining) {
    if (trainingPhase === 'backward') {
      // Backward pass: ~2× forward FLOPs (matmul gradients dY → dX and dW each ~= forward).
      f *= 2;
      // Gradient writes: add bw (gradient w.r.t. weights) as additional weight-bytes traffic.
      bw *= 2;  // approximate: one weight read, one gradient write per weight
    } else if (trainingPhase === 'optimizer') {
      // Adam optimizer: compute is trivial (~10 FLOPs per param), memory dominates.
      // Per param: read W (1×), read grad (1×), read m (1×), read v (1×), write W, write m, write v.
      // Total memory traffic = 7× weight bytes per param. Compute scales with weight count.
      f = bw * 10;  // ~10 FLOPs per param for Adam update
      bw = bw * 7;  // 7 reads/writes per param
      ba = 0;
    }
    // 'forward' phase: identical to prefill (no overlay needed)
  }
  return { flops: f, bytes_weight: bw, bytes_act: ba, bytes_kv: bk };
}

function formatFlops(f) {
  if (!f) return '—';
  if (f >= 1e15) return (f / 1e15).toFixed(2) + ' PF';
  if (f >= 1e12) return (f / 1e12).toFixed(2) + ' TF';
  if (f >= 1e9) return (f / 1e9).toFixed(2) + ' GF';
  if (f >= 1e6) return (f / 1e6).toFixed(1) + ' MF';
  if (f >= 1e3) return (f / 1e3).toFixed(1) + ' KF';
  return f.toFixed(0) + ' F';
}
function formatBytes(b) {
  if (!b) return '—';
  if (b >= 1e12) return (b / 1e12).toFixed(2) + ' TB';
  if (b >= 1e9) return (b / 1e9).toFixed(2) + ' GB';
  if (b >= 1e6) return (b / 1e6).toFixed(1) + ' MB';
  if (b >= 1e3) return (b / 1e3).toFixed(1) + ' KB';
  return b.toFixed(0) + ' B';
}
function formatTime(t) {
  if (!t || !isFinite(t)) return '—';
  if (t >= 1) return t.toFixed(3) + ' s';
  if (t >= 1e-3) return (t * 1e3).toFixed(3) + ' ms';
  if (t >= 1e-6) return (t * 1e6).toFixed(2) + ' μs';
  return (t * 1e9).toFixed(1) + ' ns';
}

// Substitute variable names in an O() formula with their values.
// Longest-first replacement so 'd_ffn' is matched before 'd', 'd_kv' before 'd_k' etc.
function substituteFormula(formula, vars) {
  if (!formula || formula === '—') return formula || '—';
  let result = formula.replace(/^O\s*\(/, '').replace(/\)\s*$/, '');
  const keys = Object.keys(vars).sort((a, b) => b.length - a.length);
  for (const k of keys) {
    const v = vars[k];
    const formatted = typeof v === 'number' ? v.toString() : v;
    result = result.split(k).join(formatted);
  }
  return result;
}

// Compute summary metrics for a (chip, model, B, T, precision, workload, phase, N_steps) combination.
// Used for both the main per-step table and the multi-chip comparison.
// Resolve KV-cache precision against chip support. KV precision is INDEPENDENT of weight precision in
// production (e.g., FP8 KV widely used with FP16 weights). Falls back to highest-supported KV precision
// if requested isn't natively supported by the chip.
function _resolveKvPrecision(requested, weightEffPrec, chip) {
  // 'auto' or unset → match the weight effective precision (legacy / no-surprise behavior)
  if (!requested || requested === 'auto') return weightEffPrec;
  // Chip's supported KV precisions are inferred from its compute-precision support
  // (a chip with FP8 compute virtually always supports FP8 KV reads; FP16 is universal).
  const kvSupport = ['fp16'];
  if (chip.peak_fp8) kvSupport.push('fp8');
  if (chip.peak_fp4) kvSupport.push('fp4');
  if (kvSupport.includes(requested)) return requested;
  // Fallback: FP4 → FP8 → FP16
  if (requested === 'fp4' && kvSupport.includes('fp8')) return 'fp8';
  return 'fp16';
}
function _bytesPerValOf(prec) { return prec === 'fp16' ? 2 : (prec === 'fp8' ? 1 : 0.5); }

// Optional 9th param `N_chips_override`: if provided, use that cluster size instead of the auto-fit minimum
// (used by the SLO-driven over-provisioning search — see deriveBatchFromSLO). Will never go BELOW auto-fit
// (you can't under-provision and still hold the model + KV), so the effective N_chips = max(override, auto-fit).
// Optional 10th param `kv_precision_requested`: 'auto' (match weights, default) | 'fp16' | 'fp8' | 'fp4'.
// KV reads independently of weight storage; production typically uses FP8 KV even when weights are FP16 or FP4.
// Optional 11th param `ioRatio`: input:output token ratio for the request. When provided (LLM/MoE only),
// the KV cache is sized for T_max = T × (1 + 1/ioRatio) — the maximum sequence length the cache must hold
// (input prompt + generated output). Prefill compute still uses T (input only); decode uses T_max for cache reads.
// Estimate the non-expert parameter count: attention (Q/K/V/O projections) + embeddings + layer norms.
// For dense models, this equals total params. For MoE, this is the portion that must REPLICATE under
// pure expert parallelism (since experts shard, but attention/embeddings don't). Used to gate whether
// pure EP at chassis scale is feasible: non_expert must fit per chip alongside an expert shard + KV.
function estimateNonExpertBytes(model, bytes_per_val) {
  if (!model.is_moe) {
    // Dense: all weights are "non-expert" — nothing shards naturally
    return (model.N_params || 0) * bytes_per_val;
  }
  // MoE: compute attention + embeddings only. MLA-style models (DeepSeek, Kimi) approximate via low H_kv.
  const L = model.L || 1;
  const d = model.d || 0;
  const d_k = model.d_k || 128;
  const H_q = model.H_q || 1;
  const H_kv = model.H_kv || H_q;
  const vocab = model.vocab || 0;
  // Per layer: Q (d × H_q × d_k) + K (d × H_kv × d_k) + V (d × H_kv × d_k) + O (H_q × d_k × d) + 2× norm (d)
  const attn_per_layer = d * (H_q * d_k) + 2 * d * (H_kv * d_k) + (H_q * d_k) * d + 2 * d;
  const embed = 2 * vocab * d;  // input embed + lm_head (often tied, but conservative: count both)
  const final_norm = d;
  const non_expert_params = L * attn_per_layer + embed + final_norm;
  return non_expert_params * bytes_per_val;
}

// Derive the deployment-mode plan for (model, chip) given chassis-mode toggle.
// Returns the recommended parallelism strategy and comm pattern that follows.
//   - per_replica:        chassis off — uses existing per-replica TP sizing
//   - chassis_dp:         model fits with N_perReplica < rack_size → DP scaling (independent replicas)
//   - chassis_pure_ep:    MoE + non-expert fits per chip → one chassis-wide EP instance (no TP all-reduce)
//   - chassis_forced_tp:  model needs all chassis chips via TP (no EP option) → heavy all-reduce
//   - multi_chassis:      model doesn't fit even at chassis scale via TP
function deriveDeploymentMode(model, chip, chassisModeOn, bytes_per_val, kv_bytes_per_val, B, T_max) {
  const rackSize = chip.rack_size || 8;
  const nonExpertBytes = estimateNonExpertBytes(model, bytes_per_val);
  // KV cache size estimate
  const d_kv_eff = (model.H_kv && model.d_k) ? (model.H_kv * model.d_k) : (model.d || 0);
  const kvBytesTotal = (model.L || 1) * 2 * d_kv_eff * B * T_max * kv_bytes_per_val;
  // Expert weights total = N_params - non-expert params (in bytes).
  const expertBytesTotal = model.is_moe ? Math.max(0, (model.N_params || 0) * bytes_per_val - nonExpertBytes) : 0;
  const expertBytesPerChip = expertBytesTotal / rackSize;

  // Pure EP feasibility: non-expert (replicated) + per-chip expert shard + KV all fit per chip.
  // 80% budget leaves room for activations + scratch + alignment.
  const budget = chip.mem_cap * 0.80;
  const kvBytesPerChassisChip = kvBytesTotal / rackSize;
  const pureEpFootprint = nonExpertBytes + expertBytesPerChip + kvBytesPerChassisChip;
  const pureEpViable = model.is_moe && (pureEpFootprint <= budget);

  // Per-replica TP sizing (matches computeChipSummary's autofit math).
  const modelBytes = (model.N_params || 0) * bytes_per_val;
  const N_perReplica = Math.max(1, Math.ceil((modelBytes + kvBytesTotal) / chip.mem_cap));
  // Chassis-wide TP feasibility: total bytes / rack_size fits per chip.
  const tpAtChassisFootprint = (modelBytes + kvBytesTotal) / rackSize;
  const tpAtChassisViable = tpAtChassisFootprint <= chip.mem_cap;
  const fitsInOneChassis = N_perReplica <= rackSize;

  const base = { N_perReplica, nonExpertBytes, pureEpFootprint, pureEpViable, fitsInOneChassis };

  if (!chassisModeOn) {
    return { ...base, mode: 'per_replica', parallelismType: 'tp', commNote: 'per-replica TP' };
  }
  if (!fitsInOneChassis) {
    return { ...base, mode: 'multi_chassis', parallelismType: 'tp', commNote: 'exceeds single chassis' };
  }
  // For MoE that can do pure EP, prefer it (avoids TP all-reduce penalty even at small replica sizes).
  if (pureEpViable) {
    return { ...base, mode: 'chassis_pure_ep', parallelismType: 'ep', commNote: 'pure EP · light all-to-all' };
  }
  // If per-replica fits comfortably below chassis, run DP across chassis (replicas).
  if (N_perReplica < rackSize) {
    return { ...base, mode: 'chassis_dp', parallelismType: 'tp', commNote: `DP × ${Math.floor(rackSize / N_perReplica)} replicas` };
  }
  // Model needs full chassis as one TP unit (dense at scale, or MoE with too-big non-expert).
  if (tpAtChassisViable) {
    return { ...base, mode: 'chassis_forced_tp', parallelismType: 'tp', commNote: model.is_moe ? 'hybrid TP+EP · heavy all-reduce' : 'chassis TP · heavy all-reduce' };
  }
  return { ...base, mode: 'multi_chassis', parallelismType: 'tp', commNote: 'exceeds single chassis' };
}

function computeChipSummary(chipKey, model, B, T, precision, workload, phase, N_steps, N_chips_override, kv_precision_requested, ioRatio, parallelismMode) {
  const chip = chipPerfSpecs[chipKey];
  if (!chip) return null;
  // Effective precision: if the chip doesn't support the requested precision natively,
  // fall back to the highest supported precision. We need to use this fallback for BOTH
  // the compute peak AND bytes_per_val — otherwise we get the bandwidth benefit of a
  // smaller representation without paying for the chip's actual storage format.
  let effPrec = precision;
  if (!chip[`peak_${effPrec}`]) {
    if (effPrec === 'fp4' && chip.peak_fp8) effPrec = 'fp8';
    else effPrec = 'fp16';
  }
  const peak = chip[`peak_${effPrec}`];
  if (!peak) return null;
  const bytes_per_val = _bytesPerValOf(effPrec);
  // KV cache precision — independent of weight precision in production. Defaults to matching weights.
  const kvPrec = _resolveKvPrecision(kv_precision_requested, effPrec, chip);
  const kv_bytes_per_val = _bytesPerValOf(kvPrec);
  const memBW = chip.mem_bw;
  // Sustained-effective convention: derate peak compute (MFU) and peak bandwidth (realized fraction).
  // Defaults 0.50/0.70 for GPU/ASIC; Groq & Cerebras carry 1.0 because their mem_bw is already a
  // calibrated effective constant. See src/engine/chips.js.
  const compEff = chip.comp_eff ?? 0.50;
  const bwEff = chip.bw_eff ?? 0.70;
  // T_input vs T_max: KV cache holds prompt + generated output. For LLM/MoE, derive T_max from
  // I:O ratio (default 1:1 if not provided): T_max = T × (1 + 1/ioRatio). Prefill processes T_input;
  // decode reads from T_max-sized cache; capacity reserves T_max worth of KV per user.
  const T_input = T;
  const T_max = ((workload === 'llm' || workload === 'moe') && ioRatio && ioRatio > 0)
              ? T_input * (1 + 1 / ioRatio)
              : T_input;  // training/diffusion or missing ioRatio: T is treated as full sequence
  // Scale-out: how many chips needed for model + KV cache to fit?
  // KV cache is often LARGER than the model at high batch / long context.
  // Per-token KV = 2 (K+V) * d_kv * L * kv_bytes_per_val. Total = B * T_max * per-token.
  const modelBytes = model.N_params * bytes_per_val;
  const d_kv_eff = (model.H_kv && model.d_k) ? (model.H_kv * model.d_k) : model.d;
  const kvBytes = (workload === 'llm' || workload === 'moe') ? (B * T_max * 2 * d_kv_eff * model.L * kv_bytes_per_val) : 0;
  // Training memory: weights + gradients + Adam moments (m, v, both FP32) + activations cached for backward.
  // Conservative: 2× model (weights + grads) + 8 bytes/param (Adam FP32 m+v) + ~50% B*T*d*L for activations.
  const trainingExtraBytes = (workload === 'training')
    ? (modelBytes + model.N_params * 8 + B * T * model.d * model.L * 0.5 * bytes_per_val)
    : 0;
  const totalCapacityBytes = modelBytes + kvBytes + trainingExtraBytes;
  // Two-tier memory: fast (SRAM/HBM) + optional cold (LPDDR/CXL/3D-DRAM). Capacity fit uses BOTH tiers.
  const coldCap = chip.mem_cap_cold || 0;
  const totalChipCap = chip.mem_cap + coldCap;
  let N_chips_autofit = Math.max(1, Math.ceil(totalCapacityBytes / totalChipCap));
  // SRAM-resident parts (weight_tier:'fast', e.g. d-Matrix, Cerebras-style) additionally need enough chips
  // for WEIGHTS alone to fit in the FAST tier — the architectural bet of scaling cards to keep weights hot.
  if (chip.weight_tier === 'fast') {
    N_chips_autofit = Math.max(N_chips_autofit, Math.ceil(modelBytes / chip.mem_cap));
  }
  N_chips_autofit = Math.max(1, N_chips_autofit);
  // Allow callers to over-provision (more chips than min-fit) for better latency.
  const N_chips = N_chips_override ? Math.max(N_chips_override, N_chips_autofit) : N_chips_autofit;
  // Cross-chip bandwidth: chips have an intra-rack fabric (NVLink, Infinity Fabric, Credo, etc.)
  // that's fast within a single fabric domain (rack_size chips). Beyond that, you're crossing
  // racks via Ethernet/Infiniband at much lower effective bandwidth (~100 GB/s). This penalty
  // is crucial for huge MoE workloads where many chips are needed and all-to-all dispatch dominates.
  const rackSize = chip.rack_size || 8;  // default: typical HGX/DGX size
  const intraRackBW = chip.xc_interconnect_bw || chip.interconnect_bw || 1e9;
  const interRackBW = chip.scaleout_bw || 100e9;  // multi-rack IB/Ethernet effective
  const interconnectBW = N_chips <= 1 ? (chip.interconnect_bw || 1e9)
                       : N_chips <= rackSize ? intraRackBW
                       : interRackBW;

  // ── Tiered bandwidth: weights vs KV may live in different tiers (fast SRAM/HBM vs cold LPDDR/CXL). ──
  const coldBW = chip.mem_bw_cold || 0;
  const weightBytesPerChip = modelBytes / N_chips;
  let weightBW;
  if (chip.weight_tier === 'fast' || !coldBW || weightBytesPerChip <= chip.mem_cap) {
    weightBW = memBW;  // weights hot in fast tier (or single-tier chip)
  } else {
    // weights spill into cold tier → capacity-weighted harmonic mean of the two bandwidths
    const fFast = Math.min(1, chip.mem_cap / weightBytesPerChip);
    weightBW = 1 / ((fFast / memBW) + ((1 - fFast) / coldBW));
  }
  const kvBW = coldBW || memBW;  // KV → cold tier when present (SRAM weights + LPDDR/HBM KV); else fast
  const actBW = memBW;           // transient activations stay on the fast tier
  // MoE decode under-utilization (fix M1): at low batch only a fraction of experts is touched, each by ~1
  // token (GEMV, poor HBM locality, all-to-all latency) — can't approach peak BW. Effective BW ≈ fraction
  // of the expert bank actively used. Calibrated so DeepSeek-V3 B=1 lands ~40 tok/s (was ~770 unpenalized).
  let moeDecodeEff = 1;
  if (workload === 'moe' && phase === 'decode') {
    const E = model.E || 64, K = model.K || 8;
    const expertsTouched = Math.min(E, Math.max(K, K * B));
    moeDecodeEff = Math.max(0.05, Math.min(1, expertsTouched / E));
  }

  // Per-step costs, divided by N_chips for compute & memory (TP sharding).
  let layerTime = 0;
  let totalCompute = 0, totalMem = 0, totalInterconnect = 0;
  let totalMem_weight = 0, totalMem_kv = 0, totalMem_act = 0;
  const N_STEPS = workload === 'moe' ? 21 : 16;
  // Per-step T: prefill processes T_input tokens (only the input); decode reads from a T_max-sized cache.
  // Training/diffusion: T_input = T_max so this is a no-op.
  const T_for_step = (phase === 'prefill') ? T_input : T_max;
  for (let step = 0; step < N_STEPS; step++) {
    const cost = getStepCost(step, workload, phase, model, B, T_for_step);
    // Expert FFN weight reads (MoE steps 14 & 16) take the small-batch GEMV penalty.
    const isExpertWeightStep = (workload === 'moe' && (step === 14 || step === 16));
    const wBW = isExpertWeightStep ? weightBW * moeDecodeEff : weightBW;
    const t_weight = wBW   > 0 ? (cost.bytes_weight * bytes_per_val)    / (wBW   * bwEff * N_chips) : 0;
    const t_kv     = kvBW  > 0 ? (cost.bytes_kv     * kv_bytes_per_val) / (kvBW  * bwEff * N_chips) : 0;
    const t_act    = actBW > 0 ? (cost.bytes_act    * bytes_per_val)    / (actBW * bwEff * N_chips) : 0;
    const mem_t = t_weight + t_kv + t_act;
    const compute_t = peak > 0 ? cost.flops / (peak * compEff * N_chips) : 0;
    const step_t = Math.max(compute_t, mem_t);
    totalCompute += compute_t;
    totalMem += mem_t;
    totalMem_weight += t_weight;
    totalMem_kv     += t_kv;
    totalMem_act    += t_act;
    layerTime += step_t;
  }
  // Interconnect overhead per layer. Mode determines comm pattern:
  //   - 'tp' (default): TP all-reduce of B·T·d activations × 2 per layer (attn out + FFN out).
  //                     For MoE, also adds the 2 all-to-alls for expert dispatch/gather.
  //   - 'ep': pure expert parallelism — no TP all-reduce (model wasn't TP-sharded for attention).
  //           Only the MoE all-to-all routing cost remains, much lighter than TP all-reduce.
  // Ring all-reduce: 2·(N-1)/N · data_size / BW (reduce-scatter + all-gather).
  const T_eff = (((workload === 'llm' || workload === 'moe') && phase === 'decode')) ? 1 : T;
  if (N_chips > 1) {
    if (parallelismMode === 'ep') {
      // Pure EP: only the MoE all-to-all dispatch + gather. No TP all-reduce of attention
      // activations because attention weights replicate per chip (didn't shard via TP).
      if (workload === 'moe') {
        // Ring all-to-all on switched/non-blocking fabric (NVLink, Juniper, NeuronLink, etc.):
        // per-chip outgoing bytes per dispatch = (B × T_eff × top_k × d × bytes) / N_chips.
        // Each token routes to top_k experts; tokens shard across N chips so per-chip outgoing
        // = total_routed_bytes / N. Time = per_chip_outgoing / per_chip_BW.
        const K_active = model.K || 8;
        const per_chip_outgoing = (B * T_eff * K_active * model.d * bytes_per_val) / N_chips;
        totalInterconnect = 2 * per_chip_outgoing / interconnectBW;  // dispatch + gather
      }
      // Dense in pure-EP mode shouldn't exist (pure EP requires experts) — treat as no comm.
    } else {
      // TP (default): ring all-reduce of activations × 2 per layer (after attn + after FFN)
      const allreduce_bytes = 2 * (N_chips - 1) / N_chips * B * T_eff * model.d * bytes_per_val;
      totalInterconnect = 2 * allreduce_bytes / interconnectBW;  // 2 all-reduces per layer
      // MoE adds 2 all-to-alls per layer (dispatch + gather) on top of TP all-reduce.
      // Same ring all-to-all formula as in EP branch — per-chip outgoing scales as 1/N_chips.
      if (workload === 'moe') {
        const K_active = model.K || 8;
        const per_chip_outgoing = (B * T_eff * K_active * model.d * bytes_per_val) / N_chips;
        totalInterconnect += 2 * per_chip_outgoing / interconnectBW;
      }
    }
  }
  const layerTimeWithInterconnect = layerTime + totalInterconnect;
  const totalLayerTime = layerTimeWithInterconnect * model.L;

  const roofline = peak / memBW;
  // Refined bind: split "memory" into 'weights' (incl. activations) vs 'kv' so the table
  // distinguishes "model too big for HBM" from "context too long". Interconnect wins if
  // cross-chip traffic dominates the per-layer cost.
  const totalMemWeightAct = totalMem_weight + totalMem_act;  // bundle activations with weights
  let overallBinds;
  if (totalInterconnect > 0.5 * (totalCompute + totalMem)) {
    overallBinds = 'interconnect';
  } else if (totalCompute > totalMem * 1.5) {
    overallBinds = 'compute';
  } else if (totalMem > totalCompute * 1.5) {
    // Memory-bound — pick the dominant memory category.
    overallBinds = totalMem_kv > totalMemWeightAct * 1.2 ? 'kv'
                 : totalMemWeightAct > totalMem_kv * 1.2 ? 'weights'
                 : 'weights';  // tie → call it weights (more common dominant axis)
  } else if (totalInterconnect > 0.3 * (totalCompute + totalMem)) {
    overallBinds = 'interconnect';
  } else {
    overallBinds = 'balanced';
  }

  // Workload-specific top-line metric.
  // Decode topValue is AGGREGATE tokens/sec (across all B users in the batch),
  // because the chip generates B tokens per layer-pass — not 1. This is the right
  // unit for cost comparisons; per-user latency is 1 / totalLayerTime separately.
  let topMetric = '', topValue = 0;
  if (((workload === 'llm' || workload === 'moe') && phase === 'decode')) {
    topMetric = 'tokens/sec (aggregate)';
    topValue = B / totalLayerTime;
  } else if (workload === 'llm' || workload === 'moe') {
    topMetric = 'prefill time';
    topValue = totalLayerTime;
  } else if (workload === 'training') {
    topMetric = `training tokens/sec (${phase})`;
    // Express as training-tokens per second so it slots into the same column position as decode tok/s.
    // Total tokens processed per step = B × T; divide by step time = throughput.
    topValue = totalLayerTime > 0 ? (B * T) / totalLayerTime : 0;
  } else {
    topMetric = `time per image (×${N_steps})`;
    topValue = totalLayerTime * N_steps;
  }

  return {
    chipKey, chip, peak, memBW, interconnectBW, bytes_per_val, kv_bytes_per_val,
    effPrec, kvPrec,
    N_chips, modelBytes,
    layerTime, layerTimeWithInterconnect, totalLayerTime,
    totalCompute, totalMem, totalInterconnect,
    totalMem_weight, totalMem_kv, totalMem_act,
    roofline, overallBinds,
    topMetric, topValue,
    fits_single_chip: N_chips === 1,
  };
}

// ─── Fork additions (not in original transformer_math.html) ───
// Engine extracted verbatim from reference/transformer_math.html lines 8179–8756.
// Nothing above this line was modified; exports added for testing + reuse.
export {
  modelPresets,
  chipPerfSpecs,
  getStepCost,
  computeChipSummary,
  deriveDeploymentMode,
  estimateNonExpertBytes,
  _resolveKvPrecision,
  _bytesPerValOf,
};
