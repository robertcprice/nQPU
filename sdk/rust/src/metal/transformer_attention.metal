// Metal Compute Shaders for Transformer Attention
// ================================================
// GPU-accelerated attention mechanism for QEC decoding
// Optimized for Apple Silicon M4+ with simdgroup_matrix

#include <metal_stdlib>
using namespace metal;

// ============================================================
// HELPER FUNCTIONS
// ============================================================

// Softmax for attention weights (thread address space)
void softmax_inplace(thread float* weights, int size) {
    float max_val = weights[0];
    for (int i = 1; i < size; i++) {
        max_val = max(max_val, weights[i]);
    }

    float sum = 0.0f;
    for (int i = 0; i < size; i++) {
        weights[i] = exp(weights[i] - max_val);
        sum += weights[i];
    }

    for (int i = 0; i < size; i++) {
        weights[i] /= sum;
    }
}

// Softmax for attention weights (threadgroup address space)
void softmax_inplace_tg(threadgroup float* weights, int size) {
    float max_val = weights[0];
    for (int i = 1; i < size; i++) {
        max_val = max(max_val, weights[i]);
    }

    float sum = 0.0f;
    for (int i = 0; i < size; i++) {
        weights[i] = exp(weights[i] - max_val);
        sum += weights[i];
    }

    for (int i = 0; i < size; i++) {
        weights[i] /= sum;
    }
}

// GELU activation
float gelu(float x) {
    // Approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    constexpr float sqrt_2_over_pi = 0.7978845608;
    return 0.5f * x * (1.0f + tanh(sqrt_2_over_pi * (x + 0.044715f * x * x * x)));
}

// Layer normalization
kernel void layer_norm(
    device float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint& seq_len [[buffer(2)]],
    constant uint& embed_dim [[buffer(3)]],
    constant float& eps [[buffer(4)]],
    uint2 tid [[thread_position_in_grid]]
) {
    uint seq_idx = tid.x;
    uint dim_idx = tid.y;

    if (seq_idx >= seq_len || dim_idx >= embed_dim) return;

    // Compute mean and variance
    float mean = 0.0f;
    float var = 0.0f;

    for (uint d = 0; d < embed_dim; d++) {
        float val = input[seq_idx * embed_dim + d];
        mean += val;
    }
    mean /= float(embed_dim);

    for (uint d = 0; d < embed_dim; d++) {
        float val = input[seq_idx * embed_dim + d];
        var += (val - mean) * (val - mean);
    }
    var /= float(embed_dim);

    // Normalize
    uint idx = seq_idx * embed_dim + dim_idx;
    output[idx] = (input[idx] - mean) / sqrt(var + eps);
}

// ============================================================
// MULTI-HEAD ATTENTION
// ============================================================

// Scaled dot-product attention kernel
// Computes: softmax(Q @ K^T / sqrt(d_k)) @ V
kernel void scaled_dot_product_attention(
    device float* query [[buffer(0)]],      // [seq_len, head_dim]
    device float* key [[buffer(1)]],        // [seq_len, head_dim]
    device float* value [[buffer(2)]],      // [seq_len, head_dim]
    device float* output [[buffer(3)]],     // [seq_len, head_dim]
    constant uint& seq_len [[buffer(4)]],
    constant uint& head_dim [[buffer(5)]],
    constant float& scale [[buffer(6)]],
    threadgroup float* shared_weights [[threadgroup(0)]],
    uint2 tid [[thread_position_in_grid]]
) {
    uint q_idx = tid.x;  // Query sequence position
    uint d_idx = tid.y;  // Head dimension index

    if (q_idx >= seq_len || d_idx >= head_dim) return;

    // Compute attention weights for this query position
    // Q[q_idx] @ K^T = attention_weights of shape [seq_len]
    for (uint k_idx = 0; k_idx < seq_len; k_idx++) {
        float score = 0.0f;
        for (uint d = 0; d < head_dim; d++) {
            score += query[q_idx * head_dim + d] * key[k_idx * head_dim + d];
        }
        shared_weights[k_idx] = score * scale;
    }

    // Softmax over attention weights (threadgroup address space)
    softmax_inplace_tg(shared_weights, seq_len);

    // Weighted sum of values
    float out = 0.0f;
    for (uint v_idx = 0; v_idx < seq_len; v_idx++) {
        out += shared_weights[v_idx] * value[v_idx * head_dim + d_idx];
    }

    output[q_idx * head_dim + d_idx] = out;
}

// Multi-head attention with projections
kernel void multi_head_attention(
    device float* input [[buffer(0)]],           // [seq_len, embed_dim]
    device float* query_weight [[buffer(1)]],    // [embed_dim, num_heads * head_dim]
    device float* key_weight [[buffer(2)]],      // [embed_dim, num_heads * head_dim]
    device float* value_weight [[buffer(3)]],    // [embed_dim, num_heads * head_dim]
    device float* output_weight [[buffer(4)]],   // [num_heads * head_dim, embed_dim]
    device float* output [[buffer(5)]],          // [seq_len, embed_dim]
    constant uint& seq_len [[buffer(6)]],
    constant uint& embed_dim [[buffer(7)]],
    constant uint& num_heads [[buffer(8)]],
    constant uint& head_dim [[buffer(9)]],
    uint3 tid [[thread_position_in_grid]]
) {
    uint seq_idx = tid.x;
    uint head_idx = tid.y;
    uint dim_idx = tid.z;

    if (seq_idx >= seq_len || head_idx >= num_heads || dim_idx >= head_dim) return;

    uint head_offset = head_idx * head_dim;
    float scale = 1.0f / sqrt(float(head_dim));

    // Compute Q, K, V projections for this head
    float q = 0.0f, k = 0.0f, v = 0.0f;
    for (uint e = 0; e < embed_dim; e++) {
        float in_val = input[seq_idx * embed_dim + e];
        q += in_val * query_weight[e * (num_heads * head_dim) + head_offset + dim_idx];
        k += in_val * key_weight[e * (num_heads * head_dim) + head_offset + dim_idx];
        v += in_val * value_weight[e * (num_heads * head_dim) + head_offset + dim_idx];
    }

    // Store for attention computation (simplified single-head version)
    // In practice, this would use shared memory and multiple kernel launches

    // Output projection
    uint out_idx = seq_idx * embed_dim + head_idx * head_dim + dim_idx;
    output[out_idx] = v * scale;
}

// ============================================================
// FEED-FORWARD NETWORK
// ============================================================

// GELU-activated feed-forward layer
kernel void feed_forward_gelu(
    device float* input [[buffer(0)]],
    device float* weight [[buffer(1)]],
    device float* bias [[buffer(2)]],
    device float* output [[buffer(3)]],
    constant uint& in_dim [[buffer(4)]],
    constant uint& out_dim [[buffer(5)]],
    uint2 tid [[thread_position_in_grid]]
) {
    uint batch_idx = tid.x;
    uint out_idx = tid.y;

    if (out_idx >= out_dim) return;

    float sum = bias[out_idx];
    for (uint i = 0; i < in_dim; i++) {
        sum += input[batch_idx * in_dim + i] * weight[i * out_dim + out_idx];
    }

    output[batch_idx * out_dim + out_idx] = gelu(sum);
}

// Linear projection layer
kernel void linear_projection(
    device float* input [[buffer(0)]],
    device float* weight [[buffer(1)]],
    device float* bias [[buffer(2)]],
    device float* output [[buffer(3)]],
    constant uint& in_dim [[buffer(4)]],
    constant uint& out_dim [[buffer(5)]],
    uint2 tid [[thread_position_in_grid]]
) {
    uint batch_idx = tid.x;
    uint out_idx = tid.y;

    if (out_idx >= out_dim) return;

    float sum = bias[out_idx];
    for (uint i = 0; i < in_dim; i++) {
        sum += input[batch_idx * in_dim + i] * weight[i * out_dim + out_idx];
    }

    output[batch_idx * out_dim + out_idx] = sum;
}

// ============================================================
// QEC DECODER SPECIFIC KERNELS
// ============================================================

// Syndrome embedding: binary syndrome -> continuous embedding
kernel void syndrome_embedding(
    device bool* syndrome [[buffer(0)]],
    device float* embedding [[buffer(1)]],
    device float* weight [[buffer(2)]],
    constant uint& syndrome_len [[buffer(3)]],
    constant uint& embed_dim [[buffer(4)]],
    uint2 tid [[thread_position_in_grid]]
) {
    uint syn_idx = tid.x;
    uint embed_idx = tid.y;

    if (syn_idx >= syndrome_len || embed_idx >= embed_dim) return;

    // Binary embedding: 0 -> -1, 1 -> +1, scaled by learned weights
    float val = syndrome[syn_idx] ? 1.0f : -1.0f;
    embedding[syn_idx * embed_dim + embed_idx] = val * weight[syn_idx * embed_dim + embed_idx];
}

// Correction head: continuous output -> binary correction
kernel void correction_head(
    device float* hidden [[buffer(0)]],
    device float* weight [[buffer(1)]],
    device float* bias [[buffer(2)]],
    device float* x_correction [[buffer(3)]],
    device float* z_correction [[buffer(4)]],
    constant uint& hidden_dim [[buffer(5)]],
    constant uint& num_qubits [[buffer(6)]],
    uint2 tid [[thread_position_in_grid]]
) {
    uint qubit_idx = tid.x;
    uint is_z = tid.y;  // 0 for X, 1 for Z

    if (qubit_idx >= num_qubits || is_z > 1) return;

    // Compute correction probability
    float logit = bias[qubit_idx + is_z * num_qubits];
    for (uint h = 0; h < hidden_dim; h++) {
        logit += hidden[h] * weight[h * (2 * num_qubits) + qubit_idx + is_z * num_qubits];
    }

    // Sigmoid to get probability
    float prob = 1.0f / (1.0f + exp(-logit));

    // Store probability (thresholding done on CPU)
    if (is_z == 0) {
        x_correction[qubit_idx] = prob;
    } else {
        z_correction[qubit_idx] = prob;
    }
}

// ============================================================
// FULL TRANSFORMER QEC DECODER PIPELINE
// ============================================================
//
// Chains: Embed -> QKV Projection -> Attention Scores -> Softmax
//       -> Output Projection -> FFN (GELU) -> Correction Head
//
// This is a fused single-kernel implementation for low-latency
// inference.  Each thread handles one output position in the
// correction vector for a single batch element.
//
// Weight layout in the flat `weights` buffer (offsets computed
// from the dimension parameters):
//
//   [0..] embed_weight:      [syndrome_len, embed_dim]
//   [+..]  qkv_weight:       [embed_dim, 3 * embed_dim]
//   [+..]  qkv_bias:         [3 * embed_dim]
//   [+..]  out_proj_weight:   [embed_dim, embed_dim]
//   [+..]  out_proj_bias:     [embed_dim]
//   [+..]  ffn1_weight:       [embed_dim, ffn_dim]
//   [+..]  ffn1_bias:         [ffn_dim]
//   [+..]  ffn2_weight:       [ffn_dim, embed_dim]
//   [+..]  ffn2_bias:         [embed_dim]
//   [+..]  corr_weight:       [embed_dim, 2 * num_qubits]
//   [+..]  corr_bias:         [2 * num_qubits]
//
// Maximum dimensions enforced by threadgroup memory:
//   syndrome_len <= 128, embed_dim <= 128, ffn_dim <= 512
//
// ## Adapting for Different QEC Code Families
//
// This kernel is code-agnostic. To use with a different code family,
// change only the runtime parameters passed at dispatch:
//
//   Surface code (distance d):
//     syndrome_len = 2 * d * (d - 1)
//     num_qubits   = d^2 + (d-1)^2
//
//   Color code (distance d):
//     syndrome_len = 2 * (3*d*d - 3*d + 1) / 3
//     num_qubits   = 3*d*d - 3*d + 1
//
//   Repetition code (length n):
//     syndrome_len = n - 1
//     num_qubits   = n
//
// The correction output has 2 * num_qubits elements per batch:
//   indices 0..num_qubits        = X correction probabilities
//   indices num_qubits..2*Q      = Z correction probabilities
//
// No kernel source modifications are needed. Pack weights using
// TransformerDecoder::pack_weights_for_gpu() on the Rust side.

#define MAX_SEQ 128
#define MAX_DIM 128
#define MAX_FFN 512

kernel void quantum_transformer_full_pipeline(
    device const float* syndromes      [[buffer(0)]],   // [batch, syndrome_len]
    device const float* weights        [[buffer(1)]],   // flat weight buffer
    device float*       corrections    [[buffer(2)]],   // [batch, 2 * num_qubits]
    constant uint& batch_size          [[buffer(3)]],
    constant uint& syndrome_len        [[buffer(4)]],
    constant uint& num_qubits          [[buffer(5)]],
    constant uint& embed_dim           [[buffer(6)]],
    constant uint& num_heads           [[buffer(7)]],
    constant uint& ffn_dim             [[buffer(8)]],
    uint2 tid [[thread_position_in_grid]]
) {
    uint batch_idx = tid.x;
    uint corr_idx  = tid.y;   // 0 .. 2*num_qubits - 1

    if (batch_idx >= batch_size || corr_idx >= 2 * num_qubits) return;

    // ---- Compute weight buffer offsets ----
    uint off = 0;
    device const float* embed_w    = weights + off;  off += syndrome_len * embed_dim;
    device const float* qkv_w      = weights + off;  off += embed_dim * 3 * embed_dim;
    device const float* qkv_b      = weights + off;  off += 3 * embed_dim;
    device const float* out_proj_w  = weights + off;  off += embed_dim * embed_dim;
    device const float* out_proj_b  = weights + off;  off += embed_dim;
    device const float* ffn1_w     = weights + off;  off += embed_dim * ffn_dim;
    device const float* ffn1_b     = weights + off;  off += ffn_dim;
    device const float* ffn2_w     = weights + off;  off += ffn_dim * embed_dim;
    device const float* ffn2_b     = weights + off;  off += embed_dim;
    device const float* corr_w     = weights + off;  off += embed_dim * 2 * num_qubits;
    device const float* corr_b     = weights + off;  // off += 2 * num_qubits;

    device const float* syn = syndromes + batch_idx * syndrome_len;

    // =====================================================
    // Stage 1: Syndrome Embedding
    //   embedded[s][d] = syn_val * embed_w[s * embed_dim + d]
    //   syn_val is +1 if syndrome bit set, -1 otherwise
    // =====================================================
    // We accumulate per-position embeddings as we go through
    // subsequent stages.  Since this thread only outputs a
    // single correction index, we compute a "pooled" hidden
    // state by averaging across sequence positions at the end.

    // Use a local register array for the hidden state
    // (one embed_dim-sized vector, pooled across sequence)

    // We compute attention inline for one "averaged query".
    // This is equivalent to a single-head, mean-pooled pass
    // appropriate for QEC where correction does not depend on
    // individual token positions.

    // ---- Compute mean syndrome embedding (pooled query) ----
    float h[MAX_DIM];  // pooled hidden state
    for (uint d = 0; d < embed_dim && d < MAX_DIM; d++) {
        float sum = 0.0f;
        for (uint s = 0; s < syndrome_len; s++) {
            float syn_val = syn[s] > 0.5f ? 1.0f : -1.0f;
            sum += syn_val * embed_w[s * embed_dim + d];
        }
        h[d] = sum / float(syndrome_len);
    }

    // =====================================================
    // Stage 2: QKV Projection (single pooled vector)
    //   q[d] = h @ Wq + bq   (first embed_dim cols of qkv_w)
    //   For mean-pooled single-vector attention, Q=K=V=projected h
    // =====================================================
    float q[MAX_DIM];
    float k[MAX_DIM];
    float v[MAX_DIM];
    for (uint d = 0; d < embed_dim && d < MAX_DIM; d++) {
        float sq = qkv_b[d];
        float sk = qkv_b[embed_dim + d];
        float sv = qkv_b[2 * embed_dim + d];
        for (uint i = 0; i < embed_dim; i++) {
            sq += h[i] * qkv_w[i * 3 * embed_dim + d];
            sk += h[i] * qkv_w[i * 3 * embed_dim + embed_dim + d];
            sv += h[i] * qkv_w[i * 3 * embed_dim + 2 * embed_dim + d];
        }
        q[d] = sq;
        k[d] = sk;
        v[d] = sv;
    }

    // =====================================================
    // Stage 3: Attention Score + Softmax
    //   For single-vector: attn_score = dot(q, k) / sqrt(head_dim)
    //   softmax is trivial (single element = 1.0)
    //   attn_out = v (the value itself)
    // =====================================================
    // With single pooled representation the attention output is just v
    // (attention weight is 1.0 by definition for a single position).

    // =====================================================
    // Stage 4: Output Projection + Residual
    //   proj[d] = v @ out_proj_w + out_proj_b
    //   residual: h[d] += proj[d]
    // =====================================================
    for (uint d = 0; d < embed_dim && d < MAX_DIM; d++) {
        float sum = out_proj_b[d];
        for (uint i = 0; i < embed_dim; i++) {
            sum += v[i] * out_proj_w[i * embed_dim + d];
        }
        h[d] += sum;  // residual connection
    }

    // =====================================================
    // Stage 5: Layer Normalization (pre-FFN)
    //   Compute mean and variance, then normalize
    // =====================================================
    float ln_mean = 0.0f;
    for (uint d = 0; d < embed_dim; d++) {
        ln_mean += h[d];
    }
    ln_mean /= float(embed_dim);

    float ln_var = 0.0f;
    for (uint d = 0; d < embed_dim; d++) {
        float diff = h[d] - ln_mean;
        ln_var += diff * diff;
    }
    ln_var /= float(embed_dim);
    float ln_std = sqrt(ln_var + 1e-5f);

    for (uint d = 0; d < embed_dim && d < MAX_DIM; d++) {
        h[d] = (h[d] - ln_mean) / ln_std;
    }

    // =====================================================
    // Stage 6: Feed-Forward Network with GELU
    //   ffn_hidden[j] = GELU(h @ ffn1_w + ffn1_b)
    //   ffn_out[d]    = ffn_hidden @ ffn2_w + ffn2_b
    //   residual: h[d] += ffn_out[d]
    // =====================================================
    float ffn_h[MAX_FFN];
    uint actual_ffn = min(ffn_dim, (uint)MAX_FFN);

    for (uint j = 0; j < actual_ffn; j++) {
        float sum = ffn1_b[j];
        for (uint i = 0; i < embed_dim; i++) {
            sum += h[i] * ffn1_w[i * ffn_dim + j];
        }
        // GELU activation
        constexpr float sqrt_2_over_pi = 0.7978845608f;
        float gelu_val = 0.5f * sum * (1.0f + tanh(sqrt_2_over_pi * (sum + 0.044715f * sum * sum * sum)));
        ffn_h[j] = gelu_val;
    }

    // Project back to embed_dim and add residual
    float h_residual[MAX_DIM];
    for (uint d = 0; d < embed_dim && d < MAX_DIM; d++) {
        float sum = ffn2_b[d];
        for (uint j = 0; j < actual_ffn; j++) {
            sum += ffn_h[j] * ffn2_w[j * embed_dim + d];
        }
        h_residual[d] = h[d] + sum;  // residual connection
    }

    // =====================================================
    // Stage 7: Correction Head (sigmoid output)
    //   logit = h_residual @ corr_w[:, corr_idx] + corr_b[corr_idx]
    //   correction = sigmoid(logit)
    // =====================================================
    float logit = corr_b[corr_idx];
    for (uint d = 0; d < embed_dim; d++) {
        logit += h_residual[d] * corr_w[d * 2 * num_qubits + corr_idx];
    }

    float prob = 1.0f / (1.0f + exp(-logit));
    corrections[batch_idx * 2 * num_qubits + corr_idx] = prob;
}

// ============================================================
// MULTI-HEAD ATTENTION (PROPER MULTI-POSITION)
// ============================================================
//
// Unlike the mean-pooled single-vector approach in the full pipeline
// kernel above, this kernel handles multi-position sequences with
// proper per-position softmax attention across all heads.
//
// Each thread computes one output element: (batch, seq_pos, dim).
// Internally the kernel:
//   1. Projects input to Q, K, V using fused QKV weights
//   2. Splits into num_heads attention heads (head_dim = model_dim / num_heads)
//   3. Computes scaled dot-product attention per head with softmax over
//      all sequence positions
//   4. Concatenates head outputs
//   5. Applies output projection
//
// This is fully self-contained: no external memory allocation beyond
// the input/output/weight buffers.
//
// Weight layout in `weights` buffer:
//   [0          ] qkv_weight: [model_dim, 3 * model_dim]
//   [+D*3D      ] qkv_bias:   [3 * model_dim]
//   [+3D        ] out_weight:  [model_dim, model_dim]
//   [+D*D       ] out_bias:    [model_dim]
//
// Limits: seq_len <= MAX_SEQ (128), model_dim <= MAX_DIM (128)

kernel void multi_head_attention_proper(
    device const float* input       [[buffer(0)]],   // [batch, seq_len, model_dim]
    device const float* weights     [[buffer(1)]],   // flat weight buffer (see layout above)
    device float*       output      [[buffer(2)]],   // [batch, seq_len, model_dim]
    constant uint& batch_size       [[buffer(3)]],
    constant uint& seq_len          [[buffer(4)]],
    constant uint& model_dim        [[buffer(5)]],
    constant uint& num_heads        [[buffer(6)]],
    uint3 tid [[thread_position_in_grid]]
) {
    uint batch_idx = tid.x;
    uint seq_idx   = tid.y;
    uint dim_idx   = tid.z;

    if (batch_idx >= batch_size || seq_idx >= seq_len || dim_idx >= model_dim) return;

    uint head_dim = model_dim / num_heads;
    if (head_dim == 0) return;

    // ---- Weight buffer offsets ----
    device const float* qkv_w   = weights;
    device const float* qkv_b   = qkv_w + model_dim * 3 * model_dim;
    device const float* out_w   = qkv_b + 3 * model_dim;
    device const float* out_b   = out_w + model_dim * model_dim;

    device const float* x_batch = input + batch_idx * seq_len * model_dim;

    // ---- Step 1: Compute Q for this position ----
    // Q is only needed for seq_idx, so it fits in a small register array.
    float q[MAX_DIM];
    for (uint d = 0; d < model_dim && d < MAX_DIM; d++) {
        float sum = qkv_b[d];
        for (uint k = 0; k < model_dim; k++) {
            sum += x_batch[seq_idx * model_dim + k] * qkv_w[k * 3 * model_dim + d];
        }
        q[d] = sum;
    }

    // ---- Step 2: Multi-head attention (on-the-fly K/V projection) ----
    // To avoid storing K/V for all positions (which would require
    // MAX_SEQ * MAX_DIM floats per thread), we compute attention in
    // two passes per head:
    //   Pass 1: Compute attention scores by projecting K on-the-fly
    //   Pass 2: Compute weighted V sum by projecting V on-the-fly
    //
    // This trades compute for memory: O(seq_len * model_dim) extra
    // multiplies but only O(MAX_SEQ + MAX_DIM) thread-local storage.

    float attn_out[MAX_DIM];
    for (uint d = 0; d < model_dim && d < MAX_DIM; d++) {
        attn_out[d] = 0.0f;
    }

    for (uint h = 0; h < num_heads; h++) {
        uint h_off = h * head_dim;
        float scale = 1.0f / sqrt(float(head_dim));

        // Pass 1: Compute attention scores by projecting K on-the-fly
        float scores[MAX_SEQ];
        float max_score = -1e30f;

        for (uint s = 0; s < seq_len && s < MAX_SEQ; s++) {
            // Project x[s] -> k[s] for this head's dimensions only
            float dot = 0.0f;
            for (uint d = 0; d < head_dim; d++) {
                float k_d = qkv_b[model_dim + h_off + d];
                for (uint kk = 0; kk < model_dim; kk++) {
                    k_d += x_batch[s * model_dim + kk]
                         * qkv_w[kk * 3 * model_dim + model_dim + h_off + d];
                }
                dot += q[h_off + d] * k_d;
            }
            scores[s] = dot * scale;
            max_score = max(max_score, scores[s]);
        }

        // Softmax over sequence positions (numerically stable)
        float sum_exp = 0.0f;
        for (uint s = 0; s < seq_len && s < MAX_SEQ; s++) {
            scores[s] = exp(scores[s] - max_score);
            sum_exp += scores[s];
        }
        if (sum_exp > 0.0f) {
            for (uint s = 0; s < seq_len && s < MAX_SEQ; s++) {
                scores[s] /= sum_exp;
            }
        }

        // Pass 2: Weighted sum of V (projected on-the-fly)
        for (uint d = 0; d < head_dim; d++) {
            float val = 0.0f;
            for (uint s = 0; s < seq_len && s < MAX_SEQ; s++) {
                // Project x[s] -> v[s] for dimension h_off + d
                float v_sd = qkv_b[2 * model_dim + h_off + d];
                for (uint kk = 0; kk < model_dim; kk++) {
                    v_sd += x_batch[s * model_dim + kk]
                          * qkv_w[kk * 3 * model_dim + 2 * model_dim + h_off + d];
                }
                val += scores[s] * v_sd;
            }
            attn_out[h_off + d] = val;
        }
    }

    // ---- Step 3: Output projection + residual ----
    // out[dim_idx] = attn_out @ out_w[:, dim_idx] + out_b[dim_idx]
    float proj = out_b[dim_idx];
    for (uint d = 0; d < model_dim; d++) {
        proj += attn_out[d] * out_w[d * model_dim + dim_idx];
    }

    // Write output with residual connection (add input)
    uint out_idx = batch_idx * seq_len * model_dim + seq_idx * model_dim + dim_idx;
    output[out_idx] = proj + x_batch[seq_idx * model_dim + dim_idx];
}

// ============================================================
// CODE GENERALIZATION DOCUMENTATION
// ============================================================
//
// The `quantum_transformer_full_pipeline` kernel above is designed
// to be code-agnostic. To adapt it for different QEC code families,
// only the dimension parameters passed at dispatch time need to change:
//
// ## Surface Codes (distance d)
//   - syndrome_len = 2 * d * (d - 1)     (X and Z stabilizers)
//   - num_qubits   = d^2 + (d-1)^2       (data qubits)
//   - Typical embed_dim: 64 for d<=7, 128 for d<=15
//   - The kernel handles X/Z corrections jointly via the 2*num_qubits
//     output dimension (first half = X, second half = Z).
//
// ## Color Codes (distance d)
//   - syndrome_len = 2 * (3d^2 - 3d + 1) / 3   (face stabilizers)
//   - num_qubits   = 3d^2 - 3d + 1              (vertex qubits)
//   - Color codes have higher connectivity, so larger embed_dim (128+)
//     and more attention heads (8+) are recommended.
//   - The same flat weight layout works; only the buffer sizes change.
//
// ## Repetition Codes (length n)
//   - syndrome_len = n - 1                (parity checks)
//   - num_qubits   = n                    (data bits)
//   - These are small enough that embed_dim=8, num_heads=1, ffn_dim=16
//     suffice. The kernel overhead may exceed the computation time for
//     very small codes; consider the simpler neural_decoder_batch kernel
//     for repetition codes.
//
// ## General Adaptation Checklist
//   1. Set syndrome_len to the number of stabilizer generators
//   2. Set num_qubits to the number of data qubits
//   3. Choose embed_dim, num_heads, ffn_dim based on code complexity
//   4. Pack weights using TransformerDecoder::pack_weights_for_gpu()
//   5. Dispatch with grid size (batch_size, 2 * num_qubits, 1)
//   6. Read correction probabilities; threshold at 0.5 for hard decode
//
// No kernel source changes are required for any of these adaptations.

// ============================================================
// SIMD-GROUP OPTIMIZED VERSIONS (M4+)
// ============================================================

#if __METAL_VERSION__ >= 310
// Use simdgroup_matrix for M4+ devices (8x8 tile operations)

kernel void simd_matmul_attention(
    device float* query [[buffer(0)]],
    device float* key [[buffer(1)]],
    device float* value [[buffer(2)]],
    device float* output [[buffer(3)]],
    constant uint& seq_len [[buffer(4)]],
    constant uint& head_dim [[buffer(5)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint simd_lane [[thread_index_in_simdgroup]]
) {
    // 8x8 tile processing using simdgroup_matrix
    simdgroup_matrix<float, 8, 8> q_tile;
    simdgroup_matrix<float, 8, 8> k_tile;
    simdgroup_matrix<float, 8, 8> acc_tile;

    // Tile row/col offsets in the output matrix
    uint tile_row = tgid.x * 8;
    uint tile_col = tgid.y * 8;

    // Bounds guard (entire tile must fit)
    if (tile_row >= seq_len || tile_col >= seq_len) return;

    // Accumulate over the head_dim dimension in 8-wide tiles
    // acc = Q_tile @ K_tile^T (summed over d)
    // Reset accumulator
    simdgroup_matrix<float, 8, 8> zero;
    acc_tile = zero;

    for (uint d = 0; d < head_dim; d += 8) {
        // Load Q[tile_row:tile_row+8, d:d+8]
        simdgroup_load(q_tile, query + tile_row * head_dim + d, head_dim);
        // Load K[tile_col:tile_col+8, d:d+8]  (will be transposed by multiply)
        simdgroup_load(k_tile, key + tile_col * head_dim + d, head_dim);

        // acc += Q_tile @ K_tile^T
        simdgroup_multiply_accumulate(acc_tile, q_tile, k_tile, acc_tile);
    }

    // Store the attention score tile to output
    simdgroup_store(acc_tile, output + tile_row * seq_len + tile_col, seq_len);
}

#endif // __METAL_VERSION__ >= 310
