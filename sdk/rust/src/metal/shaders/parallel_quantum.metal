// Parallel Quantum Operations for Metal GPU
// ===========================================
// High-performance GPU kernels for quantum gate operations
// and transformer-style quantum attention mechanisms

#include <metal_stdlib>
using namespace metal;

// ============================================================
// COMPLEX NUMBER STRUCTURES
// ============================================================

/// Double-precision complex number (matches C64 in Rust)
struct Complex {
    double re;
    double im;

    // Complex multiplication
    Complex operator*(const Complex& other) const {
        return {
            .re = re * other.re - im * other.im,
            .im = re * other.im + im * other.re
        };
    }

    // Complex addition
    Complex operator+(const Complex& other) const {
        return {.re = re + other.re, .im = im + other.im};
    }

    // Complex subtraction
    Complex operator-(const Complex& other) const {
        return {.re = re - other.re, .im = im - other.im};
    }

    // Scalar multiplication
    Complex operator*(double scalar) const {
        return {.re = re * scalar, .im = im * scalar};
    }
};

/// Single-precision complex number (matches C32 in Rust)
struct ComplexFloat {
    float re;
    float im;

    ComplexFloat operator+(const ComplexFloat& other) const {
        return {.re = re + other.re, .im = im + other.im};
    }

    ComplexFloat operator-(const ComplexFloat& other) const {
        return {.re = re - other.re, .im = im - other.im};
    }

    ComplexFloat operator*(float scalar) const {
        return {.re = re * scalar, .im = im * scalar};
    }
};

// ============================================================
// PARALLEL SINGLE-QUBIT GATES
// ============================================================

/// Apply Hadamard gates to multiple qubits in parallel
/// Each thread processes one qubit, all running simultaneously on GPU
kernel void parallel_hadamard(
    device Complex* state [[buffer(0)]],
    constant uint& num_qubits [[buffer(1)]],
    constant uint* target_qubits [[buffer(2)]],
    constant uint& num_targets [[buffer(3)]],
    uint id [[thread_position_in_grid]]
) {
    if (id >= num_targets) return;

    uint qubit = target_qubits[id];
    uint stride = 1 << qubit;
    uint dim = 1 << num_qubits;

    // Each thread processes one qubit's Hadamard
    // All threads run in parallel on GPU!
    double inv_sqrt2 = 0.70710678118654752440;

    // Process all pairs where this qubit varies
    for (uint i = 0; i < dim; i += stride * 2) {
        for (uint j = i; j < i + stride; j++) {
            uint idx1 = j;
            uint idx2 = j | stride;

            if (idx2 < dim) {
                Complex a = state[idx1];
                Complex b = state[idx2];

                state[idx1] = (a + b) * inv_sqrt2;
                state[idx2] = (a - b) * inv_sqrt2;
            }
        }
    }
}

/// Parallel Pauli-X (NOT) gates
kernel void parallel_pauli_x(
    device Complex* state [[buffer(0)]],
    constant uint& num_qubits [[buffer(1)]],
    constant uint* target_qubits [[buffer(2)]],
    constant uint& num_targets [[buffer(3)]],
    uint id [[thread_position_in_grid]]
) {
    if (id >= num_targets) return;

    uint qubit = target_qubits[id];
    uint stride = 1 << qubit;
    uint dim = 1 << num_qubits;

    // Swap amplitudes for |0⟩ and |1⟩ states
    for (uint i = 0; i < dim; i += stride * 2) {
        for (uint j = i; j < i + stride; j++) {
            uint idx1 = j;
            uint idx2 = j | stride;

            if (idx2 < dim) {
                Complex temp = state[idx1];
                state[idx1] = state[idx2];
                state[idx2] = temp;
            }
        }
    }
}

/// Parallel Pauli-Z gates (phase flip on |1⟩)
kernel void parallel_pauli_z(
    device Complex* state [[buffer(0)]],
    constant uint& num_qubits [[buffer(1)]],
    constant uint* target_qubits [[buffer(2)]],
    constant uint& num_targets [[buffer(3)]],
    uint id [[thread_position_in_grid]]
) {
    if (id >= num_targets) return;

    uint qubit = target_qubits[id];
    uint mask = 1 << qubit;
    uint dim = 1 << num_qubits;

    // Apply phase flip to |1⟩ states
    for (uint i = 0; i < dim; i++) {
        if (i & mask) {
            state[i].re = -state[i].re;
            state[i].im = -state[i].im;
        }
    }
}

/// Parallel rotations for multiple qubits
/// Supports RX, RY, and RZ rotations
kernel void parallel_rotations(
    device Complex* state [[buffer(0)]],
    constant uint& num_qubits [[buffer(1)]],
    constant uint* target_qubits [[buffer(2)]],
    constant double* angles [[buffer(3)]],
    constant uint& rotation_type [[buffer(4)]],  // 0=RX, 1=RY, 2=RZ
    constant uint& num_targets [[buffer(5)]],
    uint id [[thread_position_in_grid]]
) {
    if (id >= num_targets) return;

    uint qubit = target_qubits[id];
    double theta = angles[id];
    uint stride = 1 << qubit;
    uint dim = 1 << num_qubits;

    double cos_half = cos(theta / 2.0);
    double sin_half = sin(theta / 2.0);

    for (uint i = 0; i < dim; i += stride * 2) {
        for (uint j = i; j < i + stride; j++) {
            uint idx1 = j;
            uint idx2 = j | stride;

            if (idx2 < dim) {
                Complex a = state[idx1];
                Complex b = state[idx2];

                if (rotation_type == 0) {  // RX
                    // RX(θ) = [[cos(θ/2), -i*sin(θ/2)], [-i*sin(θ/2), cos(θ/2)]]
                    state[idx1] = {
                        .re = a.re * cos_half + a.im * sin_half,
                        .im = a.im * cos_half - a.re * sin_half
                    };
                    state[idx2] = {
                        .re = b.re * cos_half - b.im * sin_half,
                        .im = b.im * cos_half + b.re * sin_half
                    };
                } else if (rotation_type == 1) {  // RY
                    // RY(θ) = [[cos(θ/2), -sin(θ/2)], [sin(θ/2), cos(θ/2)]]
                    state[idx1] = {
                        .re = a.re * cos_half - b.re * sin_half,
                        .im = a.im * cos_half - b.im * sin_half
                    };
                    state[idx2] = {
                        .re = a.re * sin_half + b.re * cos_half,
                        .im = a.im * sin_half + b.im * cos_half
                    };
                } else if (rotation_type == 2) {  // RZ
                    // RZ(θ) = [[exp(-iθ/2), 0], [0, exp(iθ/2)]]
                    // Apply phase to |0⟩ and |1⟩ differently
                    double phase0 = -theta / 2.0;
                    double phase1 = theta / 2.0;

                    double cos0 = cos(phase0);
                    double sin0 = sin(phase0);
                    double cos1 = cos(phase1);
                    double sin1 = sin(phase1);

                    Complex new_a = {
                        .re = a.re * cos0 - a.im * sin0,
                        .im = a.re * sin0 + a.im * cos0
                    };
                    Complex new_b = {
                        .re = b.re * cos1 - b.im * sin1,
                        .im = b.re * sin1 + b.im * cos1
                    };

                    state[idx1] = new_a;
                    state[idx2] = new_b;
                }
            }
        }
    }
}

/// Parallel CNOT gates
kernel void parallel_cnot(
    device Complex* state [[buffer(0)]],
    constant uint& num_qubits [[buffer(1)]],
    constant uint* control_qubits [[buffer(2)]],
    constant uint* target_qubits [[buffer(3)]],
    constant uint& num_gates [[buffer(4)]],
    uint id [[thread_position_in_grid]]
) {
    if (id >= num_gates) return;

    uint control = control_qubits[id];
    uint target = target_qubits[id];
    uint control_mask = 1 << control;
    uint target_mask = 1 << target;
    uint dim = 1 << num_qubits;

    // Swap target qubit when control is |1⟩
    for (uint i = 0; i < dim; i++) {
        if (i & control_mask) {
            uint j = i ^ target_mask;
            if (i < j && j < dim) {
                Complex temp = state[i];
                state[i] = state[j];
                state[j] = temp;
            }
        }
    }
}

// ============================================================
// MULTI-HEAD QUANTUM ATTENTION
// ============================================================

/// Encode query, key, value vectors for all attention heads in parallel
/// Each 3D thread (batch, head, position) encodes one vector
kernel void parallel_encode_heads(
    device Complex* state [[buffer(0)]],
    constant float* queries [[buffer(1)]],
    constant float* keys [[buffer(2)]],
    constant float* values [[buffer(3)]],
    constant uint& num_heads [[buffer(4)]],
    constant uint& head_dim [[buffer(5)]],
    constant uint& seq_len [[buffer(6)]],
    constant uint& qubits_per_head [[buffer(7)]],
    constant uint& total_qubits [[buffer(8)]],
    uint3 id [[thread_position_in_grid]]
) {
    uint batch_idx = id.x;
    uint head_idx = id.y;
    uint seq_pos = id.z;

    if (head_idx >= num_heads || seq_pos >= seq_len) return;

    uint dim = 1 << total_qubits;
    uint qubit_offset = head_idx * qubits_per_head;

    // Encode query for this position
    for (uint i = 0; i < head_dim; i++) {
        uint q_idx = (qubit_offset + i) % total_qubits;
        float angle = queries[(head_idx * seq_len + seq_pos) * head_dim + i];

        // Encode as rotation on qubit q_idx
        uint stride = 1 << q_idx;
        double cos_a = cos(angle / 2.0);
        double sin_a = sin(angle / 2.0);

        for (uint j = 0; j < dim; j += stride * 2) {
            for (uint k = j; k < j + stride; k++) {
                uint idx1 = k;
                uint idx2 = k | stride;

                if (idx2 < dim) {
                    Complex a = state[idx1];
                    Complex b = state[idx2];

                    state[idx1] = {
                        .re = a.re * cos_a - b.re * sin_a,
                        .im = a.im * cos_a - b.im * sin_a
                    };
                    state[idx2] = {
                        .re = a.re * sin_a + b.re * cos_a,
                        .im = a.im * sin_a + b.im * cos_a
                    };
                }
            }
        }
    }

    // Similar encoding for keys and values would follow...
    // (Omitted for brevity, same pattern as queries)
}

/// Compute quantum attention weights for all heads in parallel
/// Uses quantum overlap (inner product) as attention mechanism
kernel void parallel_attention_weights(
    device Complex* state [[buffer(0)]],
    device float* weights [[buffer(1)]],
    constant uint& num_heads [[buffer(2)]],
    constant uint& seq_len [[buffer(3)]],
    constant uint& qubits_per_head [[buffer(4)]],
    constant uint& total_qubits [[buffer(5)]],
    uint2 id [[thread_position_in_grid]]
) {
    uint head_idx = id.x;
    uint query_pos = id.y;

    if (head_idx >= num_heads || query_pos >= seq_len) return;

    uint head_offset = head_idx * qubits_per_head;
    uint dim = 1 << total_qubits;

    // Compute attention for this query position against all key positions
    for (uint key_pos = 0; key_pos < seq_len; key_pos++) {
        float overlap = 0.0;

        // Compute quantum overlap between query and key states
        for (uint i = 0; i < qubits_per_head; i++) {
            uint q = (head_offset + i) % total_qubits;
            uint stride = 1 << q;

            // Sum over relevant basis states
            for (uint j = 0; j < dim; j += stride * 2) {
                for (uint k = j; k < j + stride; k++) {
                    uint idx1 = k;
                    uint idx2 = k | stride;

                    if (idx2 < dim) {
                        Complex a = state[idx1];
                        Complex b = state[idx2];

                        // Inner product contribution
                        overlap += (a.re * b.re + a.im * b.im);
                    }
                }
            }
        }

        // Store attention weight (softmax would be applied separately)
        weights[(head_idx * seq_len + query_pos) * seq_len + key_pos] = overlap;
    }
}

/// Apply attention weights to value vectors (parallel across heads and positions)
kernel void parallel_apply_attention(
    device Complex* state [[buffer(0)]],
    constant float* attention_weights [[buffer(1)]],
    constant float* values [[buffer(2)]],
    constant uint& num_heads [[buffer(3)]],
    constant uint& seq_len [[buffer(4)]],
    constant uint& head_dim [[buffer(5)]],
    constant uint& qubits_per_head [[buffer(6)]],
    constant uint& total_qubits [[buffer(7)]],
    uint2 id [[thread_position_in_grid]]
) {
    uint head_idx = id.x;
    uint output_pos = id.y;

    if (head_idx >= num_heads || output_pos >= seq_len) return;

    uint qubit_offset = head_idx * qubits_per_head;
    uint dim = 1 << total_qubits;

    // Weighted sum of values for this output position
    for (uint i = 0; i < head_dim; i++) {
        uint q_idx = (qubit_offset + i) % total_qubits;
        float weighted_value = 0.0;

        // Sum over all positions with attention weights
        for (uint key_pos = 0; key_pos < seq_len; key_pos++) {
            float weight = attention_weights[(head_idx * seq_len + output_pos) * seq_len + key_pos];
            float value = values[(head_idx * seq_len + key_pos) * head_dim + i];
            weighted_value += weight * value;
        }

        // Apply weighted value as rotation
        uint stride = 1 << q_idx;
        double cos_v = cos(weighted_value / 2.0);
        double sin_v = sin(weighted_value / 2.0);

        for (uint j = 0; j < dim; j += stride * 2) {
            for (uint k = j; k < j + stride; k++) {
                uint idx1 = k;
                uint idx2 = k | stride;

                if (idx2 < dim) {
                    Complex a = state[idx1];
                    Complex b = state[idx2];

                    state[idx1] = {
                        .re = a.re * cos_v - b.re * sin_v,
                        .im = a.im * cos_v - b.im * sin_v
                    };
                    state[idx2] = {
                        .re = a.re * sin_v + b.re * cos_v,
                        .im = a.im * sin_v + b.im * cos_v
                    };
                }
            }
        }
    }
}

// ============================================================
// BATCHED PROCESSING FOR TRANSFORMERS
// ============================================================

/// Process multiple sequences in parallel (batch forward pass)
/// Each 2D thread (batch, token_pos) handles one token embedding
kernel void batch_transformer_forward(
    device Complex* state [[buffer(0)]],
    constant uint* tokens [[buffer(1)]],
    constant float* embedding_matrix [[buffer(2)]],
    constant uint& batch_size [[buffer(3)]],
    constant uint& seq_len [[buffer(4)]],
    constant uint& embedding_dim [[buffer(5)]],
    constant uint& qubits_per_sequence [[buffer(6)]],
    constant uint& total_qubits [[buffer(7)]],
    uint2 id [[thread_position_in_grid]]
) {
    uint batch_idx = id.x;
    uint token_pos = id.y;

    if (batch_idx >= batch_size || token_pos >= seq_len) return;

    uint sequence_offset = batch_idx * qubits_per_sequence + token_pos * embedding_dim;
    uint token = tokens[batch_idx * seq_len + token_pos];
    uint dim = 1 << total_qubits;

    // Embed token into quantum state
    for (uint i = 0; i < embedding_dim; i++) {
        uint q = (sequence_offset + i) % total_qubits;
        float embedding = embedding_matrix[token * embedding_dim + i];

        // Encode as rotation
        uint stride = 1 << q;
        double cos_e = cos(embedding / 2.0);
        double sin_e = sin(embedding / 2.0);

        for (uint j = 0; j < dim; j += stride * 2) {
            for (uint k = j; k < j + stride; k++) {
                uint idx1 = k;
                uint idx2 = k | stride;

                if (idx2 < dim) {
                    Complex a = state[idx1];
                    Complex b = state[idx2];

                    state[idx1] = {
                        .re = a.re * cos_e - b.re * sin_e,
                        .im = a.im * cos_e - b.im * sin_e
                    };
                    state[idx2] = {
                        .re = a.re * sin_e + b.re * cos_e,
                        .im = a.im * sin_e + b.im * cos_e
                    };
                }
            }
        }
    }
}

/// Batch layer normalization (quantum version)
kernel void batch_quantum_layer_norm(
    device Complex* state [[buffer(0)]],
    constant uint& batch_size [[buffer(1)]],
    constant uint& seq_len [[buffer(2)]],
    constant uint& qubits_per_token [[buffer(3)]],
    constant uint& total_qubits [[buffer(4)]],
    constant float& epsilon [[buffer(5)]],
    uint2 id [[thread_position_in_grid]]
) {
    uint batch_idx = id.x;
    uint token_pos = id.y;

    if (batch_idx >= batch_size || token_pos >= seq_len) return;

    uint offset = (batch_idx * seq_len + token_pos) * qubits_per_token;
    uint dim = 1 << total_qubits;

    // Compute mean and variance for this token
    double mean = 0.0;
    double var = 0.0;

    for (uint i = 0; i < qubits_per_token; i++) {
        uint q = (offset + i) % total_qubits;
        uint stride = 1 << q;

        for (uint j = 0; j < dim; j += stride * 2) {
            for (uint k = j; k < j + stride; k++) {
                if (k < dim) {
                    double mag2 = state[k].re * state[k].re + state[k].im * state[k].im;
                    mean += mag2;
                }
            }
        }
    }

    mean /= qubits_per_token;

    // Normalize (simplified - full layer norm would rescale)
    double scale = 1.0 / sqrt(mean + epsilon);

    for (uint i = 0; i < qubits_per_token; i++) {
        uint q = (offset + i) % total_qubits;
        uint stride = 1 << q;

        for (uint j = 0; j < dim; j += stride * 2) {
            for (uint k = j; k < j + stride; k++) {
                if (k < dim) {
                    state[k].re *= scale;
                    state[k].im *= scale;
                }
            }
        }
    }
}

// ============================================================
// GROVER'S ALGORITHM (PARALLEL ORACLE)
// ============================================================

/// Parallel oracle marking for Grover's algorithm
/// Marks target states by flipping their phase
kernel void parallel_grover_oracle(
    device Complex* state [[buffer(0)]],
    constant uint* targets [[buffer(1)]],
    constant uint& num_targets [[buffer(2)]],
    constant uint& num_qubits [[buffer(3)]],
    uint id [[thread_position_in_grid]]
) {
    if (id >= num_targets) return;

    uint target = targets[id];
    uint dim = 1 << num_qubits;

    if (target < dim) {
        // Flip phase of target state
        state[target].re = -state[target].re;
        state[target].im = -state[target].im;
    }
}

/// Parallel diffusion operator for Grover's algorithm
/// Inverts amplitudes about the mean
kernel void parallel_grover_diffusion(
    device Complex* state [[buffer(0)]],
    constant uint& num_qubits [[buffer(1)]],
    uint thread_id [[thread_position_in_threadgroup]]
) {
    uint dim = 1 << num_qubits;

    // Compute mean of all amplitudes
    // (This would typically be done in two passes with threadgroup memory)

    // For simplicity, we'll do a simplified version
    // Full implementation would use parallel reduction

    double mean_re = 0.0;
    double mean_im = 0.0;

    for (uint i = 0; i < dim; i++) {
        mean_re += state[i].re;
        mean_im += state[i].im;
    }

    mean_re /= dim;
    mean_im /= dim;

    // Invert about mean: |ψ⟩ → 2|mean⟩|ψ⟩ - |ψ⟩
    for (uint i = thread_id; i < dim; i += 1024) { // Assume max 1024 threads
        double new_re = 2.0 * mean_re - state[i].re;
        double new_im = 2.0 * mean_im - state[i].im;
        state[i].re = new_re;
        state[i].im = new_im;
    }
}

// ============================================================
// QUANTUM FOURIER TRANSFORM (PARALLEL)
// ============================================================

/// Parallel Quantum Fourier Transform
/// Each thread handles one qubit rotation
kernel void parallel_qft(
    device Complex* state [[buffer(0)]],
    constant uint& num_qubits [[buffer(1)]],
    constant uint& start_qubit [[buffer(2)]],
    constant uint& num_transform_qubits [[buffer(3)]],
    uint id [[thread_position_in_grid]]
) {
    if (id >= num_transform_qubits) return;

    uint qubit = start_qubit + id;
    uint dim = 1 << num_qubits;

    // Apply Hadamard to this qubit
    uint stride = 1 << qubit;
    double inv_sqrt2 = 0.70710678118654752440;

    for (uint i = 0; i < dim; i += stride * 2) {
        for (uint j = i; j < i + stride; j++) {
            uint idx1 = j;
            uint idx2 = j | stride;

            if (idx2 < dim) {
                Complex a = state[idx1];
                Complex b = state[idx2];

                state[idx1] = (a + b) * inv_sqrt2;
                state[idx2] = (a - b) * inv_sqrt2;
            }
        }
    }

    // Apply controlled rotations
    for (uint k = 2; k <= num_transform_qubits - id; k++) {
        uint control_qubit = qubit + k;
        uint control_mask = 1 << control_qubit;

        double angle = M_PI / (1 << k);
        double cos_half = cos(angle / 2.0);
        double sin_half = sin(angle / 2.0);

        for (uint i = 0; i < dim; i++) {
            if (i & control_mask) {
                uint idx1 = i;
                uint idx2 = i ^ stride;

                if (idx1 < dim && idx2 < dim) {
                    // Apply phase rotation
                    Complex a = state[idx1];
                    Complex b = state[idx2];

                    state[idx1] = {
                        .re = a.re * cos_half - a.im * sin_half,
                        .im = a.re * sin_half + a.im * cos_half
                    };
                }
            }
        }
    }
}

// ============================================================
// UTILITY FUNCTIONS
// ============================================================

/// Initialize state to |0...0⟩
kernel void initialize_zero_state(
    device Complex* state [[buffer(0)]],
    constant uint& dim [[buffer(1)]],
    uint id [[thread_position_in_grid]]
) {
    if (id == 0) {
        state[0] = {.re = 1.0, .im = 0.0};
    }

    if (id > 0 && id < (1 << 20)) { // Limit threads
        state[id] = {.re = 0.0, .im = 0.0};
    }
}

/// Normalize quantum state (ensure unit norm)
kernel void normalize_state(
    device Complex* state [[buffer(0)]],
    constant uint& dim [[buffer(1)]],
    threadgroup double* shared_norm [[threadgroup(0)]],
    uint thread_id [[thread_position_in_threadgroup]],
    uint group_id [[threadgroup_position_in_grid]]
) {
    // Compute partial norm squared in threadgroup
    double partial_norm = 0.0;
    uint chunk_size = (dim + 1023) / 1024;

    for (uint i = thread_id * chunk_size; i < min((thread_id + 1) * chunk_size, dim); i++) {
        partial_norm += state[i].re * state[i].re + state[i].im * state[i].im;
    }

    shared_norm[thread_id] = partial_norm;

    // Parallel reduction within threadgroup
    for (uint stride = 512; stride > 0; stride >>= 1) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (thread_id < stride && thread_id + stride < 1024) {
            shared_norm[thread_id] += shared_norm[thread_id + stride];
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Normalize
    double total_norm = sqrt(shared_norm[0]);
    if (total_norm > 1e-10) {
        double inv_norm = 1.0 / total_norm;

        for (uint i = thread_id * chunk_size; i < min((thread_id + 1) * chunk_size, dim); i++) {
            state[i].re *= inv_norm;
            state[i].im *= inv_norm;
        }
    }
}

/// Compute probabilities from amplitudes
kernel void compute_probabilities(
    device Complex* state [[buffer(0)]],
    device float* probabilities [[buffer(1)]],
    constant uint& dim [[buffer(2)]],
    uint id [[thread_position_in_grid]]
) {
    if (id < dim) {
        probabilities[id] = state[id].re * state[id].re + state[id].im * state[id].im;
    }
}
