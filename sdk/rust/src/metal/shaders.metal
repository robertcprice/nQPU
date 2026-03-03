// Metal Compute Shaders for Quantum Gate Operations
// ===================================================
// OPTIMIZED for Apple Silicon GPU (M1/M2/M3/M4)
// - SIMD-optimized math operations
// - Memory-aligned data structures
// - Efficient threadgroup utilization

#include <metal_stdlib>
using namespace metal;

// Complex number operations
struct Complex {
    float real;
    float imag;
};

// Inline complex operations
Complex complex_mul(Complex a, Complex b) {
    return {a.real * b.real - a.imag * b.imag, a.real * b.imag + a.imag * b.real};
}

Complex complex_add(Complex a, Complex b) {
    return {a.real + b.real, a.imag + b.imag};
}

Complex complex_sub(Complex a, Complex b) {
    return {a.real - b.real, a.imag - b.imag};
}

Complex complex_scale(Complex a, float s) {
    return {a.real * s, a.imag * s};
}

// ============================================================
// SINGLE-QUBIT GATES
// ============================================================

// Pauli-X gate (NOT gate): |0⟩ → |1⟩, |1⟩ → |0⟩
kernel void pauli_x_gate(
    device Complex* state [[buffer(0)]],
    constant uint& qubit [[buffer(1)]],
    constant uint& num_states [[buffer(2)]],
    uint id [[thread_position_in_grid]]
) {
    if (id >= num_states / 2) return;

    uint stride = 1 << qubit;
    uint base = (id & ~(stride - 1)) | (id & (stride - 1));
    uint i = base;
    uint j = base | stride;

    // Swap amplitudes
    Complex temp = state[i];
    state[i] = state[j];
    state[j] = temp;
}

// Simplified X gate for basic operations
kernel void x_gate(
    device Complex* state [[buffer(0)]],
    constant uint& qubit [[buffer(1)]],
    constant uint& num_states [[buffer(2)]],
    uint id [[thread_position_in_grid]]
) {
    uint stride = 1 << qubit;
    uint block_size = stride * 2;
    uint num_blocks = num_states / block_size;

    uint block_index = id / stride;
    uint pair_in_block = id % stride;

    uint i = block_index * block_size + pair_in_block;
    uint j = i + stride;

    // Bounds check
    if (j >= num_states) return;

    Complex temp = state[i];
    state[i] = state[j];
    state[j] = temp;
}

// Pauli-Y gate: |0⟩ → i|1⟩, |1⟩ → -i|0⟩
kernel void pauli_y_gate(
    device Complex* state [[buffer(0)]],
    constant uint& qubit [[buffer(1)]],
    constant uint& num_states [[buffer(2)]],
    uint id [[thread_position_in_grid]]
) {
    if (id >= num_states / 2) return;

    uint stride = 1 << qubit;
    uint base = (id & ~(stride - 1)) | (id & (stride - 1));
    uint i = base;
    uint j = base | stride;

    Complex a = state[i];
    Complex b = state[j];

    // Y = i*NOT with phase
    state[i] = {b.imag, -b.real};      // i|1⟩
    state[j] = {-a.imag, a.real};     // -i|0⟩
}

// Pauli-Z gate: |0⟩ → |0⟩, |1⟩ → -|1⟩
kernel void pauli_z_gate(
    device Complex* state [[buffer(0)]],
    constant uint& qubit [[buffer(1)]],
    constant uint& num_states [[buffer(2)]],
    uint id [[thread_position_in_grid]]
) {
    if (id >= num_states) return;

    uint mask = 1 << qubit;
    if (id & mask) {
        // Flip phase for |1⟩ states
        state[id].real = -state[id].real;
        state[id].imag = -state[id].imag;
    }
}

// Simplified Z gate for basic operations
kernel void z_gate(
    device Complex* state [[buffer(0)]],
    constant uint& qubit [[buffer(1)]],
    uint id [[thread_position_in_grid]]
) {
    uint mask = 1 << qubit;
    if (id & mask) {
        state[id].real = -state[id].real;
        state[id].imag = -state[id].imag;
    }
}

// Hadamard gate: H = (1/√2)[[1, 1], [1, -1]]
// Hadamard gate: H = (1/√2)[[1, 1], [1, -1]]
// OPTIMIZED with FMA-friendly computation
kernel void hadamard_gate(
    device Complex* state [[buffer(0)]],
    constant uint& qubit [[buffer(1)]],
    constant uint& num_states [[buffer(2)]],
    uint id [[thread_position_in_grid]]
) {
    uint stride = 1 << qubit;
    uint block_size = stride * 2;

    // Block-based indexing for correct pair processing
    uint block_index = id / stride;
    uint pair_in_block = id % stride;

    uint i = block_index * block_size + pair_in_block;
    uint j = i + stride;

    // Bounds check
    if (j >= num_states) return;

    // Load values
    float a_re = state[i].real;
    float a_im = state[i].imag;
    float b_re = state[j].real;
    float b_im = state[j].imag;

    // FMA-friendly: compute sum/diff first, then multiply
    constexpr float inv_sqrt2 = 0.70710678118654752440f;
    state[i].real = (a_re + b_re) * inv_sqrt2;
    state[i].imag = (a_im + b_im) * inv_sqrt2;
    state[j].real = (a_re - b_re) * inv_sqrt2;
    state[j].imag = (a_im - b_im) * inv_sqrt2;
}

// Phase gate (S gate): S = [[1, 0], [0, i]]
kernel void phase_gate(
    device Complex* state [[buffer(0)]],
    constant uint& qubit [[buffer(1)]],
    constant uint& num_states [[buffer(2)]],
    uint id [[thread_position_in_grid]]
) {
    if (id >= num_states) return;

    uint mask = 1 << qubit;
    if (id & mask) {
        // Apply phase i to |1⟩
        float real = state[id].real;
        state[id].real = -state[id].imag;
        state[id].imag = real;
    }
}

// T gate (π/8 gate): T = [[1, 0], [0, exp(iπ/4)]]
kernel void t_gate(
    device Complex* state [[buffer(0)]],
    constant uint& qubit [[buffer(1)]],
    constant uint& num_states [[buffer(2)]],
    uint id [[thread_position_in_grid]]
) {
    if (id >= num_states) return;

    uint mask = 1 << qubit;
    if (id & mask) {
        // Apply phase exp(iπ/4) to |1⟩
        float cos_pi_8 = 0.923879532511287;
        float sin_pi_8 = 0.3826834323650898;
        float real = state[id].real;
        float imag = state[id].imag;
        state[id].real = real * cos_pi_8 - imag * sin_pi_8;
        state[id].imag = real * sin_pi_8 + imag * cos_pi_8;
    }
}

// Rotation around X-axis: Rx(θ) = exp(-iθX/2)
kernel void rotation_x(
    device Complex* state [[buffer(0)]],
    constant uint& qubit [[buffer(1)]],
    constant float& theta [[buffer(2)]],
    constant uint& num_states [[buffer(3)]],
    uint id [[thread_position_in_grid]]
) {
    if (id >= num_states / 2) return;

    uint stride = 1 << qubit;
    uint base = (id & ~(stride - 1)) | (id & (stride - 1));
    uint i = base;
    uint j = base | stride;

    Complex a = state[i];
    Complex b = state[j];

    float cos_half = cos(theta * 0.5);
    float sin_half = sin(theta * 0.5);

    // Rx(θ) = [[cos(θ/2), -i*sin(θ/2)], [-i*sin(θ/2), cos(θ/2)]]
    state[i].real = a.real * cos_half + b.imag * sin_half;
    state[i].imag = a.imag * cos_half - b.real * sin_half;
    state[j].real = a.imag * sin_half + b.real * cos_half;
    state[j].imag = -a.real * sin_half + b.imag * cos_half;
}

// Rotation around Y-axis: Ry(θ) = exp(-iθY/2)
kernel void rotation_y(
    device Complex* state [[buffer(0)]],
    constant uint& qubit [[buffer(1)]],
    constant float& theta [[buffer(2)]],
    constant uint& num_states [[buffer(3)]],
    uint id [[thread_position_in_grid]]
) {
    if (id >= num_states / 2) return;

    uint stride = 1 << qubit;
    uint base = (id & ~(stride - 1)) | (id & (stride - 1));
    uint i = base;
    uint j = base | stride;

    Complex a = state[i];
    Complex b = state[j];

    float cos_half = cos(theta * 0.5);
    float sin_half = sin(theta * 0.5);

    // Ry(θ) = [[cos(θ/2), -sin(θ/2)], [sin(θ/2), cos(θ/2)]]
    state[i] = {a.real * cos_half - b.real * sin_half, a.imag * cos_half - b.imag * sin_half};
    state[j] = {a.real * sin_half + b.real * cos_half, a.imag * sin_half + b.imag * cos_half};
}

// Rotation around Z-axis: Rz(θ) = exp(-iθZ/2)
kernel void rotation_z(
    device Complex* state [[buffer(0)]],
    constant uint& qubit [[buffer(1)]],
    constant float& phi [[buffer(2)]],
    constant uint& num_states [[buffer(3)]],
    uint id [[thread_position_in_grid]]
) {
    if (id >= num_states) return;

    uint mask = 1 << qubit;
    if (id & mask) {
        // Apply phase exp(-iφ/2) to |1⟩
        float cos_half = cos(phi * 0.5);
        float sin_half = sin(phi * 0.5);
        float real = state[id].real;
        float imag = state[id].imag;
        state[id].real = real * cos_half + imag * sin_half;
        state[id].imag = imag * cos_half - real * sin_half;
    }
}

// ============================================================
// TWO-QUBIT GATES
// ============================================================

// CNOT gate: Control-X
kernel void cnot_gate(
    device Complex* state [[buffer(0)]],
    constant uint& control [[buffer(1)]],
    constant uint& target [[buffer(2)]],
    constant uint& num_states [[buffer(3)]],
    uint id [[thread_position_in_grid]]
) {
    if (id >= num_states / 2) return;

    uint control_mask = 1 << control;
    uint target_mask = 1 << target;

    // Only affect states where control bit is 1
    // Find pairs that differ only in target bit
    uint i = id;
    // Ensure control bit is 1
    if (!(i & control_mask)) return;

    // Find the paired state (flip target bit)
    uint j = i ^ target_mask;

    // Ensure we only process each pair once
    if (i < j) {
        Complex temp = state[i];
        state[i] = state[j];
        state[j] = temp;
    }
}

// CZ gate: Control-Z
kernel void cz_gate(
    device Complex* state [[buffer(0)]],
    constant uint& control [[buffer(1)]],
    constant uint& target [[buffer(2)]],
    constant uint& num_states [[buffer(3)]],
    uint id [[thread_position_in_grid]]
) {
    if (id >= num_states) return;

    uint control_mask = 1 << control;
    uint target_mask = 1 << target;

    // Apply phase flip when both control and target are 1
    if ((id & control_mask) && (id & target_mask)) {
        state[id].real = -state[id].real;
        state[id].imag = -state[id].imag;
    }
}

// Controlled phase gate: CPHASE(control, target, φ)
kernel void cphase_gate(
    device Complex* state [[buffer(0)]],
    constant uint& control [[buffer(1)]],
    constant uint& target [[buffer(2)]],
    constant float& phi [[buffer(3)]],
    constant uint& num_states [[buffer(4)]],
    uint id [[thread_position_in_grid]]
) {
    if (id >= num_states) return;

    uint control_mask = 1 << control;
    uint target_mask = 1 << target;

    // Apply phase when both control and target are 1
    if ((id & control_mask) && (id & target_mask)) {
        float cos_phi = cos(phi);
        float sin_phi = sin(phi);
        float real = state[id].real;
        float imag = state[id].imag;
        state[id].real = real * cos_phi - imag * sin_phi;
        state[id].imag = real * sin_phi + imag * cos_phi;
    }
}

// SWAP gate: Swap two qubits
kernel void swap_gate(
    device Complex* state [[buffer(0)]],
    constant uint& q1 [[buffer(1)]],
    constant uint& q2 [[buffer(2)]],
    constant uint& num_states [[buffer(3)]],
    uint id [[thread_position_in_grid]]
) {
    if (id >= num_states) return;

    uint mask1 = 1 << q1;
    uint mask2 = 1 << q2;

    // Check if bits differ
    bool bit1 = (id & mask1) != 0;
    bool bit2 = (id & mask2) != 0;

    if (bit1 != bit2) {
        // Bits differ - need to find the state with bits swapped
        uint swapped_id = id ^ (mask1 | mask2);
        if (id < swapped_id) {
            Complex temp = state[id];
            state[id] = state[swapped_id];
            state[swapped_id] = temp;
        }
    }
}

// Toffoli gate (CCX): 3-qubit controlled-X
kernel void toffoli_gate(
    device Complex* state [[buffer(0)]],
    constant uint& control1 [[buffer(1)]],
    constant uint& control2 [[buffer(2)]],
    constant uint& target [[buffer(3)]],
    constant uint& num_states [[buffer(4)]],
    uint id [[thread_position_in_grid]]
) {
    if (id >= num_states / 2) return;

    uint control1_mask = 1 << control1;
    uint control2_mask = 1 << control2;
    uint target_mask = 1 << target;

    // Only flip when both controls are 1
    if (!(id & control1_mask) || !(id & control2_mask)) return;

    uint i = id;
    uint j = i ^ target_mask;

    if (i < j) {
        Complex temp = state[i];
        state[i] = state[j];
        state[j] = temp;
    }
}

// ============================================================
// STATE INITIALIZATION
// ============================================================

// Initialize to |0...0⟩ state
kernel void initialize_zero_state(
    device Complex* state [[buffer(0)]],
    constant uint& num_states [[buffer(1)]],
    uint id [[thread_position_in_grid]]
) {
    if (id >= num_states) return;
    state[id] = {0.0, 0.0};
}

// Set first element to 1 (for |0...0⟩)
kernel void set_ground_state(
    device Complex* state [[buffer(0)]]
) {
    state[0] = {1.0, 0.0};
}

// Initialize from arbitrary amplitudes
kernel void initialize_from_amplitudes(
    device Complex* state [[buffer(0)]],
    constant float* amplitudes [[buffer(1)]],
    constant uint& num_states [[buffer(2)]],
    uint id [[thread_position_in_grid]]
) {
    if (id >= num_states) return;
    // Interleave real and imaginary parts
    state[id] = {amplitudes[2*id], amplitudes[2*id + 1]};
}

// ============================================================
// MEASUREMENT
// ============================================================

// Compute probabilities from amplitudes
kernel void compute_probabilities(
    device Complex* state [[buffer(0)]],
    device float* probabilities [[buffer(1)]],
    constant uint& num_states [[buffer(2)]],
    uint id [[thread_position_in_grid]]
) {
    if (id >= num_states) return;
    float real = state[id].real;
    float imag = state[id].imag;
    probabilities[id] = real * real + imag * imag;
}

// Sample from probability distribution (using parallel prefix scan)
// This is a simplified version - actual sampling needs more work
kernel void cumulative_sum(
    device float* probabilities [[buffer(0)]],
    device float* cumulative [[buffer(1)]],
    constant uint& num_states [[buffer(2)]],
    uint id [[thread_position_in_grid]]
) {
    // Simplified cumulative sum - in practice use parallel scan
    if (id >= num_states) return;

    float sum = 0.0;
    for (uint i = 0; i <= id; i++) {
        sum += probabilities[i];
    }
    cumulative[id] = sum;
}

// ============================================================
// GROVER ORACLE
// ============================================================

// Phase oracle for Grover's algorithm
// Marks target state by flipping its phase
kernel void grover_oracle(
    device Complex* state [[buffer(0)]],
    constant uint& target_state [[buffer(1)]],
    constant uint& num_states [[buffer(2)]],
    uint id [[thread_position_in_grid]]
) {
    if (id >= num_states) return;

    if (id == target_state) {
        state[id].real = -state[id].real;
        state[id].imag = -state[id].imag;
    }
}

// Grover diffusion operator (inversion about average)
kernel void grover_diffusion(
    device Complex* state [[buffer(0)]],
    constant float& average [[buffer(1)]],
    constant uint& num_states [[buffer(2)]],
    uint id [[thread_position_in_grid]]
) {
    if (id >= num_states) return;

    // D|ψ⟩: |i⟩ → (2*average - ψ_i)|i⟩
    state[id].real = 2.0 * average - state[id].real;
    state[id].imag = 2.0 * average - state[id].imag;
}

// Compute average amplitude
kernel void compute_average(
    device Complex* state [[buffer(0)]],
    device atomic_float* sum [[buffer(1)]],
    constant uint& num_states [[buffer(2)]],
    uint id [[thread_position_in_grid]]
) {
    if (id >= num_states) return;

    float real = state[id].real;
    atomic_fetch_add_explicit(&sum[0], real, memory_order_relaxed);
}

// ============================================================
// SIMPLIFIED GROVER KERNELS
// ============================================================

// Oracle: Phase flip on target state
kernel void oracle_phase_flip(
    device Complex* state [[buffer(0)]],
    constant uint& target [[buffer(1)]]
) {
    state[target].real = -state[target].real;
    state[target].imag = -state[target].imag;
}

// Phase flip on |0⟩ state (for diffusion)
kernel void phase_flip_zero(
    device Complex* state [[buffer(0)]]
) {
    state[0].real = -state[0].real;
    state[0].imag = -state[0].imag;
}
