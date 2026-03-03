// nQPU-Metal Quantum Gate Compute Shaders
// =========================================
// Correct, high-performance Metal GPU kernels for Apple Silicon.
//
// Design:
//   - f32 complex arithmetic (Metal doesn't support f64 compute)
//   - Block-based pair indexing for pair gates (dim/2 threads)
//   - Simple masking for diagonal gates (dim threads)
//   - All kernels have bounds checks for non-power-of-2 threadgroups
//
// Thread dispatch guide:
//   Pair gates (H, X, Y, Rx, Ry):  grid = dim/2
//   Diagonal gates (Z, S, T, Rz):  grid = dim
//   CNOT:                           grid = dim
//   CZ, CPhase:                     grid = dim
//   SWAP:                           grid = dim
//   Toffoli:                        grid = dim
//   Probabilities:                  grid = dim
//   Generic unitary:                grid = dim/2

#include <metal_stdlib>
using namespace metal;

struct Complex {
    float real;
    float imag;
};

// ============================================================
// PAIR-INDEXED SINGLE-QUBIT GATES (grid = dim/2)
// ============================================================
// Maps thread id in [0, dim/2) to a unique (i, j) pair where
// j = i + stride, stride = 1 << qubit.

kernel void gpu_hadamard(
    device Complex* state [[buffer(0)]],
    constant uint& qubit [[buffer(1)]],
    constant uint& num_states [[buffer(2)]],
    uint id [[thread_position_in_grid]]
) {
    uint half_dim = num_states >> 1;
    if (id >= half_dim) return;

    uint stride = 1u << qubit;
    uint block  = stride << 1;
    uint i = (id / stride) * block + (id % stride);
    uint j = i + stride;

    float a_re = state[i].real;
    float a_im = state[i].imag;
    float b_re = state[j].real;
    float b_im = state[j].imag;

    constexpr float inv_sqrt2 = 0.70710678118654752440f;
    state[i].real = (a_re + b_re) * inv_sqrt2;
    state[i].imag = (a_im + b_im) * inv_sqrt2;
    state[j].real = (a_re - b_re) * inv_sqrt2;
    state[j].imag = (a_im - b_im) * inv_sqrt2;
}

kernel void gpu_pauli_x(
    device Complex* state [[buffer(0)]],
    constant uint& qubit [[buffer(1)]],
    constant uint& num_states [[buffer(2)]],
    uint id [[thread_position_in_grid]]
) {
    uint half_dim = num_states >> 1;
    if (id >= half_dim) return;

    uint stride = 1u << qubit;
    uint block  = stride << 1;
    uint i = (id / stride) * block + (id % stride);
    uint j = i + stride;

    Complex temp = state[i];
    state[i] = state[j];
    state[j] = temp;
}

kernel void gpu_pauli_y(
    device Complex* state [[buffer(0)]],
    constant uint& qubit [[buffer(1)]],
    constant uint& num_states [[buffer(2)]],
    uint id [[thread_position_in_grid]]
) {
    uint half_dim = num_states >> 1;
    if (id >= half_dim) return;

    uint stride = 1u << qubit;
    uint block  = stride << 1;
    uint i = (id / stride) * block + (id % stride);
    uint j = i + stride;

    Complex a = state[i];
    Complex b = state[j];

    // Y|0> = i|1>, Y|1> = -i|0>
    state[i] = {b.imag, -b.real};
    state[j] = {-a.imag, a.real};
}

kernel void gpu_rotation_x(
    device Complex* state [[buffer(0)]],
    constant uint& qubit [[buffer(1)]],
    constant float& theta [[buffer(2)]],
    constant uint& num_states [[buffer(3)]],
    uint id [[thread_position_in_grid]]
) {
    uint half_dim = num_states >> 1;
    if (id >= half_dim) return;

    uint stride = 1u << qubit;
    uint block  = stride << 1;
    uint i = (id / stride) * block + (id % stride);
    uint j = i + stride;

    Complex a = state[i];
    Complex b = state[j];

    float c = cos(theta * 0.5f);
    float s = sin(theta * 0.5f);

    // Rx = [[cos, -i*sin], [-i*sin, cos]]
    state[i].real = a.real * c + b.imag * s;
    state[i].imag = a.imag * c - b.real * s;
    state[j].real = a.imag * s + b.real * c;
    state[j].imag = -a.real * s + b.imag * c;
}

kernel void gpu_rotation_y(
    device Complex* state [[buffer(0)]],
    constant uint& qubit [[buffer(1)]],
    constant float& theta [[buffer(2)]],
    constant uint& num_states [[buffer(3)]],
    uint id [[thread_position_in_grid]]
) {
    uint half_dim = num_states >> 1;
    if (id >= half_dim) return;

    uint stride = 1u << qubit;
    uint block  = stride << 1;
    uint i = (id / stride) * block + (id % stride);
    uint j = i + stride;

    Complex a = state[i];
    Complex b = state[j];

    float c = cos(theta * 0.5f);
    float s = sin(theta * 0.5f);

    // Ry = [[cos, -sin], [sin, cos]]
    state[i] = {a.real * c - b.real * s, a.imag * c - b.imag * s};
    state[j] = {a.real * s + b.real * c, a.imag * s + b.imag * c};
}

// Generic 2x2 unitary (for fused gates): grid = dim/2
struct Matrix2x2 {
    float m00_re; float m00_im; float m01_re; float m01_im;
    float m10_re; float m10_im; float m11_re; float m11_im;
};

kernel void gpu_unitary(
    device Complex* state [[buffer(0)]],
    constant uint& qubit [[buffer(1)]],
    constant Matrix2x2& mat [[buffer(2)]],
    constant uint& num_states [[buffer(3)]],
    uint id [[thread_position_in_grid]]
) {
    uint half_dim = num_states >> 1;
    if (id >= half_dim) return;

    uint stride = 1u << qubit;
    uint block  = stride << 1;
    uint i = (id / stride) * block + (id % stride);
    uint j = i + stride;

    float a_re = state[i].real;
    float a_im = state[i].imag;
    float b_re = state[j].real;
    float b_im = state[j].imag;

    // new_a = m00 * a + m01 * b
    state[i].real = (mat.m00_re * a_re - mat.m00_im * a_im) + (mat.m01_re * b_re - mat.m01_im * b_im);
    state[i].imag = (mat.m00_re * a_im + mat.m00_im * a_re) + (mat.m01_re * b_im + mat.m01_im * b_re);
    // new_b = m10 * a + m11 * b
    state[j].real = (mat.m10_re * a_re - mat.m10_im * a_im) + (mat.m11_re * b_re - mat.m11_im * b_im);
    state[j].imag = (mat.m10_re * a_im + mat.m10_im * a_re) + (mat.m11_re * b_im + mat.m11_im * b_re);
}

// ============================================================
// DIAGONAL SINGLE-QUBIT GATES (grid = dim)
// ============================================================
// Only modify amplitudes where qubit bit is 1.

kernel void gpu_pauli_z(
    device Complex* state [[buffer(0)]],
    constant uint& qubit [[buffer(1)]],
    constant uint& num_states [[buffer(2)]],
    uint id [[thread_position_in_grid]]
) {
    if (id >= num_states) return;
    if (id & (1u << qubit)) {
        state[id].real = -state[id].real;
        state[id].imag = -state[id].imag;
    }
}

kernel void gpu_phase_s(
    device Complex* state [[buffer(0)]],
    constant uint& qubit [[buffer(1)]],
    constant uint& num_states [[buffer(2)]],
    uint id [[thread_position_in_grid]]
) {
    if (id >= num_states) return;
    if (id & (1u << qubit)) {
        float r = state[id].real;
        state[id].real = -state[id].imag;
        state[id].imag = r;
    }
}

kernel void gpu_t_gate(
    device Complex* state [[buffer(0)]],
    constant uint& qubit [[buffer(1)]],
    constant uint& num_states [[buffer(2)]],
    uint id [[thread_position_in_grid]]
) {
    if (id >= num_states) return;
    if (id & (1u << qubit)) {
        constexpr float cos_pi4 = 0.7071067811865476f;
        constexpr float sin_pi4 = 0.7071067811865476f;
        float r = state[id].real;
        float im = state[id].imag;
        state[id].real = r * cos_pi4 - im * sin_pi4;
        state[id].imag = r * sin_pi4 + im * cos_pi4;
    }
}

kernel void gpu_rotation_z(
    device Complex* state [[buffer(0)]],
    constant uint& qubit [[buffer(1)]],
    constant float& phi [[buffer(2)]],
    constant uint& num_states [[buffer(3)]],
    uint id [[thread_position_in_grid]]
) {
    if (id >= num_states) return;
    if (id & (1u << qubit)) {
        float c = cos(phi * 0.5f);
        float s = sin(phi * 0.5f);
        float r = state[id].real;
        float im = state[id].imag;
        state[id].real = r * c + im * s;
        state[id].imag = im * c - r * s;
    }
}

// ============================================================
// TWO-QUBIT GATES (grid = dim)
// ============================================================

kernel void gpu_cnot(
    device Complex* state [[buffer(0)]],
    constant uint& control [[buffer(1)]],
    constant uint& target [[buffer(2)]],
    constant uint& num_states [[buffer(3)]],
    uint id [[thread_position_in_grid]]
) {
    if (id >= num_states) return;

    uint ctrl_mask = 1u << control;
    uint tgt_mask  = 1u << target;

    // Process only |control=1, target=0> states
    if ((id & ctrl_mask) && !(id & tgt_mask)) {
        uint j = id | tgt_mask;
        Complex temp = state[id];
        state[id] = state[j];
        state[j] = temp;
    }
}

kernel void gpu_cz(
    device Complex* state [[buffer(0)]],
    constant uint& control [[buffer(1)]],
    constant uint& target [[buffer(2)]],
    constant uint& num_states [[buffer(3)]],
    uint id [[thread_position_in_grid]]
) {
    if (id >= num_states) return;
    if ((id & (1u << control)) && (id & (1u << target))) {
        state[id].real = -state[id].real;
        state[id].imag = -state[id].imag;
    }
}

kernel void gpu_cphase(
    device Complex* state [[buffer(0)]],
    constant uint& control [[buffer(1)]],
    constant uint& target [[buffer(2)]],
    constant float& phi [[buffer(3)]],
    constant uint& num_states [[buffer(4)]],
    uint id [[thread_position_in_grid]]
) {
    if (id >= num_states) return;
    if ((id & (1u << control)) && (id & (1u << target))) {
        float c = cos(phi);
        float s = sin(phi);
        float r = state[id].real;
        float im = state[id].imag;
        state[id].real = r * c - im * s;
        state[id].imag = r * s + im * c;
    }
}

kernel void gpu_swap(
    device Complex* state [[buffer(0)]],
    constant uint& q1 [[buffer(1)]],
    constant uint& q2 [[buffer(2)]],
    constant uint& num_states [[buffer(3)]],
    uint id [[thread_position_in_grid]]
) {
    if (id >= num_states) return;

    uint mask1 = 1u << q1;
    uint mask2 = 1u << q2;

    // Process only states where bits differ and id < swapped_id
    bool bit1 = (id & mask1) != 0;
    bool bit2 = (id & mask2) != 0;
    if (bit1 != bit2) {
        uint swapped = id ^ (mask1 | mask2);
        if (id < swapped) {
            Complex temp = state[id];
            state[id] = state[swapped];
            state[swapped] = temp;
        }
    }
}

// ============================================================
// THREE-QUBIT GATES (grid = dim)
// ============================================================

kernel void gpu_toffoli(
    device Complex* state [[buffer(0)]],
    constant uint& control1 [[buffer(1)]],
    constant uint& control2 [[buffer(2)]],
    constant uint& target [[buffer(3)]],
    constant uint& num_states [[buffer(4)]],
    uint id [[thread_position_in_grid]]
) {
    if (id >= num_states) return;

    uint ctrl1_mask = 1u << control1;
    uint ctrl2_mask = 1u << control2;
    uint tgt_mask   = 1u << target;

    // Process only |c1=1,c2=1,t=0> states
    if ((id & ctrl1_mask) && (id & ctrl2_mask) && !(id & tgt_mask)) {
        uint j = id | tgt_mask;
        Complex temp = state[id];
        state[id] = state[j];
        state[j] = temp;
    }
}

// ============================================================
// GENERIC 4x4 UNITARY (grid = dim/4) — fused two-qubit gates
// ============================================================
// Maps thread id in [0, dim/4) to a unique (i00, i01, i10, i11)
// quadruplet indexed by two qubit bits.
// qubit_lo < qubit_hi are the two qubit indices.

struct Matrix4x4 {
    float data[32]; // 4x4 complex matrix: data[2*(4*r+c)] = re, data[2*(4*r+c)+1] = im
};

// Insert a zero-bit at position `pos` in index `val`.
// All bits at position >= pos are shifted left by 1.
inline uint insert_zero_bit(uint val, uint pos) {
    uint mask_lo = (1u << pos) - 1u;
    uint lo = val & mask_lo;
    uint hi = val & ~mask_lo;
    return lo | (hi << 1u);
}

kernel void gpu_unitary4x4(
    device Complex* state [[buffer(0)]],
    constant uint& qubit_lo [[buffer(1)]],
    constant uint& qubit_hi [[buffer(2)]],
    constant Matrix4x4& mat [[buffer(3)]],
    constant uint& num_states [[buffer(4)]],
    uint id [[thread_position_in_grid]]
) {
    uint quarter_dim = num_states >> 2;
    if (id >= quarter_dim) return;

    // Build base index by inserting two zero bits at qubit_lo and qubit_hi positions
    uint base = insert_zero_bit(id, qubit_lo);
    base = insert_zero_bit(base, qubit_hi);

    uint stride_lo = 1u << qubit_lo;
    uint stride_hi = 1u << qubit_hi;

    // Four indices: |00⟩, |01⟩, |10⟩, |11⟩ in (qubit_hi, qubit_lo) basis
    uint idx[4];
    idx[0] = base;                          // |00⟩
    idx[1] = base | stride_lo;              // |01⟩
    idx[2] = base | stride_hi;              // |10⟩
    idx[3] = base | stride_lo | stride_hi;  // |11⟩

    // Load amplitudes
    float amp_re[4], amp_im[4];
    for (int k = 0; k < 4; k++) {
        amp_re[k] = state[idx[k]].real;
        amp_im[k] = state[idx[k]].imag;
    }

    // Matrix-vector multiply: new_amp = mat * amp
    for (int r = 0; r < 4; r++) {
        float new_re = 0.0f;
        float new_im = 0.0f;
        for (int c = 0; c < 4; c++) {
            float m_re = mat.data[2 * (4 * r + c)];
            float m_im = mat.data[2 * (4 * r + c) + 1];
            // (m_re + i*m_im) * (amp_re + i*amp_im)
            new_re += m_re * amp_re[c] - m_im * amp_im[c];
            new_im += m_re * amp_im[c] + m_im * amp_re[c];
        }
        state[idx[r]].real = new_re;
        state[idx[r]].imag = new_im;
    }
}

// ============================================================
// MEASUREMENT / UTILITIES (grid = dim)
// ============================================================

kernel void gpu_probabilities(
    device Complex* state [[buffer(0)]],
    device float* probs [[buffer(1)]],
    constant uint& num_states [[buffer(2)]],
    uint id [[thread_position_in_grid]]
) {
    if (id >= num_states) return;
    float r = state[id].real;
    float im = state[id].imag;
    probs[id] = r * r + im * im;
}

kernel void gpu_init_zero(
    device Complex* state [[buffer(0)]],
    constant uint& num_states [[buffer(1)]],
    uint id [[thread_position_in_grid]]
) {
    if (id >= num_states) return;
    state[id] = {id == 0 ? 1.0f : 0.0f, 0.0f};
}
