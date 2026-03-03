// Metal GPU Stabilizer Simulation Kernels
// ========================================
// GPU-accelerated stabilizer simulation for Apple Silicon
// Target: Match or exceed Stim throughput on M4+ GPUs

#include <metal_stdlib>
using namespace metal;

// ============================================================
// STABILIZER TABLEAU DATA STRUCTURES
// ============================================================

// Packed Pauli row: X bits and Z bits for n qubits
// Each qubit encoded as: (x,z) = I(0,0), X(1,0), Y(1,1), Z(0,1)
struct PackedPauliRow {
    device uint64_t* x_bits;
    device uint64_t* z_bits;
    uint32_t num_words;  // ceil(num_qubits / 64)
};

// Full stabilizer tableau: 2n rows × n qubits
struct StabilizerTableau {
    device uint64_t* xs;  // 2n rows × num_words
    device uint64_t* zs;  // 2n rows × num_words
    device uint8_t* phases;  // 2n phase bits
    uint32_t num_qubits;
    uint32_t num_words;
};

// ============================================================
// SINGLE-QUBIT GATES (H, S, X, Y, Z)
// ============================================================
//
// All per-gate kernels use the INDIVIDUAL BUFFER layout matching
// the Rust apply_gate_inner() dispatch:
//   buffer(0) = xs     (device uint64_t*)
//   buffer(1) = zs     (device uint64_t*)
//   buffer(2) = phases (device uint8_t*)
//   buffer(3) = num_qubits (constant uint&)
//   buffer(4) = num_words  (constant uint&)
//   buffer(5) = qubit1     (constant uint&)
//   buffer(6) = qubit2     (constant uint&)  -- two-qubit gates only
//
// Each thread processes ONE row of the 2n-row tableau.
// Dispatch with (2 * num_qubits) threads.

/// Hadamard gate on qubit q: swaps X and Z bits, phase ^= (x AND z)
kernel void stabilizer_h(
    device uint64_t* xs [[buffer(0)]],
    device uint64_t* zs [[buffer(1)]],
    device uint8_t* phases [[buffer(2)]],
    constant uint& num_qubits [[buffer(3)]],
    constant uint& num_words [[buffer(4)]],
    constant uint& qubit [[buffer(5)]],
    uint gid [[thread_position_in_grid]]
) {
    uint row = gid;
    if (row >= 2 * num_qubits) return;

    uint qword = qubit / 64;
    uint64_t qmask = 1ULL << (qubit % 64);
    uint idx = row * num_words + qword;

    uint64_t x = xs[idx];
    uint64_t z = zs[idx];
    uint64_t xb = x & qmask;
    uint64_t zb = z & qmask;

    // H: swap X <-> Z
    xs[idx] = (x & ~qmask) | (zb ? qmask : 0);
    zs[idx] = (z & ~qmask) | (xb ? qmask : 0);

    // Phase: xor when both X and Z were set (Y -> -Y under H)
    if (xb && zb) phases[row] ^= 1;
}

/// Phase gate (S) on qubit q: X -> Y (Z ^= X), phase ^= (x AND z)
kernel void stabilizer_s(
    device uint64_t* xs [[buffer(0)]],
    device uint64_t* zs [[buffer(1)]],
    device uint8_t* phases [[buffer(2)]],
    constant uint& num_qubits [[buffer(3)]],
    constant uint& num_words [[buffer(4)]],
    constant uint& qubit [[buffer(5)]],
    uint gid [[thread_position_in_grid]]
) {
    uint row = gid;
    if (row >= 2 * num_qubits) return;

    uint qword = qubit / 64;
    uint64_t qmask = 1ULL << (qubit % 64);
    uint idx = row * num_words + qword;

    uint64_t xb = xs[idx] & qmask;
    uint64_t zb = zs[idx] & qmask;

    // S: Z ^= X
    if (xb) zs[idx] ^= qmask;

    // Phase update: when X=1 and Z=1 before the update
    if (xb && zb) phases[row] ^= 1;
}

/// Pauli X gate: conjugation flips phase when Z bit is set
kernel void stabilizer_x(
    device uint64_t* xs [[buffer(0)]],
    device uint64_t* zs [[buffer(1)]],
    device uint8_t* phases [[buffer(2)]],
    constant uint& num_qubits [[buffer(3)]],
    constant uint& num_words [[buffer(4)]],
    constant uint& qubit [[buffer(5)]],
    uint gid [[thread_position_in_grid]]
) {
    uint row = gid;
    if (row >= 2 * num_qubits) return;

    uint qword = qubit / 64;
    uint64_t qmask = 1ULL << (qubit % 64);
    uint idx = row * num_words + qword;

    // X conjugation: X*P*X^dag
    // X commutes with I,X; anticommutes with Y,Z
    // Phase flips when Z bit is set (Z -> -Z, Y -> -Y under X conjugation)
    if (zs[idx] & qmask) phases[row] ^= 1;
}

/// Pauli Z gate: conjugation flips phase when X bit is set
kernel void stabilizer_z(
    device uint64_t* xs [[buffer(0)]],
    device uint64_t* zs [[buffer(1)]],
    device uint8_t* phases [[buffer(2)]],
    constant uint& num_qubits [[buffer(3)]],
    constant uint& num_words [[buffer(4)]],
    constant uint& qubit [[buffer(5)]],
    uint gid [[thread_position_in_grid]]
) {
    uint row = gid;
    if (row >= 2 * num_qubits) return;

    uint qword = qubit / 64;
    uint64_t qmask = 1ULL << (qubit % 64);
    uint idx = row * num_words + qword;

    // Z conjugation: Z*P*Z^dag
    // Z commutes with I,Z; anticommutes with X,Y
    // Phase flips when X bit is set
    if (xs[idx] & qmask) phases[row] ^= 1;
}

/// Pauli Y gate: conjugation flips phase when exactly one of X,Z is set
kernel void stabilizer_y(
    device uint64_t* xs [[buffer(0)]],
    device uint64_t* zs [[buffer(1)]],
    device uint8_t* phases [[buffer(2)]],
    constant uint& num_qubits [[buffer(3)]],
    constant uint& num_words [[buffer(4)]],
    constant uint& qubit [[buffer(5)]],
    uint gid [[thread_position_in_grid]]
) {
    uint row = gid;
    if (row >= 2 * num_qubits) return;

    uint qword = qubit / 64;
    uint64_t qmask = 1ULL << (qubit % 64);
    uint idx = row * num_words + qword;

    uint64_t xb = xs[idx] & qmask;
    uint64_t zb = zs[idx] & qmask;

    // Y conjugation: Y*P*Y^dag
    // Y commutes with I,Y; anticommutes with X,Z
    // Phase flips when X xor Z (i.e., exactly one is set but not both)
    if (xb != zb) phases[row] ^= 1;
}

/// SWAP gate: exchange two qubits in every row
kernel void stabilizer_swap(
    device uint64_t* xs [[buffer(0)]],
    device uint64_t* zs [[buffer(1)]],
    device uint8_t* phases [[buffer(2)]],
    constant uint& num_qubits [[buffer(3)]],
    constant uint& num_words [[buffer(4)]],
    constant uint& qubit_a [[buffer(5)]],
    constant uint& qubit_b [[buffer(6)]],
    uint gid [[thread_position_in_grid]]
) {
    uint row = gid;
    if (row >= 2 * num_qubits) return;

    uint aword = qubit_a / 64;
    uint bword = qubit_b / 64;
    uint64_t amask = 1ULL << (qubit_a % 64);
    uint64_t bmask = 1ULL << (qubit_b % 64);

    uint row_base = row * num_words;

    // Read X bits for both qubits
    uint64_t xa = xs[row_base + aword] & amask;
    uint64_t xb = xs[row_base + bword] & bmask;

    // Read Z bits for both qubits
    uint64_t za = zs[row_base + aword] & amask;
    uint64_t zb = zs[row_base + bword] & bmask;

    // Clear both positions in X
    xs[row_base + aword] &= ~amask;
    xs[row_base + bword] &= ~bmask;
    // Set swapped X bits
    if (xb) xs[row_base + aword] |= amask;
    if (xa) xs[row_base + bword] |= bmask;

    // Clear both positions in Z
    zs[row_base + aword] &= ~amask;
    zs[row_base + bword] &= ~bmask;
    // Set swapped Z bits
    if (zb) zs[row_base + aword] |= amask;
    if (za) zs[row_base + bword] |= bmask;

    // SWAP is phase-free (no phase update needed)
}

// ============================================================
// TWO-QUBIT GATES (CX, CZ)
// ============================================================

/// CNOT gate: control=qubit1, target=qubit2
/// X_c -> X_c X_t,  Z_t -> Z_c Z_t
kernel void stabilizer_cx(
    device uint64_t* xs [[buffer(0)]],
    device uint64_t* zs [[buffer(1)]],
    device uint8_t* phases [[buffer(2)]],
    constant uint& num_qubits [[buffer(3)]],
    constant uint& num_words [[buffer(4)]],
    constant uint& control [[buffer(5)]],
    constant uint& target [[buffer(6)]],
    uint gid [[thread_position_in_grid]]
) {
    uint row = gid;
    if (row >= 2 * num_qubits) return;

    uint cword = control / 64;
    uint tword = target / 64;
    uint64_t cmask = 1ULL << (control % 64);
    uint64_t tmask = 1ULL << (target % 64);
    uint row_base = row * num_words;

    uint64_t xc = xs[row_base + cword] & cmask;
    uint64_t zt = zs[row_base + tword] & tmask;

    // Read xt and zc BEFORE the update for phase calculation
    uint64_t xt = xs[row_base + tword] & tmask;
    uint64_t zc = zs[row_base + cword] & cmask;

    // CX update: X_c propagates to target, Z_t propagates to control
    if (xc) xs[row_base + tword] ^= tmask;
    if (zt) zs[row_base + cword] ^= cmask;

    // Phase: ^= xc & zt & (xt ^ zc ^ 1)
    // This is the Aaronson-Gottesman phase rule for CNOT:
    // sign flips when xc=1 AND zt=1 AND (xt XOR zc) = 0
    // i.e., xc AND zt AND NOT(xt XOR zc)
    if (xc && zt && !(xt ^ zc)) {
        phases[row] ^= 1;
    }
}

/// CZ gate: symmetric controlled-phase
/// X_a -> X_a Z_b,  X_b -> Z_a X_b
kernel void stabilizer_cz(
    device uint64_t* xs [[buffer(0)]],
    device uint64_t* zs [[buffer(1)]],
    device uint8_t* phases [[buffer(2)]],
    constant uint& num_qubits [[buffer(3)]],
    constant uint& num_words [[buffer(4)]],
    constant uint& qubit_a [[buffer(5)]],
    constant uint& qubit_b [[buffer(6)]],
    uint gid [[thread_position_in_grid]]
) {
    uint row = gid;
    if (row >= 2 * num_qubits) return;

    uint aword = qubit_a / 64;
    uint bword = qubit_b / 64;
    uint64_t amask = 1ULL << (qubit_a % 64);
    uint64_t bmask = 1ULL << (qubit_b % 64);
    uint row_base = row * num_words;

    uint64_t xa = xs[row_base + aword] & amask;
    uint64_t xb = xs[row_base + bword] & bmask;

    // CZ: X_a picks up Z_b and vice versa
    if (xa) zs[row_base + bword] ^= bmask;
    if (xb) zs[row_base + aword] ^= amask;
}

// ============================================================
// BATCH GATE APPLICATION
// ============================================================

/// Apply multiple gates sequentially to the full tableau.
///
/// This is a SINGLE-THREADED kernel (dispatch with 1 thread, gid=0 only)
/// that iterates over a list of gates with separate type/qubit arrays.
/// For high-throughput use, prefer stabilizer_apply_batch_v2 which uses
/// packed gate encoding and parallelises over rows.
///
/// Gate types: 0=H, 1=S, 2=X, 3=Z, 4=CX, 5=CZ
kernel void stabilizer_gate_batch(
    device StabilizerTableau& tab [[buffer(0)]],
    device const uint* gate_types [[buffer(1)]],
    device const uint* gate_qubits [[buffer(2)]],  // qubit pairs: [q1_0, q2_0, q1_1, q2_1, ...]
    constant uint& num_gates [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid != 0) return;  // single-threaded sequential execution

    uint n  = tab.num_qubits;
    uint nw = tab.num_words;
    uint num_rows = 2 * n;

    for (uint g = 0; g < num_gates; g++) {
        uint gate_type = gate_types[g];
        uint q1 = gate_qubits[g * 2];
        uint q2 = gate_qubits[g * 2 + 1];

        if (q1 >= n) continue;
        if (gate_type >= 4 && q2 >= n) continue;  // 2-qubit gates: 4=CX, 5=CZ

        uint q1_word = q1 / 64;
        uint64_t q1_mask = 1ULL << (q1 % 64);

        switch (gate_type) {
            case 0: { // Hadamard on q1
                for (uint row = 0; row < num_rows; row++) {
                    uint idx = row * nw + q1_word;
                    uint64_t x = tab.xs[idx];
                    uint64_t z = tab.zs[idx];
                    uint64_t xb = x & q1_mask;
                    uint64_t zb = z & q1_mask;

                    tab.xs[idx] = (x & ~q1_mask) | (zb ? q1_mask : 0);
                    tab.zs[idx] = (z & ~q1_mask) | (xb ? q1_mask : 0);

                    if (xb && zb) tab.phases[row] ^= 1;
                }
                break;
            }
            case 1: { // S (Phase gate) on q1
                for (uint row = 0; row < num_rows; row++) {
                    uint idx = row * nw + q1_word;
                    uint64_t x = tab.xs[idx];
                    uint64_t xb = x & q1_mask;
                    uint64_t zb = tab.zs[idx] & q1_mask;

                    if (xb) tab.zs[idx] ^= q1_mask;
                    if (xb && zb) tab.phases[row] ^= 1;
                }
                break;
            }
            case 2: { // Pauli X on q1
                for (uint row = 0; row < num_rows; row++) {
                    uint idx = row * nw + q1_word;
                    uint64_t zb = tab.zs[idx] & q1_mask;
                    // X commutes with X, anticommutes with Z and Y
                    // Conjugation: X*Z = -Z (flip phase when Z bit set)
                    if (zb) tab.phases[row] ^= 1;
                }
                break;
            }
            case 3: { // Pauli Z on q1
                for (uint row = 0; row < num_rows; row++) {
                    uint idx = row * nw + q1_word;
                    uint64_t xb = tab.xs[idx] & q1_mask;
                    // Z commutes with Z, anticommutes with X and Y
                    // Conjugation: Z*X = -X (flip phase when X bit set)
                    if (xb) tab.phases[row] ^= 1;
                }
                break;
            }
            case 4: { // CX (CNOT) - control=q1, target=q2
                uint q2_word = q2 / 64;
                uint64_t q2_mask = 1ULL << (q2 % 64);

                for (uint row = 0; row < num_rows; row++) {
                    uint64_t xc = tab.xs[row * nw + q1_word] & q1_mask;
                    uint64_t zt = tab.zs[row * nw + q2_word] & q2_mask;

                    if (xc) tab.xs[row * nw + q2_word] ^= q2_mask;
                    if (zt) tab.zs[row * nw + q1_word] ^= q1_mask;

                    // Phase: xc & zt & (xt ^ zc) after the X/Z update
                    uint64_t xt = tab.xs[row * nw + q2_word] & q2_mask;
                    uint64_t zc = tab.zs[row * nw + q1_word] & q1_mask;
                    if (xc && zt && (xt ^ zc)) {
                        tab.phases[row] ^= 1;
                    }
                }
                break;
            }
            case 5: { // CZ - symmetric on q1, q2
                uint q2_word = q2 / 64;
                uint64_t q2_mask = 1ULL << (q2 % 64);

                for (uint row = 0; row < num_rows; row++) {
                    uint64_t xa = tab.xs[row * nw + q1_word] & q1_mask;
                    uint64_t xb = tab.xs[row * nw + q2_word] & q2_mask;

                    if (xa) tab.zs[row * nw + q2_word] ^= q2_mask;
                    if (xb) tab.zs[row * nw + q1_word] ^= q1_mask;
                }
                break;
            }
        }
    }
}

// ============================================================
// MEASUREMENT SIMULATION
// ============================================================

/// Measure qubit in the Z basis using the Gottesman-Knill algorithm.
///
/// This is a SINGLE-THREADED kernel (dispatch with 1 thread) that performs
/// the full measurement update on the tableau:
///
///   1. Scan stabilizer rows (n..2n-1) for one whose X-component
///      anticommutes with Z on the measured qubit (x_bit[q] set).
///   2. If found (row p): outcome is random.
///      - Row-multiply every other anticommuting stabilizer row by row p.
///      - Row-multiply every anticommuting destabilizer row by row p.
///      - Copy row p into the corresponding destabilizer slot (row p-n).
///      - Replace row p with +/-Z_q (Z bit set, X bits cleared, phase = random).
///   3. If not found: outcome is deterministic.
///      - Accumulate (XOR) the phases of stabilizer rows (d+n) for every
///        destabilizer row d (0..n-1) whose X-component has bit q set.
///
/// result[0]: 0 = measured |0>, 1 = measured |1>
/// result[1]: 0 = random outcome, 1 = deterministic outcome
///
/// random_seed is used to produce a pseudo-random bit when the outcome is random.
kernel void stabilizer_measurement_check(
    device StabilizerTableau& tab [[buffer(0)]],
    constant uint& qubit [[buffer(1)]],
    device uint* result [[buffer(2)]],
    constant uint& random_seed [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    // Only thread 0 runs the full measurement algorithm
    if (gid != 0) return;

    uint n  = tab.num_qubits;
    uint nw = tab.num_words;
    uint qword  = qubit / 64;
    uint64_t qmask = 1ULL << (qubit % 64);

    // ------------------------------------------------------------------
    // Step 1: Find the first anticommuting stabilizer row
    // ------------------------------------------------------------------
    int p = -1;  // index of anticommuting stabilizer row (n..2n-1)
    for (uint s = n; s < 2 * n; s++) {
        if (tab.xs[s * nw + qword] & qmask) {
            p = (int)s;
            break;
        }
    }

    if (p >= 0) {
        // ==============================================================
        // RANDOM OUTCOME
        // ==============================================================
        // Helper lambda-like inline: row-multiply row_dst by row_src.
        // Pauli product (X1,Z1)*(X2,Z2):
        //   X_out = X1 ^ X2,  Z_out = Z1 ^ Z2
        //   phase ^= phase1 ^ phase2 ^ 2*(X1*Z2 inner product mod 2)
        // We only need the phase sign bit (mod 2), so we track the
        // "extra sign" from X1.Z2 via popcount parity.

        // (a) Row-multiply every OTHER anticommuting stabilizer row i by row p
        for (uint i = n; i < 2 * n; i++) {
            if ((int)i == p) continue;
            if (!(tab.xs[i * nw + qword] & qmask)) continue;

            // Compute sign contribution: popcount(xi & zp) parity
            uint sign = 0;
            for (uint w = 0; w < nw; w++) {
                uint64_t overlap = tab.xs[i * nw + w] & tab.zs[p * nw + w];
                sign ^= (uint)popcount(overlap);
            }
            // XOR X and Z words
            for (uint w = 0; w < nw; w++) {
                tab.xs[i * nw + w] ^= tab.xs[p * nw + w];
                tab.zs[i * nw + w] ^= tab.zs[p * nw + w];
            }
            // Phase: phase_i ^= phase_p ^ (sign & 1)
            tab.phases[i] ^= tab.phases[p] ^ (uint8_t)(sign & 1u);
        }

        // (b) Row-multiply every anticommuting DESTABILIZER row d by row p
        for (uint d = 0; d < n; d++) {
            if (!(tab.xs[d * nw + qword] & qmask)) continue;

            uint sign = 0;
            for (uint w = 0; w < nw; w++) {
                uint64_t overlap = tab.xs[d * nw + w] & tab.zs[p * nw + w];
                sign ^= (uint)popcount(overlap);
            }
            for (uint w = 0; w < nw; w++) {
                tab.xs[d * nw + w] ^= tab.xs[p * nw + w];
                tab.zs[d * nw + w] ^= tab.zs[p * nw + w];
            }
            tab.phases[d] ^= tab.phases[p] ^ (uint8_t)(sign & 1u);
        }

        // (c) Copy stabilizer row p to destabilizer slot (p - n)
        uint dest = (uint)p - n;
        for (uint w = 0; w < nw; w++) {
            tab.xs[dest * nw + w] = tab.xs[p * nw + w];
            tab.zs[dest * nw + w] = tab.zs[p * nw + w];
        }
        tab.phases[dest] = tab.phases[p];

        // (d) Replace row p with +/-Z_q
        for (uint w = 0; w < nw; w++) {
            tab.xs[p * nw + w] = 0;
            tab.zs[p * nw + w] = 0;
        }
        tab.zs[p * nw + qword] = qmask;

        // Random bit from seed (simple hash)
        uint random_bit = ((random_seed * 2654435761u) >> 16) & 1u;
        tab.phases[p] = (uint8_t)random_bit;

        result[0] = random_bit;      // measurement outcome
        result[1] = 0;               // flag: random
    } else {
        // ==============================================================
        // DETERMINISTIC OUTCOME
        // ==============================================================
        // XOR the phases of stabilizer rows (d+n) for every destabilizer
        // row d (0..n-1) whose X-component has x_bit[q] set.
        uint det_result = 0;
        for (uint d = 0; d < n; d++) {
            if (tab.xs[d * nw + qword] & qmask) {
                det_result ^= (uint)tab.phases[d + n];
            }
        }

        result[0] = det_result;      // measurement outcome
        result[1] = 1;               // flag: deterministic
    }
}

// ============================================================
// BULK SAMPLING (HIGH-THROUGHPUT)
// ============================================================

/// Sample many stabilizer circuits in parallel
/// Used for QEC threshold estimation
///
/// Each GPU thread runs a complete circuit on its own tableau copy,
/// then measures qubit 0 using the stabilizer formalism:
///   - Scan stabilizer rows (n..2n-1) for one whose X-component
///     anticommutes with Z on qubit 0 (i.e. x_bit[q0] is set).
///   - If found: measurement is random; result = phase of that row.
///   - If not found: measurement is deterministic; compute from
///     the product of destabilizer phases whose x_bit[q0] is set.
///
/// Gate encoding (packed uint, same as v2 kernel):
///   gate_type = packed & 0x3       -- 0=H, 1=S, 2=CX, 3=CZ
///   q1        = (packed >> 2)  & 0xFFF
///   q2        = (packed >> 14) & 0xFFF  (for CX/CZ only)
kernel void stabilizer_bulk_sample(
    device StabilizerTableau* tables [[buffer(0)]],  // Array of tableaus
    device const uint* circuit_gates [[buffer(1)]],
    device uint* measurement_results [[buffer(2)]],
    constant uint& num_samples [[buffer(3)]],
    constant uint& circuit_length [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= num_samples) return;

    // Each thread runs a full circuit on its own tableau
    device StabilizerTableau& tab = tables[gid];
    uint n = tab.num_qubits;
    uint nw = tab.num_words;
    uint num_rows = 2 * n;

    // -------------------------------------------------------
    // Phase 1: Apply the full circuit gate-by-gate
    // -------------------------------------------------------
    for (uint g = 0; g < circuit_length; g++) {
        uint packed = circuit_gates[g];
        uint gate_type = packed & 0x3u;
        uint q1 = (packed >> 2) & 0xFFFu;
        uint q2 = (packed >> 14) & 0xFFFu;

        // Bounds check
        if (q1 >= n) continue;
        if (gate_type >= 2 && q2 >= n) continue;

        uint q1_word = q1 / 64;
        uint64_t q1_mask = 1ULL << (q1 % 64);

        switch (gate_type) {
            case 0: { // Hadamard on q1
                for (uint row = 0; row < num_rows; row++) {
                    uint idx = row * nw + q1_word;
                    uint64_t x = tab.xs[idx];
                    uint64_t z = tab.zs[idx];
                    uint64_t xb = x & q1_mask;
                    uint64_t zb = z & q1_mask;

                    // H: swap X <-> Z bits
                    tab.xs[idx] = (x & ~q1_mask) | (zb ? q1_mask : 0);
                    tab.zs[idx] = (z & ~q1_mask) | (xb ? q1_mask : 0);

                    // Phase update: phase ^= (x_bit AND z_bit)
                    if (xb && zb) {
                        tab.phases[row] ^= 1;
                    }
                }
                break;
            }
            case 1: { // S (Phase gate) on q1
                for (uint row = 0; row < num_rows; row++) {
                    uint idx = row * nw + q1_word;
                    uint64_t x = tab.xs[idx];
                    uint64_t xb = x & q1_mask;
                    uint64_t zb = tab.zs[idx] & q1_mask;

                    // S: Z ^= X (X -> Y when X-bit set)
                    if (xb) tab.zs[idx] ^= q1_mask;

                    // Phase update
                    if (xb && zb) {
                        tab.phases[row] ^= 1;
                    }
                }
                break;
            }
            case 2: { // CX (CNOT) — control=q1, target=q2
                uint q2_word = q2 / 64;
                uint64_t q2_mask = 1ULL << (q2 % 64);

                for (uint row = 0; row < num_rows; row++) {
                    uint64_t xc = tab.xs[row * nw + q1_word] & q1_mask;
                    uint64_t zt = tab.zs[row * nw + q2_word] & q2_mask;

                    // CX: X_c propagates to target, Z_t propagates to control
                    if (xc) tab.xs[row * nw + q2_word] ^= q2_mask;
                    if (zt) tab.zs[row * nw + q1_word] ^= q1_mask;

                    // Phase: ^= xc & zt & (xt ^ zc)
                    uint64_t xt = tab.xs[row * nw + q2_word] & q2_mask;
                    uint64_t zc = tab.zs[row * nw + q1_word] & q1_mask;
                    if (xc && zt && (xt ^ zc)) {
                        tab.phases[row] ^= 1;
                    }
                }
                break;
            }
            case 3: { // CZ — symmetric on q1, q2
                uint q2_word = q2 / 64;
                uint64_t q2_mask = 1ULL << (q2 % 64);

                for (uint row = 0; row < num_rows; row++) {
                    uint64_t xa = tab.xs[row * nw + q1_word] & q1_mask;
                    uint64_t xb = tab.xs[row * nw + q2_word] & q2_mask;

                    // CZ: X_a picks up Z_b and vice versa
                    if (xa) tab.zs[row * nw + q2_word] ^= q2_mask;
                    if (xb) tab.zs[row * nw + q1_word] ^= q1_mask;
                }
                break;
            }
        }
    }

    // -------------------------------------------------------
    // Phase 2: Measure qubit 0 via stabilizer formalism
    // -------------------------------------------------------
    // We measure in the Z basis on qubit 0.
    // A stabilizer row anticommutes with Z_0 iff its X-component
    // has bit 0 set.
    //
    // Scan stabilizer rows (indices n .. 2n-1) for one with x_bit[0].
    // - If found: outcome is random; use phase of that row as result.
    // - If not found: outcome is deterministic; XOR the phases of all
    //   destabilizer rows (indices 0 .. n-1) that have x_bit[0] set.
    // -------------------------------------------------------
    uint q0_word = 0;                // qubit 0 is always in word 0
    uint64_t q0_mask = 1ULL;         // bit 0

    // Search stabilizer rows for anticommuting generator
    int found_row = -1;
    for (uint s = n; s < num_rows; s++) {
        uint64_t x = tab.xs[s * nw + q0_word];
        if (x & q0_mask) {
            found_row = (int)s;
            break;
        }
    }

    if (found_row >= 0) {
        // Random outcome — report the phase of the anticommuting row.
        // In a full implementation we would also update the tableau
        // (row-reduce), but for bulk sampling the phase gives the
        // measurement bit: 0 -> |0>, 1 -> |1>.
        measurement_results[gid] = (uint)tab.phases[found_row];
    } else {
        // Deterministic outcome — XOR phases of destabilizer rows
        // whose X-component has bit 0 set.
        uint result = 0;
        for (uint d = 0; d < n; d++) {
            uint64_t x = tab.xs[d * nw + q0_word];
            if (x & q0_mask) {
                result ^= (uint)tab.phases[d];
            }
        }
        measurement_results[gid] = result;
    }
}

// ============================================================
// OPTIMIZED BATCH KERNEL (Z3 OPTIMIZATION) - V2
// ============================================================
//
// Key optimizations:
// 1. Entire circuit in ONE kernel launch (no per-gate dispatch)
// 2. Threadgroup shared memory for row caching
// 3. NO atomics - use thread-local phase tracking
// 4. Sequential gate processing per row (better cache locality)
// 5. Unrolled inner loops

/// ULTRA-OPTIMIZED: Each thread processes ALL gates for ONE row
/// This eliminates atomic operations and maximizes cache locality
/// Uses individual buffers for flexibility (not struct reference)
kernel void stabilizer_apply_batch_v2(
    device uint64_t* xs [[buffer(0)]],
    device uint64_t* zs [[buffer(1)]],
    device uint8_t* phases [[buffer(2)]],
    constant uint& num_qubits [[buffer(3)]],
    constant uint& num_words [[buffer(4)]],
    device const uint* packed_gates [[buffer(5)]],
    constant uint& num_gates [[buffer(6)]],
    uint gid [[thread_position_in_grid]]
) {
    // Each thread handles one row
    uint row = gid;
    uint num_rows = 2 * num_qubits;

    if (row >= num_rows) return;

    // Pre-compute row base index (avoid multiply in loop)
    uint row_base = row * num_words;

    // Cache phase locally (no atomics needed - each thread owns its row)
    uint8_t local_phase = phases[row];

    // Process ALL gates for this row
    for (uint g = 0; g < num_gates; g++) {
        uint packed = packed_gates[g];
        uint gate_type = packed & 0x3u;
        uint q1 = (packed >> 2) & 0xFFFu;
        uint q2 = (packed >> 14) & 0xFFFu;

        // Validate qubits (branch predictor will optimize)
        if (q1 >= num_qubits) continue;
        if (gate_type >= 2 && q2 >= num_qubits) continue;

        // Pre-compute word indices and masks
        uint q1_word = q1 / 64;
        uint q1_bit = q1 % 64;
        uint64_t q1_mask = 1ULL << q1_bit;

        switch (gate_type) {
            case 0: { // H
                uint64_t x = xs[row_base + q1_word];
                uint64_t z = zs[row_base + q1_word];
                uint64_t xb = x & q1_mask;
                uint64_t zb = z & q1_mask;

                // Swap X and Z
                xs[row_base + q1_word] = (x & ~q1_mask) | (zb ? q1_mask : 0);
                zs[row_base + q1_word] = (z & ~q1_mask) | (xb ? q1_mask : 0);

                // Phase: xor if both were set
                local_phase ^= (xb && zb) ? 1 : 0;
                break;
            }
            case 1: { // S
                uint64_t x = xs[row_base + q1_word];
                uint64_t xb = x & q1_mask;
                uint64_t zb = zs[row_base + q1_word] & q1_mask;

                // X -> Y: Z ^= X
                if (xb) zs[row_base + q1_word] ^= q1_mask;

                local_phase ^= (xb && zb) ? 1 : 0;
                break;
            }
            case 2: { // CX
                uint q2_word = q2 / 64;
                uint q2_bit = q2 % 64;
                uint64_t q2_mask = 1ULL << q2_bit;

                uint64_t xc = xs[row_base + q1_word] & q1_mask;
                uint64_t zt = zs[row_base + q2_word] & q2_mask;

                // X_c -> X_c X_t
                if (xc) xs[row_base + q2_word] ^= q2_mask;
                // Z_t -> Z_c Z_t
                if (zt) zs[row_base + q1_word] ^= q1_mask;
                break;
            }
            case 3: { // CZ
                uint q2_word = q2 / 64;
                uint q2_bit = q2 % 64;
                uint64_t q2_mask = 1ULL << q2_bit;

                uint64_t xa = xs[row_base + q1_word] & q1_mask;
                uint64_t xb = xs[row_base + q2_word] & q2_mask;

                if (xa) zs[row_base + q2_word] ^= q2_mask;
                if (xb) zs[row_base + q1_word] ^= q1_mask;
                break;
            }
        }
    }

    // Write back phase once at the end
    phases[row] = local_phase;
}

/// Apply gates using threadgroup shared memory for maximum performance
/// Loads row data into fast threadgroup memory, processes all gates, writes back
kernel void stabilizer_apply_batch_tg(
    device StabilizerTableau& tab [[buffer(0)]],
    device const uint* packed_gates [[buffer(1)]],
    constant uint& num_gates [[buffer(2)]],
    threadgroup uint64_t* tg_xs [[threadgroup(0)]],  // Shared row cache
    threadgroup uint64_t* tg_zs [[threadgroup(1)]],
    threadgroup uint8_t* tg_phase [[threadgroup(2)]],
    uint3 gid [[thread_position_in_grid]],
    uint3 tg_pos [[thread_position_in_threadgroup]],
    uint3 tg_size [[threads_per_threadgroup]]
) {
    uint row = gid.x;
    uint num_rows = 2 * tab.num_qubits;

    if (row >= num_rows) return;

    uint row_base = row * tab.num_words;
    uint tid = tg_pos.x;

    // Cooperatively load row into threadgroup memory
    for (uint w = tid; w < tab.num_words; w += tg_size.x) {
        tg_xs[w] = tab.xs[row_base + w];
        tg_zs[w] = tab.zs[row_base + w];
    }
    if (tid == 0) {
        tg_phase[0] = tab.phases[row];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Process all gates
    for (uint g = 0; g < num_gates; g++) {
        uint packed = packed_gates[g];
        uint gate_type = packed & 0x3u;
        uint q1 = (packed >> 2) & 0xFFFu;
        uint q2 = (packed >> 14) & 0xFFFu;

        if (q1 >= tab.num_qubits) continue;
        if (gate_type >= 2 && q2 >= tab.num_qubits) continue;

        uint q1_word = q1 / 64;
        uint64_t q1_mask = 1ULL << (q1 % 64);

        switch (gate_type) {
            case 0: { // H
                uint64_t x = tg_xs[q1_word];
                uint64_t z = tg_zs[q1_word];
                uint64_t xb = x & q1_mask;
                uint64_t zb = z & q1_mask;
                tg_xs[q1_word] = (x & ~q1_mask) | (zb ? q1_mask : 0);
                tg_zs[q1_word] = (z & ~q1_mask) | (xb ? q1_mask : 0);
                if (xb && zb) tg_phase[0] ^= 1;
                break;
            }
            case 1: { // S
                uint64_t x = tg_xs[q1_word];
                uint64_t xb = x & q1_mask;
                uint64_t zb = tg_zs[q1_word] & q1_mask;
                if (xb) tg_zs[q1_word] ^= q1_mask;
                if (xb && zb) tg_phase[0] ^= 1;
                break;
            }
            case 2: { // CX
                uint q2_word = q2 / 64;
                uint64_t q2_mask = 1ULL << (q2 % 64);
                uint64_t xc = tg_xs[q1_word] & q1_mask;
                uint64_t zt = tg_zs[q2_word] & q2_mask;
                if (xc) tg_xs[q2_word] ^= q2_mask;
                if (zt) tg_zs[q1_word] ^= q1_mask;
                break;
            }
            case 3: { // CZ
                uint q2_word = q2 / 64;
                uint64_t q2_mask = 1ULL << (q2 % 64);
                uint64_t xa = tg_xs[q1_word] & q1_mask;
                uint64_t xb = tg_xs[q2_word] & q2_mask;
                if (xa) tg_zs[q2_word] ^= q2_mask;
                if (xb) tg_zs[q1_word] ^= q1_mask;
                break;
            }
        }
    }

    // Write back
    for (uint w = tid; w < tab.num_words; w += tg_size.x) {
        tab.xs[row_base + w] = tg_xs[w];
        tab.zs[row_base + w] = tg_zs[w];
    }
    if (tid == 0) {
        tab.phases[row] = tg_phase[0];
    }
}

// Keep original batch kernel for compatibility.
// Uses individual buffers matching the Rust run_circuit_batch_v1() layout:
//   buffer(0) = xs,  buffer(1) = zs,  buffer(2) = phases,
//   buffer(3) = packed_gates,  buffer(4) = num_gates,
//   buffer(5) = num_qubits,  buffer(6) = num_words
/// Apply a batch of gates in a single kernel launch.
/// Each threadgroup processes one gate; threads parallelise over rows.
kernel void stabilizer_apply_batch(
    device uint64_t* xs [[buffer(0)]],
    device uint64_t* zs [[buffer(1)]],
    device uint8_t* phases [[buffer(2)]],
    device const uint* packed_gates [[buffer(3)]],
    constant uint& num_gates [[buffer(4)]],
    constant uint& num_qubits [[buffer(5)]],
    constant uint& num_words [[buffer(6)]],
    uint3 tid [[thread_position_in_grid]],
    uint3 tg_pos [[thread_position_in_threadgroup]],
    uint3 tg_size [[threads_per_threadgroup]]
) {
    // Each threadgroup handles one gate
    uint gate_idx = tid.x / tg_size.x;
    if (gate_idx >= num_gates) return;

    // Decode packed gate
    uint packed = packed_gates[gate_idx];
    uint gate_type = (packed >> 0) & 0x3u;
    uint q1 = (packed >> 2) & 0xFFFu;
    uint q2 = (packed >> 14) & 0xFFFu;

    if (q1 >= num_qubits) return;
    if ((gate_type >= 2) && q2 >= num_qubits) return;

    // Thread handles specific row
    uint row = tg_pos.x + (gate_idx * tg_size.x);
    if (row >= 2 * num_qubits) return;

    // Get word indices and masks
    uint q1_word = q1 / 64;
    uint64_t q1_mask = 1ULL << (q1 % 64);
    uint q2_word = q2 / 64;
    uint64_t q2_mask = 1ULL << (q2 % 64);

    // Row base index
    uint idx = row * num_words;

    switch (gate_type) {
        case 0: { // H
            uint64_t x = xs[idx + q1_word];
            uint64_t z = zs[idx + q1_word];
            uint64_t xb = x & q1_mask;
            uint64_t zb = z & q1_mask;

            xs[idx + q1_word] = (x & ~q1_mask) | (zb ? q1_mask : 0);
            zs[idx + q1_word] = (z & ~q1_mask) | (xb ? q1_mask : 0);

            if (xb && zb) phases[row] ^= 1;
            break;
        }
        case 1: { // S
            uint64_t x = xs[idx + q1_word];
            uint64_t z = zs[idx + q1_word];
            uint64_t xb = x & q1_mask;
            uint64_t zb = z & q1_mask;

            if (xb) zs[idx + q1_word] ^= q1_mask;
            if (xb && zb) phases[row] ^= 1;
            break;
        }
        case 2: { // CX
            uint64_t xc = xs[idx + q1_word] & q1_mask;
            uint64_t zt = zs[idx + q2_word] & q2_mask;

            if (xc) xs[idx + q2_word] ^= q2_mask;
            if (zt) zs[idx + q1_word] ^= q1_mask;
            break;
        }
        case 3: { // CZ
            uint64_t xa = xs[idx + q1_word] & q1_mask;
            uint64_t xb = xs[idx + q2_word] & q2_mask;

            if (xa) zs[idx + q2_word] ^= q2_mask;
            if (xb) zs[idx + q1_word] ^= q1_mask;
            break;
        }
    }
}

/// Bulk sampling kernel - run same circuit on many tableaus in parallel
kernel void stabilizer_bulk_sample_batch(
    device uint64_t* all_xs [[buffer(0)]],        // [num_samples][2n*num_words]
    device uint64_t* all_zs [[buffer(1)]],
    device uint8_t* all_phases [[buffer(2)]],
    device const uint* packed_gates [[buffer(3)]],
    constant uint& num_gates [[buffer(4)]],
    constant uint& num_qubits [[buffer(5)]],
    constant uint& num_words [[buffer(6)]],
    constant uint& num_samples [[buffer(7)]],
    device uint* results [[buffer(8)]],
    uint2 tid [[thread_position_in_grid]]
) {
    uint sample_idx = tid.x;
    uint row = tid.y;

    if (sample_idx >= num_samples || row >= 2 * num_qubits) return;

    // Offset to this sample's tableau
    uint row_size = 2 * num_qubits * num_words;
    device uint64_t* xs = all_xs + sample_idx * row_size + row * num_words;
    device uint64_t* zs = all_zs + sample_idx * row_size + row * num_words;
    device uint8_t* phase = all_phases + sample_idx * 2 * num_qubits + row;

    // Apply all gates
    for (uint g = 0; g < num_gates; g++) {
        uint packed = packed_gates[g];
        uint gate_type = (packed >> 0) & 0x3u;
        uint q1 = (packed >> 2) & 0xFFFu;
        uint q2 = (packed >> 14) & 0xFFFu;

        uint q1_word = q1 / 64;
        uint64_t q1_mask = 1ULL << (q1 % 64);
        uint q2_word = q2 / 64;
        uint64_t q2_mask = 1ULL << (q2 % 64);

        switch (gate_type) {
            case 0: { // H
                uint64_t x = xs[q1_word];
                uint64_t z = zs[q1_word];
                uint64_t xb = x & q1_mask;
                uint64_t zb = z & q1_mask;
                xs[q1_word] = (x & ~q1_mask) | (zb ? q1_mask : 0);
                zs[q1_word] = (z & ~q1_mask) | (xb ? q1_mask : 0);
                if (xb && zb) *phase ^= 1;
                break;
            }
            case 1: { // S
                uint64_t x = xs[q1_word];
                uint64_t xb = x & q1_mask;
                uint64_t zb = zs[q1_word] & q1_mask;
                if (xb) zs[q1_word] ^= q1_mask;
                if (xb && zb) *phase ^= 1;
                break;
            }
            case 2: { // CX
                uint64_t xc = xs[q1_word] & q1_mask;
                uint64_t zt = zs[q2_word] & q2_mask;
                if (xc) xs[q2_word] ^= q2_mask;
                if (zt) zs[q1_word] ^= q1_mask;
                break;
            }
            case 3: { // CZ
                uint64_t xa = xs[q1_word] & q1_mask;
                uint64_t xb = xs[q2_word] & q2_mask;
                if (xa) zs[q2_word] ^= q2_mask;
                if (xb) zs[q1_word] ^= q1_mask;
                break;
            }
        }
    }
}
