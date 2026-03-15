// nQPU-Metal Pulse-Level Simulation Compute Shaders
// ===================================================
// GPU-accelerated kernels for Hamiltonian-level quantum simulation:
//
//   1. complex_matmul         — NxN complex matrix multiplication
//   2. lindblad_rhs           — Lindblad master equation RHS
//   3. batch_rk4_lindblad     — Batched RK4 integration (for GRAPE)
//   4. hermitian_expm         — Matrix exponential via eigendecomposition
//   5. batch_gate_fidelity    — Parallel gate fidelity computation
//
// All kernels use f32 complex arithmetic (Metal has no f64 compute).
// The Rust host converts f64 <-> f32 at the boundary.
//
// Thread dispatch guide:
//   complex_matmul:      grid = (N, N)  — one thread per output element
//   lindblad_rhs:        grid = (N, N)  — one thread per drho element
//   batch_rk4_lindblad:  grid = M       — one thread per batch slice
//   hermitian_expm:      grid = (N, N)  — one thread per output element
//   batch_gate_fidelity: grid = M       — one thread per fidelity value

#include <metal_stdlib>
using namespace metal;

// ============================================================
// COMPLEX NUMBER HELPERS
// ============================================================

struct Complex {
    float re;
    float im;
};

// Complex addition
inline Complex c_add(Complex a, Complex b) {
    return { a.re + b.re, a.im + b.im };
}

// Complex subtraction
inline Complex c_sub(Complex a, Complex b) {
    return { a.re - b.re, a.im - b.im };
}

// Complex multiplication
inline Complex c_mul(Complex a, Complex b) {
    return { a.re * b.re - a.im * b.im, a.re * b.im + a.im * b.re };
}

// Complex conjugate
inline Complex c_conj(Complex a) {
    return { a.re, -a.im };
}

// Complex scale by real
inline Complex c_scale(Complex a, float s) {
    return { a.re * s, a.im * s };
}

// Complex modulus squared
inline float c_norm_sq(Complex a) {
    return a.re * a.re + a.im * a.im;
}

// Zero constant
constant Complex C_ZERO = { 0.0f, 0.0f };

// ============================================================
// KERNEL 1: COMPLEX MATRIX MULTIPLY — C = A * B
// ============================================================
// Each thread computes one element C[row][col] of the NxN output.
// Dispatch: grid = (N, N), threadgroup = (16, 16) or similar.

kernel void complex_matmul(
    device const Complex* A [[buffer(0)]],
    device const Complex* B [[buffer(1)]],
    device Complex* C [[buffer(2)]],
    constant uint& N [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint row = gid.y;
    uint col = gid.x;
    if (row >= N || col >= N) return;

    Complex acc = C_ZERO;
    for (uint k = 0; k < N; k++) {
        Complex a_ik = A[row * N + k];
        Complex b_kj = B[k * N + col];
        acc = c_add(acc, c_mul(a_ik, b_kj));
    }
    C[row * N + col] = acc;
}

// ============================================================
// KERNEL 1b: TILED COMPLEX MATRIX MULTIPLY (shared memory)
// ============================================================
// Uses threadgroup memory for better cache performance on larger matrices.
// Tile size = 16x16. Dispatch: grid = (ceil(N/16)*16, ceil(N/16)*16).

constant uint TILE_SIZE = 16;

kernel void complex_matmul_tiled(
    device const Complex* A [[buffer(0)]],
    device const Complex* B [[buffer(1)]],
    device Complex* C [[buffer(2)]],
    constant uint& N [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]],
    uint2 lid [[thread_position_in_threadgroup]]
) {
    threadgroup Complex tile_A[TILE_SIZE * TILE_SIZE];
    threadgroup Complex tile_B[TILE_SIZE * TILE_SIZE];

    uint row = gid.y;
    uint col = gid.x;
    uint lr = lid.y;
    uint lc = lid.x;

    Complex acc = C_ZERO;

    uint num_tiles = (N + TILE_SIZE - 1) / TILE_SIZE;
    for (uint t = 0; t < num_tiles; t++) {
        // Load tile from A
        uint a_col = t * TILE_SIZE + lc;
        if (row < N && a_col < N) {
            tile_A[lr * TILE_SIZE + lc] = A[row * N + a_col];
        } else {
            tile_A[lr * TILE_SIZE + lc] = C_ZERO;
        }

        // Load tile from B
        uint b_row = t * TILE_SIZE + lr;
        if (b_row < N && col < N) {
            tile_B[lr * TILE_SIZE + lc] = B[b_row * N + col];
        } else {
            tile_B[lr * TILE_SIZE + lc] = C_ZERO;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Accumulate tile product
        for (uint k = 0; k < TILE_SIZE; k++) {
            acc = c_add(acc, c_mul(tile_A[lr * TILE_SIZE + k], tile_B[k * TILE_SIZE + lc]));
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (row < N && col < N) {
        C[row * N + col] = acc;
    }
}

// ============================================================
// KERNEL 2: LINDBLAD RHS
// ============================================================
// Computes drho/dt = -i[H, rho] + sum_k (L_k rho L_k† - 0.5 {L_k†L_k, rho})
//
// Each thread computes one element drho[row][col].
// Dispatch: grid = (N, N).
//
// Buffer layout for L_ops: K collapse operators packed as K * N * N contiguous.

kernel void lindblad_rhs(
    device const Complex* H [[buffer(0)]],      // Hamiltonian NxN
    device const Complex* rho [[buffer(1)]],     // density matrix NxN
    device const Complex* L_ops [[buffer(2)]],   // collapse operators, K * N * N
    device Complex* drho [[buffer(3)]],          // output drho NxN
    constant uint& N [[buffer(4)]],              // Hilbert space dimension
    constant uint& K [[buffer(5)]],              // number of collapse operators
    uint2 gid [[thread_position_in_grid]]
) {
    uint row = gid.y;
    uint col = gid.x;
    if (row >= N || col >= N) return;

    uint idx = row * N + col;

    // --- Unitary part: -i[H, rho] = -i(H*rho - rho*H) ---
    // Compute (H*rho)[row][col]
    Complex h_rho = C_ZERO;
    for (uint k = 0; k < N; k++) {
        h_rho = c_add(h_rho, c_mul(H[row * N + k], rho[k * N + col]));
    }

    // Compute (rho*H)[row][col]
    Complex rho_h = C_ZERO;
    for (uint k = 0; k < N; k++) {
        rho_h = c_add(rho_h, c_mul(rho[row * N + k], H[k * N + col]));
    }

    // -i * (H*rho - rho*H): multiply by -i means (re, im) -> (im, -re)
    Complex comm = c_sub(h_rho, rho_h);
    Complex unitary_part = { comm.im, -comm.re };  // -i * comm

    // --- Dissipative part: sum over collapse operators ---
    Complex dissipator = C_ZERO;

    for (uint op = 0; op < K; op++) {
        uint base = op * N * N;

        // Compute L_k * rho * L_k†  element [row][col]
        // = sum_a sum_b L[row][a] * rho[a][b] * conj(L[col][b])
        Complex l_rho_ld = C_ZERO;
        for (uint a = 0; a < N; a++) {
            Complex l_ra = L_ops[base + row * N + a];
            for (uint b = 0; b < N; b++) {
                Complex rho_ab = rho[a * N + b];
                Complex l_cb_conj = c_conj(L_ops[base + col * N + b]);
                l_rho_ld = c_add(l_rho_ld, c_mul(c_mul(l_ra, rho_ab), l_cb_conj));
            }
        }

        // Compute (L_k† L_k) for the anticommutator terms:
        // {L†L, rho} = L†L*rho + rho*L†L
        // (L†L * rho)[row][col] = sum_a (L†L)[row][a] * rho[a][col]
        //   where (L†L)[row][a] = sum_m conj(L[m][row]) * L[m][a]
        Complex ldl_rho = C_ZERO;
        for (uint a = 0; a < N; a++) {
            // Compute (L†L)[row][a] on the fly
            Complex ldl_ra = C_ZERO;
            for (uint m = 0; m < N; m++) {
                ldl_ra = c_add(ldl_ra, c_mul(c_conj(L_ops[base + m * N + row]),
                                              L_ops[base + m * N + a]));
            }
            ldl_rho = c_add(ldl_rho, c_mul(ldl_ra, rho[a * N + col]));
        }

        // (rho * L†L)[row][col]
        Complex rho_ldl = C_ZERO;
        for (uint a = 0; a < N; a++) {
            Complex ldl_ac = C_ZERO;
            for (uint m = 0; m < N; m++) {
                ldl_ac = c_add(ldl_ac, c_mul(c_conj(L_ops[base + m * N + a]),
                                              L_ops[base + m * N + col]));
            }
            rho_ldl = c_add(rho_ldl, c_mul(rho[row * N + a], ldl_ac));
        }

        // Dissipator for this operator: L rho L† - 0.5 {L†L, rho}
        Complex anti = c_add(ldl_rho, rho_ldl);
        dissipator = c_add(dissipator, c_sub(l_rho_ld, c_scale(anti, 0.5f)));
    }

    drho[idx] = c_add(unitary_part, dissipator);
}

// ============================================================
// KERNEL 3: BATCH RK4 LINDBLAD INTEGRATION
// ============================================================
// Evolves M density matrices by one dt, each with its own Hamiltonian.
// Shared collapse operators across the batch (for GRAPE: same decoherence
// model, different drive Hamiltonians per time slice).
//
// Each thread handles one of the M batch elements. Inside the thread we
// do a full RK4 step for an NxN density matrix. This is efficient when
// N is small (transmon dim ~ 3-6) and M is large (hundreds of time slices).
//
// Dispatch: grid = M (1D).

kernel void batch_rk4_lindblad(
    device const Complex* H_batch [[buffer(0)]],   // M Hamiltonians, M * N * N
    device Complex* rho_batch [[buffer(1)]],        // M density matrices, M * N * N (read-write)
    device const Complex* L_ops [[buffer(2)]],      // K collapse operators, K * N * N
    constant float& dt [[buffer(3)]],
    constant uint& N [[buffer(4)]],
    constant uint& K [[buffer(5)]],
    constant uint& M [[buffer(6)]],                 // batch size
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= M) return;

    uint nn = N * N;
    uint h_offset = gid * nn;
    uint rho_offset = gid * nn;

    // We need scratch space for RK4 stages. Since N is small (~3-6 for transmons),
    // we can use thread-private arrays. Max N = 8 -> nn = 64.
    // For larger N this kernel should be replaced with a grid-stride variant.
    Complex rho_local[64];       // current rho
    Complex k1[64], k2[64], k3[64], k4[64];
    Complex rho_temp[64];

    // Load rho into local
    for (uint i = 0; i < nn; i++) {
        rho_local[i] = rho_batch[rho_offset + i];
    }

    // --- Helper lambda-like inline: compute Lindblad RHS into output array ---
    // We cannot use actual function pointers in Metal, so we use a macro-like pattern.
    // For each RK4 stage, we compute drho for the entire NxN matrix in-thread.

    // ---- k1 = f(rho) ----
    for (uint r = 0; r < N; r++) {
        for (uint c = 0; c < N; c++) {
            uint rc = r * N + c;

            // Unitary: -i[H, rho]
            Complex h_rho = C_ZERO;
            Complex rho_h = C_ZERO;
            for (uint m = 0; m < N; m++) {
                h_rho = c_add(h_rho, c_mul(H_batch[h_offset + r * N + m], rho_local[m * N + c]));
                rho_h = c_add(rho_h, c_mul(rho_local[r * N + m], H_batch[h_offset + m * N + c]));
            }
            Complex comm = c_sub(h_rho, rho_h);
            Complex result = { comm.im, -comm.re };

            // Dissipative
            for (uint op = 0; op < K; op++) {
                uint base = op * nn;
                Complex l_rho_ld = C_ZERO;
                Complex ldl_rho = C_ZERO;
                Complex rho_ldl = C_ZERO;
                for (uint a = 0; a < N; a++) {
                    for (uint b = 0; b < N; b++) {
                        l_rho_ld = c_add(l_rho_ld, c_mul(c_mul(L_ops[base + r * N + a],
                                          rho_local[a * N + b]), c_conj(L_ops[base + c * N + b])));
                    }
                    Complex ldl_ra = C_ZERO;
                    Complex ldl_ac = C_ZERO;
                    for (uint m2 = 0; m2 < N; m2++) {
                        ldl_ra = c_add(ldl_ra, c_mul(c_conj(L_ops[base + m2 * N + r]),
                                                      L_ops[base + m2 * N + a]));
                        ldl_ac = c_add(ldl_ac, c_mul(c_conj(L_ops[base + m2 * N + a]),
                                                      L_ops[base + m2 * N + c]));
                    }
                    ldl_rho = c_add(ldl_rho, c_mul(ldl_ra, rho_local[a * N + c]));
                    rho_ldl = c_add(rho_ldl, c_mul(rho_local[r * N + a], ldl_ac));
                }
                result = c_add(result, c_sub(l_rho_ld, c_scale(c_add(ldl_rho, rho_ldl), 0.5f)));
            }
            k1[rc] = result;
        }
    }

    // ---- k2 = f(rho + 0.5*dt*k1) ----
    for (uint i = 0; i < nn; i++) {
        rho_temp[i] = c_add(rho_local[i], c_scale(k1[i], 0.5f * dt));
    }
    for (uint r = 0; r < N; r++) {
        for (uint c = 0; c < N; c++) {
            uint rc = r * N + c;
            Complex h_rho = C_ZERO;
            Complex rho_h = C_ZERO;
            for (uint m = 0; m < N; m++) {
                h_rho = c_add(h_rho, c_mul(H_batch[h_offset + r * N + m], rho_temp[m * N + c]));
                rho_h = c_add(rho_h, c_mul(rho_temp[r * N + m], H_batch[h_offset + m * N + c]));
            }
            Complex comm = c_sub(h_rho, rho_h);
            Complex result = { comm.im, -comm.re };
            for (uint op = 0; op < K; op++) {
                uint base = op * nn;
                Complex l_rho_ld = C_ZERO;
                Complex ldl_rho = C_ZERO;
                Complex rho_ldl = C_ZERO;
                for (uint a = 0; a < N; a++) {
                    for (uint b = 0; b < N; b++) {
                        l_rho_ld = c_add(l_rho_ld, c_mul(c_mul(L_ops[base + r * N + a],
                                          rho_temp[a * N + b]), c_conj(L_ops[base + c * N + b])));
                    }
                    Complex ldl_ra = C_ZERO;
                    Complex ldl_ac = C_ZERO;
                    for (uint m2 = 0; m2 < N; m2++) {
                        ldl_ra = c_add(ldl_ra, c_mul(c_conj(L_ops[base + m2 * N + r]),
                                                      L_ops[base + m2 * N + a]));
                        ldl_ac = c_add(ldl_ac, c_mul(c_conj(L_ops[base + m2 * N + a]),
                                                      L_ops[base + m2 * N + c]));
                    }
                    ldl_rho = c_add(ldl_rho, c_mul(ldl_ra, rho_temp[a * N + c]));
                    rho_ldl = c_add(rho_ldl, c_mul(rho_temp[r * N + a], ldl_ac));
                }
                result = c_add(result, c_sub(l_rho_ld, c_scale(c_add(ldl_rho, rho_ldl), 0.5f)));
            }
            k2[rc] = result;
        }
    }

    // ---- k3 = f(rho + 0.5*dt*k2) ----
    for (uint i = 0; i < nn; i++) {
        rho_temp[i] = c_add(rho_local[i], c_scale(k2[i], 0.5f * dt));
    }
    for (uint r = 0; r < N; r++) {
        for (uint c = 0; c < N; c++) {
            uint rc = r * N + c;
            Complex h_rho = C_ZERO;
            Complex rho_h = C_ZERO;
            for (uint m = 0; m < N; m++) {
                h_rho = c_add(h_rho, c_mul(H_batch[h_offset + r * N + m], rho_temp[m * N + c]));
                rho_h = c_add(rho_h, c_mul(rho_temp[r * N + m], H_batch[h_offset + m * N + c]));
            }
            Complex comm = c_sub(h_rho, rho_h);
            Complex result = { comm.im, -comm.re };
            for (uint op = 0; op < K; op++) {
                uint base = op * nn;
                Complex l_rho_ld = C_ZERO;
                Complex ldl_rho = C_ZERO;
                Complex rho_ldl = C_ZERO;
                for (uint a = 0; a < N; a++) {
                    for (uint b = 0; b < N; b++) {
                        l_rho_ld = c_add(l_rho_ld, c_mul(c_mul(L_ops[base + r * N + a],
                                          rho_temp[a * N + b]), c_conj(L_ops[base + c * N + b])));
                    }
                    Complex ldl_ra = C_ZERO;
                    Complex ldl_ac = C_ZERO;
                    for (uint m2 = 0; m2 < N; m2++) {
                        ldl_ra = c_add(ldl_ra, c_mul(c_conj(L_ops[base + m2 * N + r]),
                                                      L_ops[base + m2 * N + a]));
                        ldl_ac = c_add(ldl_ac, c_mul(c_conj(L_ops[base + m2 * N + a]),
                                                      L_ops[base + m2 * N + c]));
                    }
                    ldl_rho = c_add(ldl_rho, c_mul(ldl_ra, rho_temp[a * N + c]));
                    rho_ldl = c_add(rho_ldl, c_mul(rho_temp[r * N + a], ldl_ac));
                }
                result = c_add(result, c_sub(l_rho_ld, c_scale(c_add(ldl_rho, rho_ldl), 0.5f)));
            }
            k3[rc] = result;
        }
    }

    // ---- k4 = f(rho + dt*k3) ----
    for (uint i = 0; i < nn; i++) {
        rho_temp[i] = c_add(rho_local[i], c_scale(k3[i], dt));
    }
    for (uint r = 0; r < N; r++) {
        for (uint c = 0; c < N; c++) {
            uint rc = r * N + c;
            Complex h_rho = C_ZERO;
            Complex rho_h = C_ZERO;
            for (uint m = 0; m < N; m++) {
                h_rho = c_add(h_rho, c_mul(H_batch[h_offset + r * N + m], rho_temp[m * N + c]));
                rho_h = c_add(rho_h, c_mul(rho_temp[r * N + m], H_batch[h_offset + m * N + c]));
            }
            Complex comm = c_sub(h_rho, rho_h);
            Complex result = { comm.im, -comm.re };
            for (uint op = 0; op < K; op++) {
                uint base = op * nn;
                Complex l_rho_ld = C_ZERO;
                Complex ldl_rho = C_ZERO;
                Complex rho_ldl = C_ZERO;
                for (uint a = 0; a < N; a++) {
                    for (uint b = 0; b < N; b++) {
                        l_rho_ld = c_add(l_rho_ld, c_mul(c_mul(L_ops[base + r * N + a],
                                          rho_temp[a * N + b]), c_conj(L_ops[base + c * N + b])));
                    }
                    Complex ldl_ra = C_ZERO;
                    Complex ldl_ac = C_ZERO;
                    for (uint m2 = 0; m2 < N; m2++) {
                        ldl_ra = c_add(ldl_ra, c_mul(c_conj(L_ops[base + m2 * N + r]),
                                                      L_ops[base + m2 * N + a]));
                        ldl_ac = c_add(ldl_ac, c_mul(c_conj(L_ops[base + m2 * N + a]),
                                                      L_ops[base + m2 * N + c]));
                    }
                    ldl_rho = c_add(ldl_rho, c_mul(ldl_ra, rho_temp[a * N + c]));
                    rho_ldl = c_add(rho_ldl, c_mul(rho_temp[r * N + a], ldl_ac));
                }
                result = c_add(result, c_sub(l_rho_ld, c_scale(c_add(ldl_rho, rho_ldl), 0.5f)));
            }
            k4[rc] = result;
        }
    }

    // ---- RK4 combine: rho_new = rho + (dt/6)(k1 + 2*k2 + 2*k3 + k4) ----
    float dt6 = dt / 6.0f;
    for (uint i = 0; i < nn; i++) {
        Complex step = c_add(c_add(k1[i], c_scale(k2[i], 2.0f)),
                             c_add(c_scale(k3[i], 2.0f), k4[i]));
        rho_batch[rho_offset + i] = c_add(rho_local[i], c_scale(step, dt6));
    }
}

// ============================================================
// KERNEL 4: HERMITIAN MATRIX EXPONENTIAL
// ============================================================
// Computes exp(-i * H * dt) for a Hermitian matrix via eigendecomposition.
// Input: real eigenvalues lambda[], eigenvector matrix V (NxN).
// Output: result = V * diag(exp(-i * lambda_k * dt)) * V†
//
// Each thread computes one element result[row][col].
// Dispatch: grid = (N, N).

kernel void hermitian_expm(
    device const float* eigenvalues [[buffer(0)]],    // N real eigenvalues
    device const Complex* eigenvectors [[buffer(1)]],  // NxN eigenvector matrix V (column-major: V[:,k] is k-th eigvec)
    device Complex* result [[buffer(2)]],              // NxN output
    constant float& dt [[buffer(3)]],
    constant uint& N [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint row = gid.y;
    uint col = gid.x;
    if (row >= N || col >= N) return;

    // result[row][col] = sum_k V[row][k] * exp(-i * lambda_k * dt) * conj(V[col][k])
    // V is stored row-major as eigenvectors[row * N + k]
    Complex acc = C_ZERO;
    for (uint k = 0; k < N; k++) {
        float phase = -eigenvalues[k] * dt;
        Complex exp_phase = { cos(phase), sin(phase) };

        Complex v_rk = eigenvectors[row * N + k];
        Complex v_ck_dag = c_conj(eigenvectors[col * N + k]);

        acc = c_add(acc, c_mul(c_mul(v_rk, exp_phase), v_ck_dag));
    }
    result[row * N + col] = acc;
}

// ============================================================
// KERNEL 5: BATCH GATE FIDELITY
// ============================================================
// Computes process fidelity F = |Tr(U_target† @ U)|^2 / d^2
// for M unitaries simultaneously.
//
// U_target and each U in U_batch are stored as d x d complex matrices
// (computational subspace). If the full Hilbert space dim N > d, only
// the top-left d x d block of each matrix is used.
//
// Each thread handles one batch element.
// Dispatch: grid = M (1D).

kernel void batch_gate_fidelity(
    device const Complex* U_target [[buffer(0)]],     // d x d target unitary
    device const Complex* U_batch [[buffer(1)]],      // M unitaries, each d x d, packed M * d * d
    device float* fidelities [[buffer(2)]],           // M output fidelity values
    constant uint& d [[buffer(3)]],                   // computational subspace dimension
    constant uint& N [[buffer(4)]],                   // full Hilbert space dimension (for stride if needed)
    constant uint& M_param [[buffer(5)]],             // batch size
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= M_param) return;

    uint dd = d * d;
    uint u_offset = gid * dd;

    // Compute Tr(U_target† @ U) = sum_{i,j} conj(U_target[i][j]) * U[i][j]
    // which is the Hilbert-Schmidt inner product <U_target, U>.
    Complex trace = C_ZERO;
    for (uint i = 0; i < d; i++) {
        for (uint k = 0; k < d; k++) {
            // (U_target† @ U)[i][i] contributes sum_k conj(U_target[k][i]) * U[k][i]
            // Actually: Tr(A† B) = sum_{i,k} conj(A[k][i]) * B[k][i] = sum_all conj(A[idx]) * B[idx]
            // Let's just do the full trace sum directly.
            uint idx = i * d + k;
            trace = c_add(trace, c_mul(c_conj(U_target[idx]), U_batch[u_offset + idx]));
        }
    }

    // F = |Tr(U_target† U)|^2 / d^2
    float norm_sq = c_norm_sq(trace);
    float d_sq = (float)(d * d);
    fidelities[gid] = norm_sq / d_sq;
}
