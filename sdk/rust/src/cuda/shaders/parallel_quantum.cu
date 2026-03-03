#include <cuComplex.h>
#include <math.h>

extern "C" {

__device__ void get_indices(long long idx, int target, long long* i0, long long* i1) {
    *i0 = ((idx >> target) << (target + 1)) | (idx & ((1LL << target) - 1));
    *i1 = *i0 | (1LL << target);
}

// Pauli-Y: [[0, -i], [i, 0]]
__global__ void y_gate(cuDoubleComplex* state, int n, int target) {
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (1LL << (n - 1))) return;
    long long i0, i1;
    get_indices(idx, target, &i0, &i1);
    cuDoubleComplex a0 = state[i0];
    cuDoubleComplex a1 = state[i1];
    state[i0] = make_cuDoubleComplex(a1.y, -a1.x); // -i * a1
    state[i1] = make_cuDoubleComplex(-a0.y, a0.x); // i * a0
}

// Pauli-Z: [[1, 0], [0, -1]]
__global__ void z_gate(cuDoubleComplex* state, int n, int target) {
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (1LL << (n - 1))) return;
    long long i0, i1;
    get_indices(idx, target, &i0, &i1);
    state[i1] = make_cuDoubleComplex(-state[i1].x, -state[i1].y);
}

// S gate (Phase): [[1, 0], [0, i]]
__global__ void s_gate(cuDoubleComplex* state, int n, int target) {
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (1LL << (n - 1))) return;
    long long i0, i1;
    get_indices(idx, target, &i0, &i1);
    cuDoubleComplex a1 = state[i1];
    state[i1] = make_cuDoubleComplex(-a1.y, a1.x);
}

// T gate (pi/8): [[1, 0], [0, exp(i*pi/4)]]
__global__ void t_gate(cuDoubleComplex* state, int n, int target) {
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (1LL << (n - 1))) return;
    long long i0, i1;
    get_indices(idx, target, &i0, &i1);
    double s, c;
    sincos(M_PI / 4.0, &s, &c);
    cuDoubleComplex phase = make_cuDoubleComplex(c, s);
    state[i1] = cuCmul(state[i1], phase);
}

// Rotation-X: exp(-i * theta/2 * X)
__global__ void rx_gate(cuDoubleComplex* state, int n, int target, double theta) {
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (1LL << (n - 1))) return;
    long long i0, i1;
    get_indices(idx, target, &i0, &i1);
    double s, c;
    sincos(theta / 2.0, &s, &c);
    cuDoubleComplex a0 = state[i0];
    cuDoubleComplex a1 = state[i1];
    state[i0] = cuCadd(cuCmul(make_cuDoubleComplex(c, 0), a0), cuCmul(make_cuDoubleComplex(0, -s), a1));
    state[i1] = cuCadd(cuCmul(make_cuDoubleComplex(0, -s), a0), cuCmul(make_cuDoubleComplex(c, 0), a1));
}

// Controlled-Z
__global__ void cz_gate(cuDoubleComplex* state, int n, int control, int target) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (1LL << n)) return;
    if (((i >> control) & 1) && ((i >> target) & 1)) {
        state[i] = make_cuDoubleComplex(-state[i].x, -state[i].y);
    }
}

// SWAP
__global__ void swap_gate(cuDoubleComplex* state, int n, int q1, int q2) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (1LL << n)) return;
    bool b1 = (i >> q1) & 1;
    bool b2 = (i >> q2) & 1;
    if (b1 != b2) {
        long long j = i ^ (1LL << q1) ^ (1LL << q2);
        if (i < j) {
            cuDoubleComplex temp = state[i];
            state[i] = state[j];
            state[j] = temp;
        }
    }
}

// ... existing H, X, CNOT, Attention ...
__global__ void apply_single_qubit_gate(cuDoubleComplex* state, int n, int target, cuDoubleComplex m00, cuDoubleComplex m01, cuDoubleComplex m10, cuDoubleComplex m11) {
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (1LL << (n - 1))) return;
    long long i0, i1;
    get_indices(idx, target, &i0, &i1);
    cuDoubleComplex a0 = state[i0]; cuDoubleComplex a1 = state[i1];
    state[i0] = cuCadd(cuCmul(m00, a0), cuCmul(m01, a1));
    state[i1] = cuCadd(cuCmul(m10, a0), cuCmul(m11, a1));
}

__global__ void hadamard(cuDoubleComplex* state, int n, int target) {
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (1LL << (n - 1))) return;
    long long i0, i1;
    get_indices(idx, target, &i0, &i1);
    cuDoubleComplex a0 = state[i0]; cuDoubleComplex a1 = state[i1];
    double inv_sqrt2 = 0.7071067811865475;
    state[i0] = make_cuDoubleComplex((a0.x + a1.x) * inv_sqrt2, (a0.y + a1.y) * inv_sqrt2);
    state[i1] = make_cuDoubleComplex((a0.x - a1.x) * inv_sqrt2, (a0.y - a1.y) * inv_sqrt2);
}

__global__ void cnot(cuDoubleComplex* state, int n, int control, int target) {
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (1LL << (n - 1))) return;
    long long i0, i1;
    get_indices(idx, target, &i0, &i1);
    if ((i0 >> control) & 1) {
        cuDoubleComplex temp = state[i0];
        state[i0] = state[i1];
        state[i1] = temp;
    }
}

__global__ void x_gate(cuDoubleComplex* state, int n, int target) {
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (1LL << (n - 1))) return;
    long long i0, i1;
    get_indices(idx, target, &i0, &i1);
    cuDoubleComplex temp = state[i0];
    state[i0] = state[i1];
    state[i1] = temp;
}

}
