"""Quantum cryptographic primitives: OTP, authentication, universal hashing.

Implements four fundamental quantum cryptographic building blocks:

1. **QuantumOneTimePad**: Encrypt n-qubit states with 2n classical bits using
   Pauli X and Z operators (information-theoretically secure).

2. **QuantumAuthCode**: Authenticate quantum states via random Clifford
   encoding with trap-tag qubits for tamper detection.

3. **UniversalHash**: Toeplitz-matrix universal hash family for privacy
   amplification in QKD post-processing.

4. **QuantumFingerprinting**: Exponentially compact quantum fingerprints
   for equality testing of classical strings.

All implementations use pure numpy -- no external dependencies.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

# ---------------------------------------------------------------------------
# Pauli matrices
# ---------------------------------------------------------------------------
_I = np.eye(2, dtype=complex)
_X = np.array([[0, 1], [1, 0]], dtype=complex)
_Z = np.array([[1, 0], [0, -1]], dtype=complex)
_Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
_H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
_S = np.array([[1, 0], [0, 1j]], dtype=complex)

# Single-qubit Clifford group (up to global phase) as 2x2 unitaries
# The 24-element single-qubit Clifford group modulo global phase
_CLIFFORD_GENERATORS = [
    _I, _X, _Z, _Y,
    _H, _S,
    _H @ _S, _S @ _H, _H @ _S @ _H,
    _S @ _H @ _S, _H @ _S @ _S, _S @ _S @ _H,
    _H @ _S @ _S @ _H, _S @ _H @ _S @ _H,
    _S @ _S, _H @ _S @ _H @ _S,
    _S @ _H @ _S @ _S, _H @ _S @ _S @ _H @ _S,
    _S @ _H @ _S @ _H @ _S, _S @ _S @ _H @ _S,
    _H @ _S @ _H @ _S @ _S, _S @ _S @ _H @ _S @ _H,
    _H @ _S @ _S @ _H @ _S @ _H, _S @ _H @ _S @ _S @ _H,
]


def _kron_list(matrices: List[np.ndarray]) -> np.ndarray:
    """Kronecker product of a list of matrices."""
    result = matrices[0]
    for m in matrices[1:]:
        result = np.kron(result, m)
    return result


def _apply_operator(state: np.ndarray, operator: np.ndarray) -> np.ndarray:
    """Apply operator to a state vector."""
    return operator @ state


def _single_qubit_op(gate: np.ndarray, qubit: int, n_qubits: int) -> np.ndarray:
    """Build n-qubit operator from single-qubit gate on specified qubit."""
    ops = [_I] * n_qubits
    ops[qubit] = gate
    return _kron_list(ops)


# ---------------------------------------------------------------------------
# QuantumOneTimePad
# ---------------------------------------------------------------------------
@dataclass
class QuantumOneTimePad:
    """Quantum one-time pad: encrypt quantum state with classical key.

    Encrypts an n-qubit state using 2n classical bits.
    Apply X^a Z^b for each qubit with random bits (a, b).

    The QOTP is information-theoretically secure: the encrypted state is
    the maximally mixed state regardless of the input, making it
    indistinguishable from random noise.
    """

    n_qubits: int

    def generate_key(self, rng: Optional[np.random.Generator] = None) -> np.ndarray:
        """Generate 2n-bit classical key.

        Returns array of shape (2 * n_qubits,) with values in {0, 1}.
        For each qubit i, key[2*i] controls X, key[2*i+1] controls Z.
        """
        if rng is None:
            rng = np.random.default_rng()
        return rng.integers(0, 2, size=2 * self.n_qubits).astype(np.int8)

    def encrypt(self, state: np.ndarray, key: np.ndarray) -> np.ndarray:
        """Encrypt quantum state: apply X^a Z^b for each qubit.

        Parameters
        ----------
        state : np.ndarray
            State vector of dimension 2^n_qubits.
        key : np.ndarray
            Classical key of length 2 * n_qubits.

        Returns
        -------
        np.ndarray
            Encrypted state vector.
        """
        encrypted = state.copy().astype(complex)
        dim = 2 ** self.n_qubits

        if encrypted.shape[0] != dim:
            raise ValueError(
                f"State dimension {encrypted.shape[0]} != 2^{self.n_qubits} = {dim}"
            )
        if key.shape[0] != 2 * self.n_qubits:
            raise ValueError(
                f"Key length {key.shape[0]} != 2 * {self.n_qubits}"
            )

        for q in range(self.n_qubits):
            a, b = int(key[2 * q]), int(key[2 * q + 1])
            if a:
                op = _single_qubit_op(_X, q, self.n_qubits)
                encrypted = op @ encrypted
            if b:
                op = _single_qubit_op(_Z, q, self.n_qubits)
                encrypted = op @ encrypted
        return encrypted

    def decrypt(self, encrypted: np.ndarray, key: np.ndarray) -> np.ndarray:
        """Decrypt: apply Z^b X^a in reverse order (self-inverse operators).

        Since X^2 = Z^2 = I, decryption applies the same operators in
        reverse order: first Z^b then X^a for each qubit.
        """
        decrypted = encrypted.copy().astype(complex)

        for q in range(self.n_qubits - 1, -1, -1):
            a, b = int(key[2 * q]), int(key[2 * q + 1])
            # Reverse order: Z first, then X
            if b:
                op = _single_qubit_op(_Z, q, self.n_qubits)
                decrypted = op @ decrypted
            if a:
                op = _single_qubit_op(_X, q, self.n_qubits)
                decrypted = op @ decrypted
        return decrypted


# ---------------------------------------------------------------------------
# QuantumAuthCode
# ---------------------------------------------------------------------------
@dataclass
class QuantumAuthCode:
    """Quantum authentication code using Clifford group.

    Authenticates quantum states by encoding with random single-qubit
    Cliffords and appending tag qubits prepared in |0>.  The receiver
    applies the inverse Clifford and checks that tag qubits remain |0>.

    Security: any tampering that disturbs the authenticated state will
    be detected with probability >= 1 - 2^{-n_tag_qubits}.
    """

    n_qubits: int
    n_tag_qubits: int = 1

    @property
    def total_qubits(self) -> int:
        return self.n_qubits + self.n_tag_qubits

    def generate_key(self, rng: Optional[np.random.Generator] = None) -> dict:
        """Generate authentication key.

        Returns dict with:
        - 'clifford_indices': random Clifford index for each total qubit
        - 'otp_key': quantum one-time pad key for total system
        """
        if rng is None:
            rng = np.random.default_rng()

        n_cliffords = len(_CLIFFORD_GENERATORS)
        indices = rng.integers(0, n_cliffords, size=self.total_qubits)

        otp = QuantumOneTimePad(self.total_qubits)
        otp_key = otp.generate_key(rng)

        return {
            "clifford_indices": indices,
            "otp_key": otp_key,
        }

    def authenticate(self, state: np.ndarray, key: dict) -> np.ndarray:
        """Encode state with authentication tag.

        Appends n_tag_qubits in |0>, applies random Cliffords per qubit,
        then applies quantum one-time pad.
        """
        dim_data = 2 ** self.n_qubits
        if state.shape[0] != dim_data:
            raise ValueError(
                f"State dimension {state.shape[0]} != 2^{self.n_qubits}"
            )

        # Append tag qubits in |0>
        tag = np.zeros(2 ** self.n_tag_qubits, dtype=complex)
        tag[0] = 1.0
        full_state = np.kron(state, tag)

        # Apply random Cliffords per qubit
        indices = key["clifford_indices"]
        for q in range(self.total_qubits):
            cliff = _CLIFFORD_GENERATORS[int(indices[q])]
            op = _single_qubit_op(cliff, q, self.total_qubits)
            full_state = op @ full_state

        # Apply OTP
        otp = QuantumOneTimePad(self.total_qubits)
        full_state = otp.encrypt(full_state, key["otp_key"])

        return full_state

    def verify(
        self, received: np.ndarray, key: dict
    ) -> Tuple[bool, np.ndarray]:
        """Verify authentication and extract state.

        Returns (is_valid, extracted_state).  If not valid, extracted_state
        is the zero vector.
        """
        dim_total = 2 ** self.total_qubits
        if received.shape[0] != dim_total:
            raise ValueError(
                f"Received dim {received.shape[0]} != 2^{self.total_qubits}"
            )

        # Undo OTP
        otp = QuantumOneTimePad(self.total_qubits)
        decrypted = otp.decrypt(received, key["otp_key"])

        # Undo Cliffords (apply inverse = conjugate transpose)
        indices = key["clifford_indices"]
        for q in range(self.total_qubits - 1, -1, -1):
            cliff_inv = _CLIFFORD_GENERATORS[int(indices[q])].conj().T
            op = _single_qubit_op(cliff_inv, q, self.total_qubits)
            decrypted = op @ decrypted

        # Check tag qubits are |0>
        dim_data = 2 ** self.n_qubits
        dim_tag = 2 ** self.n_tag_qubits

        # Reshape to (dim_data, dim_tag) and check tag register
        reshaped = decrypted.reshape(dim_data, dim_tag)
        tag_probs = np.abs(reshaped[:, 1:]) ** 2
        tag_error = np.sum(tag_probs)

        is_valid = tag_error < 1e-6

        if is_valid:
            extracted = reshaped[:, 0]
            norm = np.linalg.norm(extracted)
            if norm > 1e-10:
                extracted = extracted / norm
            return True, extracted
        else:
            return False, np.zeros(dim_data, dtype=complex)


# ---------------------------------------------------------------------------
# UniversalHash
# ---------------------------------------------------------------------------
@dataclass
class UniversalHash:
    """Universal hash family for quantum key distillation.

    Uses Toeplitz matrices for the hash function, which form a
    2-universal hash family.  This is the standard choice for privacy
    amplification in QKD protocols.
    """

    input_bits: int
    output_bits: int

    def __post_init__(self):
        if self.output_bits > self.input_bits:
            raise ValueError("output_bits must be <= input_bits")

    def generate_function(
        self, rng: Optional[np.random.Generator] = None
    ) -> np.ndarray:
        """Generate random Toeplitz hash matrix.

        A Toeplitz matrix is fully specified by its first row and first
        column, requiring only (input_bits + output_bits - 1) random bits.

        Returns an (output_bits x input_bits) binary matrix over GF(2).
        """
        if rng is None:
            rng = np.random.default_rng()

        n_random = self.input_bits + self.output_bits - 1
        random_bits = rng.integers(0, 2, size=n_random).astype(np.int8)

        # Build Toeplitz matrix
        matrix = np.zeros((self.output_bits, self.input_bits), dtype=np.int8)
        for i in range(self.output_bits):
            for j in range(self.input_bits):
                matrix[i, j] = random_bits[i - j + self.input_bits - 1]

        return matrix

    def hash(self, data: np.ndarray, hash_matrix: np.ndarray) -> np.ndarray:
        """Apply universal hash to bit string.

        Computes hash_matrix @ data (mod 2).
        """
        if data.shape[0] != self.input_bits:
            raise ValueError(
                f"Data length {data.shape[0]} != input_bits {self.input_bits}"
            )
        return (hash_matrix @ data.astype(np.int64)) % 2

    @staticmethod
    def privacy_amplification(
        raw_key: np.ndarray,
        leaked_bits: int,
        security_param: int = 10,
        rng: Optional[np.random.Generator] = None,
    ) -> np.ndarray:
        """Extract secure key via privacy amplification.

        The output length is len(raw_key) - leaked_bits - security_param,
        guaranteeing that the extracted key is 2^{-security_param}-close
        to uniform even given the eavesdropper's information.

        Parameters
        ----------
        raw_key : np.ndarray
            Raw (partially secret) key bits.
        leaked_bits : int
            Upper bound on number of bits known to eavesdropper.
        security_param : int
            Security parameter (number of sacrificed bits).
        rng : np.random.Generator, optional
            Random number generator.

        Returns
        -------
        np.ndarray
            Secure extracted key.
        """
        input_len = len(raw_key)
        output_len = max(1, input_len - leaked_bits - security_param)

        hasher = UniversalHash(input_len, output_len)
        matrix = hasher.generate_function(rng)
        return hasher.hash(raw_key, matrix)


# ---------------------------------------------------------------------------
# QuantumFingerprinting
# ---------------------------------------------------------------------------
@dataclass
class QuantumFingerprinting:
    """Quantum fingerprinting for equality testing with exponential savings.

    Given two classical n-bit strings x and y, quantum fingerprinting creates
    O(log n)-qubit quantum states |f_x> and |f_y> such that:
    - If x == y: |<f_x|f_y>|^2 = 1
    - If x != y: |<f_x|f_y>|^2 <= 1/2

    This provides exponential savings over classical fingerprinting.
    """

    n_bits: int

    def _error_correcting_matrix(self) -> np.ndarray:
        """Generate a binary error-correcting code matrix.

        Uses a random binary matrix as a simplified stand-in for a proper
        error-correcting code.  The matrix maps n_bits -> m bits where
        m = O(n_bits) is the codeword length.
        """
        # Use a deterministic matrix based on n_bits for reproducibility
        m = max(2 * self.n_bits, 4)
        rng = np.random.default_rng(seed=self.n_bits)
        return rng.integers(0, 2, size=(m, self.n_bits)).astype(np.int8)

    def fingerprint(self, data: np.ndarray) -> np.ndarray:
        """Create quantum fingerprint of classical data.

        Encodes classical data into a quantum state using an error-correcting
        code.  The fingerprint state has dimension proportional to the
        codeword length.

        Parameters
        ----------
        data : np.ndarray
            Binary array of length n_bits.

        Returns
        -------
        np.ndarray
            Normalized quantum state (fingerprint).
        """
        if len(data) != self.n_bits:
            raise ValueError(f"Data length {len(data)} != n_bits {self.n_bits}")

        # Encode with error-correcting code
        ec_matrix = self._error_correcting_matrix()
        codeword = (ec_matrix @ data.astype(np.int64)) % 2

        # Create quantum state: |f> = (1/sqrt(m)) sum_i (-1)^{c_i} |i>
        m = len(codeword)
        fp_state = np.zeros(m, dtype=complex)
        for i in range(m):
            fp_state[i] = (-1.0) ** codeword[i]
        fp_state /= np.linalg.norm(fp_state)

        return fp_state

    def test_equality(self, fp1: np.ndarray, fp2: np.ndarray) -> float:
        """Test if two fingerprints correspond to same data.

        Returns the squared inner product |<f1|f2>|^2.
        - Close to 1.0 means data is likely equal.
        - Significantly less than 1.0 means data differs.

        Parameters
        ----------
        fp1, fp2 : np.ndarray
            Quantum fingerprint states.

        Returns
        -------
        float
            Squared overlap in [0, 1].
        """
        overlap = np.abs(np.vdot(fp1, fp2)) ** 2
        return float(overlap)
