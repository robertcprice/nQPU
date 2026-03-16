"""Chemistry-specific variational ansatze for VQE.

Provides parameterized quantum circuit ansatze for molecular simulation:

- :class:`HartreeFockState` -- prepares the Hartree-Fock reference state.
- :class:`UCCSD` -- Unitary Coupled Cluster Singles and Doubles.
- :class:`UCCD` -- Unitary Coupled Cluster Doubles only.
- :class:`kUpCCGSD` -- k-UpCCGSD generalized singles and doubles.
- :class:`HardwareEfficient` -- hardware-efficient variational form.

Each ansatz produces a callable that maps a parameter vector to a state
vector, suitable for use with :class:`QubitHamiltonian.expectation` or
as a cost function in a VQE loop.

References
----------
- Peruzzo et al., Nat. Commun. 5, 4213 (2014) [original VQE paper].
- Romero et al., Quantum Sci. Technol. 4, 014008 (2018) [UCCSD].
- Lee et al., J. Chem. Theory Comput. 15, 311 (2019) [k-UpCCGSD].
- Kandala et al., Nature 549, 242 (2017) [hardware-efficient ansatz].
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from .fermion import (
    FermionicTerm,
    QubitHamiltonian,
    PauliString,
    _transform_fermionic_term_jw,
    _multiply_pauli_strings,
)
from .molecular import Molecule, BasisSet


# ============================================================
# Hartree-Fock reference state
# ============================================================


class HartreeFockState:
    """Prepare the Hartree-Fock reference state as a qubit state vector.

    In the Jordan-Wigner mapping, the HF state is simply the computational
    basis state with the first ``n_electrons`` spin orbitals occupied:
    ``|1100...0>`` for H2, ``|111100...0>`` for LiH (4 electrons), etc.

    Parameters
    ----------
    n_electrons : int
        Number of electrons.
    n_spin_orbitals : int
        Total number of spin orbitals (2 * number of spatial orbitals).
    mapping : str
        Fermion-to-qubit mapping (``'jordan_wigner'`` supported).
    """

    def __init__(
        self,
        n_electrons: int,
        n_spin_orbitals: int,
        mapping: str = "jordan_wigner",
    ) -> None:
        if mapping not in ("jordan_wigner",):
            raise ValueError(
                f"Unsupported mapping '{mapping}'. Only 'jordan_wigner' is supported."
            )
        self.n_electrons = n_electrons
        self.n_spin_orbitals = n_spin_orbitals
        self.mapping = mapping

    @classmethod
    def from_molecule(
        cls,
        molecule: Molecule,
        basis: BasisSet,
        mapping: str = "jordan_wigner",
    ) -> HartreeFockState:
        """Create HF state from a molecule and basis set.

        Parameters
        ----------
        molecule : Molecule
            Molecular system.
        basis : BasisSet
            Basis set.
        mapping : str
            Fermion-to-qubit mapping.

        Returns
        -------
        HartreeFockState
        """
        return cls(
            n_electrons=molecule.num_electrons,
            n_spin_orbitals=2 * basis.num_functions,
            mapping=mapping,
        )

    def state_vector(self) -> np.ndarray:
        """Return the HF state as a 2^n state vector.

        In JW encoding, the HF state has the first n_electrons qubits
        set to |1> and the rest to |0>.  The computational basis index
        is the binary number with the lowest qubits representing the
        first orbitals.

        Returns
        -------
        np.ndarray
            State vector of length ``2^n_spin_orbitals``.
        """
        n = self.n_spin_orbitals
        dim = 2 ** n
        state = np.zeros(dim, dtype=np.complex128)

        # In JW, occupied orbital j means qubit j is |1>
        # Basis state index: sum of 2^j for each occupied j
        idx = 0
        for j in range(self.n_electrons):
            idx += 2 ** j
        state[idx] = 1.0
        return state

    @property
    def occupation_string(self) -> str:
        """Return a binary string showing the occupation pattern."""
        bits = ["1"] * self.n_electrons + ["0"] * (
            self.n_spin_orbitals - self.n_electrons
        )
        return "|" + "".join(bits) + ">"


# ============================================================
# Excitation generators
# ============================================================


def _single_excitations(
    n_electrons: int,
    n_spin_orbitals: int,
) -> list[tuple[int, int]]:
    """Generate all single excitation pairs (i -> a).

    Parameters
    ----------
    n_electrons : int
        Number of occupied spin orbitals.
    n_spin_orbitals : int
        Total number of spin orbitals.

    Returns
    -------
    list of (occupied, virtual)
        Pairs of spin-orbital indices for single excitations.
    """
    occupied = list(range(n_electrons))
    virtual = list(range(n_electrons, n_spin_orbitals))
    excitations = []
    for i in occupied:
        for a in virtual:
            # Conserve spin: alpha->alpha, beta->beta
            if i % 2 == a % 2:
                excitations.append((i, a))
    return excitations


def _double_excitations(
    n_electrons: int,
    n_spin_orbitals: int,
) -> list[tuple[int, int, int, int]]:
    """Generate all double excitation quadruples (i,j -> a,b).

    Parameters
    ----------
    n_electrons : int
        Number of occupied spin orbitals.
    n_spin_orbitals : int
        Total number of spin orbitals.

    Returns
    -------
    list of (i, j, a, b)
        Quadruples of spin-orbital indices for double excitations.
    """
    occupied = list(range(n_electrons))
    virtual = list(range(n_electrons, n_spin_orbitals))
    excitations = []
    for idx_i, i in enumerate(occupied):
        for j in occupied[idx_i + 1:]:
            for idx_a, a in enumerate(virtual):
                for b in virtual[idx_a + 1:]:
                    # Spin conservation: total spin change must be zero
                    spin_occ = (i % 2) + (j % 2)
                    spin_vir = (a % 2) + (b % 2)
                    if spin_occ == spin_vir:
                        excitations.append((i, j, a, b))
    return excitations


def _generalized_single_excitations(
    n_spin_orbitals: int,
) -> list[tuple[int, int]]:
    """Generate all generalized single excitations (p -> q, p < q).

    For k-UpCCGSD, excitations are not restricted to occupied->virtual.
    """
    excitations = []
    for p in range(n_spin_orbitals):
        for q in range(p + 1, n_spin_orbitals):
            if p % 2 == q % 2:
                excitations.append((p, q))
    return excitations


def _generalized_double_excitations(
    n_spin_orbitals: int,
) -> list[tuple[int, int, int, int]]:
    """Generate all generalized double excitations for k-UpCCGSD."""
    excitations = []
    for p in range(n_spin_orbitals):
        for q in range(p + 1, n_spin_orbitals):
            for r in range(q + 1, n_spin_orbitals):
                for s in range(r + 1, n_spin_orbitals):
                    spin_sum = (p % 2) + (q % 2) + (r % 2) + (s % 2)
                    if spin_sum % 2 == 0:
                        excitations.append((p, q, r, s))
    return excitations


# ============================================================
# UCCSD Ansatz
# ============================================================


def _excitation_to_pauli_terms(
    excitation_ops: list[tuple[int, bool]],
    n_qubits: int,
) -> list[tuple[complex, PauliString]]:
    """Convert a fermionic excitation operator to JW Pauli terms.

    The excitation is T - T+, producing an anti-Hermitian operator
    suitable for exp(theta * (T - T+)).
    """
    # T = product of creation/annihilation operators
    t_term = FermionicTerm(coefficient=1.0, operators=excitation_ops)

    # T+ (adjoint)
    t_dag = t_term.adjoint()

    # T - T+ (anti-Hermitian)
    t_pauli = _transform_fermionic_term_jw(t_term, n_qubits)
    tdag_pauli = _transform_fermionic_term_jw(t_dag, n_qubits)

    # Combine: T - T+
    combined = list(t_pauli) + [(-c, p) for c, p in tdag_pauli]
    return combined


class UCCSD:
    """Unitary Coupled Cluster Singles and Doubles ansatz.

    Constructs the UCCSD ansatz as:

    .. math::

        |\\psi(\\theta)\\rangle = e^{T(\\theta) - T^\\dagger(\\theta)} |\\mathrm{HF}\\rangle

    where T = T_1 + T_2 (singles + doubles cluster operators).

    The state is evaluated via first-order Trotterization:
    ``exp(sum_k theta_k G_k) ~ prod_k exp(theta_k G_k)``
    where G_k are the anti-Hermitian excitation generators.

    Parameters
    ----------
    n_electrons : int
        Number of electrons.
    n_spin_orbitals : int
        Number of spin orbitals.
    mapping : str
        Fermion-to-qubit mapping (``'jordan_wigner'``).
    """

    def __init__(
        self,
        n_electrons: int,
        n_spin_orbitals: int,
        mapping: str = "jordan_wigner",
    ) -> None:
        self.n_electrons = n_electrons
        self.n_spin_orbitals = n_spin_orbitals
        self.mapping = mapping

        self.singles = _single_excitations(n_electrons, n_spin_orbitals)
        self.doubles = _double_excitations(n_electrons, n_spin_orbitals)

        # Pre-compute anti-Hermitian generators as matrices
        self._generators: list[np.ndarray] = []
        dim = 2 ** n_spin_orbitals
        self._dim = dim

        for i, a in self.singles:
            ops = [(a, True), (i, False)]  # a+_a a_i
            terms = _excitation_to_pauli_terms(ops, n_spin_orbitals)
            gen_ham = QubitHamiltonian(terms).simplify()
            self._generators.append(gen_ham.to_matrix(n_spin_orbitals))

        for i, j, a, b in self.doubles:
            ops = [(a, True), (b, True), (j, False), (i, False)]
            terms = _excitation_to_pauli_terms(ops, n_spin_orbitals)
            gen_ham = QubitHamiltonian(terms).simplify()
            self._generators.append(gen_ham.to_matrix(n_spin_orbitals))

        # HF reference state
        self._hf = HartreeFockState(n_electrons, n_spin_orbitals)

    @classmethod
    def from_molecule(
        cls,
        molecule: Molecule,
        basis: BasisSet,
        mapping: str = "jordan_wigner",
    ) -> UCCSD:
        """Create UCCSD from molecule and basis.

        Parameters
        ----------
        molecule : Molecule
        basis : BasisSet
        mapping : str

        Returns
        -------
        UCCSD
        """
        return cls(
            n_electrons=molecule.num_electrons,
            n_spin_orbitals=2 * basis.num_functions,
            mapping=mapping,
        )

    def num_parameters(self) -> int:
        """Total number of variational parameters."""
        return len(self.singles) + len(self.doubles)

    def circuit_depth(self) -> int:
        """Estimated circuit depth (Trotter steps * gates per excitation)."""
        # Rough estimate: each excitation requires O(n) CNOTs
        n_excitations = self.num_parameters()
        return n_excitations * self.n_spin_orbitals

    def state_vector(self, params: np.ndarray) -> np.ndarray:
        """Compute the UCCSD state vector for given parameters.

        Parameters
        ----------
        params : np.ndarray
            Variational parameters, one per excitation.

        Returns
        -------
        np.ndarray
            State vector of length ``2^n_spin_orbitals``.
        """
        if len(params) != self.num_parameters():
            raise ValueError(
                f"Expected {self.num_parameters()} parameters, got {len(params)}"
            )

        state = self._hf.state_vector()

        # Apply each exp(theta_k * G_k) via matrix exponential
        for k, theta in enumerate(params):
            if abs(theta) < 1e-15:
                continue
            gen = self._generators[k]
            # exp(theta * G) via eigendecomposition (G is anti-Hermitian/skew-Hermitian)
            # For small matrices this is efficient and exact
            u = _matrix_exponential(theta * gen)
            state = u @ state

        return state


# ============================================================
# UCCD (Doubles only)
# ============================================================


class UCCD:
    """Unitary Coupled Cluster Doubles ansatz (no singles).

    Identical to :class:`UCCSD` but with only double excitations.

    Parameters
    ----------
    n_electrons : int
        Number of electrons.
    n_spin_orbitals : int
        Number of spin orbitals.
    mapping : str
        Fermion-to-qubit mapping.
    """

    def __init__(
        self,
        n_electrons: int,
        n_spin_orbitals: int,
        mapping: str = "jordan_wigner",
    ) -> None:
        self.n_electrons = n_electrons
        self.n_spin_orbitals = n_spin_orbitals
        self.mapping = mapping

        self.doubles = _double_excitations(n_electrons, n_spin_orbitals)

        dim = 2 ** n_spin_orbitals
        self._dim = dim
        self._generators: list[np.ndarray] = []

        for i, j, a, b in self.doubles:
            ops = [(a, True), (b, True), (j, False), (i, False)]
            terms = _excitation_to_pauli_terms(ops, n_spin_orbitals)
            gen_ham = QubitHamiltonian(terms).simplify()
            self._generators.append(gen_ham.to_matrix(n_spin_orbitals))

        self._hf = HartreeFockState(n_electrons, n_spin_orbitals)

    @classmethod
    def from_molecule(
        cls,
        molecule: Molecule,
        basis: BasisSet,
        mapping: str = "jordan_wigner",
    ) -> UCCD:
        """Create UCCD from molecule and basis."""
        return cls(
            n_electrons=molecule.num_electrons,
            n_spin_orbitals=2 * basis.num_functions,
            mapping=mapping,
        )

    def num_parameters(self) -> int:
        """Number of variational parameters (doubles only)."""
        return len(self.doubles)

    def circuit_depth(self) -> int:
        """Estimated circuit depth."""
        return len(self.doubles) * self.n_spin_orbitals

    def state_vector(self, params: np.ndarray) -> np.ndarray:
        """Compute the UCCD state vector."""
        if len(params) != self.num_parameters():
            raise ValueError(
                f"Expected {self.num_parameters()} parameters, got {len(params)}"
            )
        state = self._hf.state_vector()
        for k, theta in enumerate(params):
            if abs(theta) < 1e-15:
                continue
            u = _matrix_exponential(theta * self._generators[k])
            state = u @ state
        return state


# ============================================================
# k-UpCCGSD
# ============================================================


class kUpCCGSD:
    """k-UpCCGSD ansatz: k layers of generalized singles and doubles.

    Unlike UCCSD, excitations are not restricted to occupied->virtual
    transitions, allowing more flexibility in the variational space.

    Parameters
    ----------
    n_spin_orbitals : int
        Number of spin orbitals.
    k : int
        Number of repetition layers (default: 1).
    mapping : str
        Fermion-to-qubit mapping.
    n_electrons : int
        Number of electrons (for HF reference state).
    """

    def __init__(
        self,
        n_spin_orbitals: int,
        k: int = 1,
        mapping: str = "jordan_wigner",
        n_electrons: int = 2,
    ) -> None:
        self.n_spin_orbitals = n_spin_orbitals
        self.k = k
        self.mapping = mapping
        self.n_electrons = n_electrons

        self.gen_singles = _generalized_single_excitations(n_spin_orbitals)
        self.gen_doubles = _generalized_double_excitations(n_spin_orbitals)

        dim = 2 ** n_spin_orbitals
        self._generators_singles: list[np.ndarray] = []
        self._generators_doubles: list[np.ndarray] = []

        for p, q in self.gen_singles:
            ops = [(q, True), (p, False)]
            terms = _excitation_to_pauli_terms(ops, n_spin_orbitals)
            gen_ham = QubitHamiltonian(terms).simplify()
            self._generators_singles.append(gen_ham.to_matrix(n_spin_orbitals))

        for p, q, r, s in self.gen_doubles:
            ops = [(s, True), (r, True), (q, False), (p, False)]
            terms = _excitation_to_pauli_terms(ops, n_spin_orbitals)
            gen_ham = QubitHamiltonian(terms).simplify()
            self._generators_doubles.append(gen_ham.to_matrix(n_spin_orbitals))

        self._hf = HartreeFockState(n_electrons, n_spin_orbitals)

    def num_parameters(self) -> int:
        """Number of variational parameters per layer * k layers."""
        per_layer = len(self.gen_singles) + len(self.gen_doubles)
        return per_layer * self.k

    def circuit_depth(self) -> int:
        """Estimated circuit depth."""
        return self.num_parameters() * self.n_spin_orbitals

    def state_vector(self, params: np.ndarray) -> np.ndarray:
        """Compute the k-UpCCGSD state vector."""
        if len(params) != self.num_parameters():
            raise ValueError(
                f"Expected {self.num_parameters()} parameters, got {len(params)}"
            )
        state = self._hf.state_vector()
        per_layer = len(self.gen_singles) + len(self.gen_doubles)

        for layer in range(self.k):
            offset = layer * per_layer
            # Singles
            for k_idx, gen in enumerate(self._generators_singles):
                theta = params[offset + k_idx]
                if abs(theta) < 1e-15:
                    continue
                u = _matrix_exponential(theta * gen)
                state = u @ state
            # Doubles
            offset_d = offset + len(self.gen_singles)
            for k_idx, gen in enumerate(self._generators_doubles):
                theta = params[offset_d + k_idx]
                if abs(theta) < 1e-15:
                    continue
                u = _matrix_exponential(theta * gen)
                state = u @ state

        return state


# ============================================================
# Hardware-Efficient Ansatz
# ============================================================


class HardwareEfficient:
    """Hardware-efficient variational ansatz.

    Alternating layers of single-qubit rotations (Ry, Rz) and
    entangling gates (CX or CZ), producing a state with minimal
    structural assumptions.

    Parameters
    ----------
    n_qubits : int
        Number of qubits.
    n_layers : int
        Number of variational layers.
    entangler : str
        Entangling gate type (``'cx'`` or ``'cz'``).
    n_electrons : int
        Number of electrons (for HF initial state, default 0 = all-zero state).
    """

    def __init__(
        self,
        n_qubits: int,
        n_layers: int = 1,
        entangler: str = "cx",
        n_electrons: int = 0,
    ) -> None:
        if entangler not in ("cx", "cz"):
            raise ValueError(f"entangler must be 'cx' or 'cz', got '{entangler}'")
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.entangler = entangler
        self.n_electrons = n_electrons

    def num_parameters(self) -> int:
        """Number of variational parameters: 2 angles per qubit per layer."""
        return 2 * self.n_qubits * self.n_layers

    def circuit_depth(self) -> int:
        """Estimated circuit depth."""
        return self.n_layers * (2 + 1)  # 2 rotation layers + 1 entangling layer

    def state_vector(self, params: np.ndarray) -> np.ndarray:
        """Compute the hardware-efficient state vector.

        Parameters
        ----------
        params : np.ndarray
            Variational parameters (2 * n_qubits * n_layers).

        Returns
        -------
        np.ndarray
            State vector.
        """
        if len(params) != self.num_parameters():
            raise ValueError(
                f"Expected {self.num_parameters()} params, got {len(params)}"
            )

        n = self.n_qubits
        dim = 2 ** n

        # Initial state
        if self.n_electrons > 0:
            state = np.zeros(dim, dtype=np.complex128)
            idx = sum(2 ** j for j in range(self.n_electrons))
            state[idx] = 1.0
        else:
            state = np.zeros(dim, dtype=np.complex128)
            state[0] = 1.0

        param_idx = 0
        for layer in range(self.n_layers):
            # Ry rotations
            for q in range(n):
                theta = params[param_idx]
                param_idx += 1
                state = _apply_single_qubit_gate(state, n, q, _ry(theta))

            # Rz rotations
            for q in range(n):
                theta = params[param_idx]
                param_idx += 1
                state = _apply_single_qubit_gate(state, n, q, _rz(theta))

            # Entangling layer: linear connectivity
            for q in range(n - 1):
                if self.entangler == "cx":
                    state = _apply_cnot(state, n, q, q + 1)
                else:
                    state = _apply_cz(state, n, q, q + 1)

        return state


# ============================================================
# Gate primitives for state-vector simulation
# ============================================================


def _ry(theta: float) -> np.ndarray:
    """Single-qubit Ry rotation matrix."""
    c = math.cos(theta / 2)
    s = math.sin(theta / 2)
    return np.array([[c, -s], [s, c]], dtype=np.complex128)


def _rz(theta: float) -> np.ndarray:
    """Single-qubit Rz rotation matrix."""
    return np.array(
        [[np.exp(-1j * theta / 2), 0], [0, np.exp(1j * theta / 2)]],
        dtype=np.complex128,
    )


def _apply_single_qubit_gate(
    state: np.ndarray,
    n_qubits: int,
    qubit: int,
    gate: np.ndarray,
) -> np.ndarray:
    """Apply a single-qubit gate to a state vector."""
    dim = 2 ** n_qubits
    new_state = np.zeros(dim, dtype=np.complex128)

    for i in range(dim):
        bit = (i >> qubit) & 1
        i_flip = i ^ (1 << qubit)

        if bit == 0:
            new_state[i] += gate[0, 0] * state[i] + gate[0, 1] * state[i_flip]
        else:
            new_state[i] += gate[1, 0] * state[i_flip] + gate[1, 1] * state[i]

    return new_state


def _apply_cnot(
    state: np.ndarray,
    n_qubits: int,
    control: int,
    target: int,
) -> np.ndarray:
    """Apply a CNOT gate to a state vector."""
    dim = 2 ** n_qubits
    new_state = state.copy()

    for i in range(dim):
        if (i >> control) & 1:
            j = i ^ (1 << target)
            new_state[i], new_state[j] = state[j], state[i]

    return new_state


def _apply_cz(
    state: np.ndarray,
    n_qubits: int,
    control: int,
    target: int,
) -> np.ndarray:
    """Apply a CZ gate to a state vector."""
    dim = 2 ** n_qubits
    new_state = state.copy()

    for i in range(dim):
        if ((i >> control) & 1) and ((i >> target) & 1):
            new_state[i] = -state[i]

    return new_state


def _matrix_exponential(m: np.ndarray) -> np.ndarray:
    """Compute matrix exponential exp(M) via eigendecomposition.

    For anti-Hermitian or skew-Hermitian matrices (as arise from
    fermionic excitation generators), eigendecomposition is numerically
    stable and efficient for small systems.

    Parameters
    ----------
    m : np.ndarray
        Square matrix.

    Returns
    -------
    np.ndarray
        Matrix exponential exp(M).
    """
    eigenvalues, eigenvectors = np.linalg.eigh(
        1j * m  # Convert to Hermitian for eigh
    )
    # exp(M) = V @ diag(exp(eigenvalues of M)) @ V+
    # Since we computed eigs of iM: eigenvalues of M = -i * eigenvalues_of_iM
    exp_diag = np.exp(-1j * eigenvalues)
    return eigenvectors @ np.diag(exp_diag) @ eigenvectors.conj().T
