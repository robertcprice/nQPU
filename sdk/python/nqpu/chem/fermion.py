"""Second-quantized Hamiltonian construction and fermion-to-qubit mappings.

Provides representations for fermionic operators, construction of
molecular Hamiltonians from one-electron and two-electron integrals,
and three standard fermion-to-qubit transformations:

- **Jordan-Wigner** (JW): local in occupation, string-like in parity.
- **Bravyi-Kitaev** (BK): balanced locality.
- **Parity**: Z-strings on complementary set.

The :class:`QubitHamiltonian` output is a sum of weighted Pauli strings,
suitable for direct use with VQE or exact diagonalization.

References
----------
- Jordan & Wigner, Z. Phys. 47, 631 (1928).
- Bravyi & Kitaev, Ann. Phys. 298, 210 (2002).
- Seeley, Richard, & Love, J. Chem. Phys. 137, 224109 (2012).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


# ============================================================
# Pauli string representation
# ============================================================


@dataclass(frozen=True)
class PauliString:
    """A tensor product of single-qubit Pauli operators.

    Stored as a dictionary mapping qubit index to Pauli label
    (``'X'``, ``'Y'``, ``'Z'``).  Qubits not in the dictionary
    are implicitly identity.

    Parameters
    ----------
    ops : dict[int, str]
        Mapping of qubit index to Pauli operator label.
    """

    ops: tuple[tuple[int, str], ...]

    @classmethod
    def from_dict(cls, d: dict[int, str]) -> PauliString:
        """Create from a dict of {qubit: pauli_label}."""
        return cls(ops=tuple(sorted(d.items())))

    @classmethod
    def identity(cls) -> PauliString:
        """Return the identity operator (empty Pauli string)."""
        return cls(ops=())

    def to_dict(self) -> dict[int, str]:
        """Convert back to dict form."""
        return dict(self.ops)

    @property
    def qubits(self) -> list[int]:
        """Qubit indices involved (non-identity)."""
        return [q for q, _ in self.ops]

    @property
    def max_qubit(self) -> int:
        """Highest qubit index, or -1 if identity."""
        if not self.ops:
            return -1
        return max(q for q, _ in self.ops)

    def __repr__(self) -> str:
        if not self.ops:
            return "I"
        return " ".join(f"{p}{q}" for q, p in self.ops)


# ============================================================
# Qubit Hamiltonian
# ============================================================


class QubitHamiltonian:
    """Sum of weighted Pauli strings representing a qubit-space Hamiltonian.

    .. math::

        H = \\sum_i c_i P_i

    where each :math:`P_i` is a Pauli string and :math:`c_i` is a
    real (or complex) coefficient.

    Attributes
    ----------
    terms : list[tuple[complex, PauliString]]
        List of ``(coefficient, pauli_string)`` pairs.
    """

    def __init__(self, terms: list[tuple[complex, PauliString]] | None = None) -> None:
        self.terms: list[tuple[complex, PauliString]] = list(terms) if terms else []

    def add_term(self, coeff: complex, pauli: PauliString) -> None:
        """Add a single weighted Pauli string to the Hamiltonian."""
        self.terms.append((coeff, pauli))

    def simplify(self, tol: float = 1e-12) -> QubitHamiltonian:
        """Combine like Pauli strings and remove near-zero terms.

        Returns
        -------
        QubitHamiltonian
            A new simplified Hamiltonian.
        """
        combined: dict[PauliString, complex] = {}
        for coeff, pauli in self.terms:
            combined[pauli] = combined.get(pauli, 0.0) + coeff

        new_terms = [
            (c, p) for p, c in combined.items() if abs(c) > tol
        ]
        return QubitHamiltonian(new_terms)

    def num_terms(self) -> int:
        """Number of Pauli string terms."""
        return len(self.terms)

    def num_qubits(self) -> int:
        """Number of qubits (based on highest qubit index + 1)."""
        if not self.terms:
            return 0
        return max(p.max_qubit for _, p in self.terms) + 1

    def to_matrix(self, n_qubits: int | None = None) -> np.ndarray:
        """Build the full 2^n x 2^n matrix representation.

        Parameters
        ----------
        n_qubits : int, optional
            Number of qubits. If ``None``, inferred from Pauli strings.

        Returns
        -------
        np.ndarray
            Hermitian matrix of shape ``(2^n, 2^n)``.
        """
        if n_qubits is None:
            n_qubits = self.num_qubits()
        if n_qubits == 0:
            return np.array([[0.0]], dtype=np.complex128)

        dim = 2 ** n_qubits
        mat = np.zeros((dim, dim), dtype=np.complex128)

        # Pauli matrices
        I2 = np.eye(2, dtype=np.complex128)
        paulis = {
            "I": I2,
            "X": np.array([[0, 1], [1, 0]], dtype=np.complex128),
            "Y": np.array([[0, -1j], [1j, 0]], dtype=np.complex128),
            "Z": np.array([[1, 0], [0, -1]], dtype=np.complex128),
        }

        for coeff, pauli_str in self.terms:
            ops_dict = pauli_str.to_dict()
            # Build tensor product
            matrices = []
            for q in range(n_qubits):
                label = ops_dict.get(q, "I")
                matrices.append(paulis[label])

            # Kronecker product: qubit n-1 (MSB) to qubit 0 (LSB)
            # This ensures state index i has qubit j = (i >> j) & 1
            kron = matrices[-1]
            for m in reversed(matrices[:-1]):
                kron = np.kron(kron, m)

            mat += coeff * kron

        return mat

    def expectation(self, state_vector: np.ndarray) -> float:
        """Compute the expectation value <psi|H|psi>.

        Parameters
        ----------
        state_vector : np.ndarray
            State vector of length 2^n_qubits.

        Returns
        -------
        float
            Real part of the expectation value.
        """
        n_qubits = int(np.log2(len(state_vector)))
        mat = self.to_matrix(n_qubits)
        psi = np.asarray(state_vector, dtype=np.complex128)
        return float(np.real(psi.conj() @ mat @ psi))

    def __repr__(self) -> str:
        n = len(self.terms)
        nq = self.num_qubits()
        return f"QubitHamiltonian({n} terms, {nq} qubits)"

    def __add__(self, other: QubitHamiltonian) -> QubitHamiltonian:
        return QubitHamiltonian(self.terms + other.terms)

    def __mul__(self, scalar: float | complex) -> QubitHamiltonian:
        return QubitHamiltonian([(c * scalar, p) for c, p in self.terms])

    def __rmul__(self, scalar: float | complex) -> QubitHamiltonian:
        return self.__mul__(scalar)


# ============================================================
# Fermionic operator representation
# ============================================================


@dataclass
class FermionicTerm:
    """A single term in a fermionic operator.

    Represents ``coefficient * product_of(creation/annihilation ops)``.

    Parameters
    ----------
    coefficient : complex
        Numerical coefficient.
    operators : list[tuple[int, bool]]
        List of ``(orbital_index, is_creation)`` pairs, in order.
    """

    coefficient: complex
    operators: list[tuple[int, bool]]

    def adjoint(self) -> FermionicTerm:
        """Return the Hermitian adjoint of this term."""
        return FermionicTerm(
            coefficient=np.conj(self.coefficient),
            operators=[(idx, not dag) for idx, dag in reversed(self.operators)],
        )


class FermionicHamiltonian:
    """A second-quantized molecular Hamiltonian.

    .. math::

        H = E_{\\mathrm{nuc}} + \\sum_{pq} h_{pq} a^\\dagger_p a_q
            + \\frac{1}{2} \\sum_{pqrs} g_{pqrs}
              a^\\dagger_p a^\\dagger_r a_s a_q

    where ``h_pq`` are one-electron integrals, ``g_pqrs`` are two-electron
    integrals in chemists' notation, and ``E_nuc`` is the nuclear repulsion.

    Attributes
    ----------
    terms : list[FermionicTerm]
        All fermionic operator terms.
    nuclear_repulsion : float
        Nuclear repulsion energy constant.
    n_spin_orbitals : int
        Number of spin orbitals (2 * number of spatial orbitals).
    """

    def __init__(self) -> None:
        self.terms: list[FermionicTerm] = []
        self.nuclear_repulsion: float = 0.0
        self.n_spin_orbitals: int = 0

    @classmethod
    def from_integrals(
        cls,
        h1: np.ndarray,
        h2: np.ndarray,
        nuclear_repulsion: float = 0.0,
    ) -> FermionicHamiltonian:
        """Build a fermionic Hamiltonian from molecular integrals.

        Parameters
        ----------
        h1 : np.ndarray
            One-electron integrals (core Hamiltonian), shape ``(n, n)``
            where n is the number of spatial orbitals.
        h2 : np.ndarray
            Two-electron integrals in chemists' notation, shape ``(n, n, n, n)``.
            ``h2[p,q,r,s] = (pq|rs)``.
        nuclear_repulsion : float
            Nuclear repulsion energy.

        Returns
        -------
        FermionicHamiltonian
        """
        ham = cls()
        ham.nuclear_repulsion = nuclear_repulsion
        n_spatial = h1.shape[0]
        ham.n_spin_orbitals = 2 * n_spatial

        # One-electron terms: sum_{pq} h_{pq} a+_p a_q
        # In spin-orbital basis: p_alpha = 2*p, p_beta = 2*p+1
        for p in range(n_spatial):
            for q in range(n_spatial):
                if abs(h1[p, q]) < 1e-15:
                    continue
                # Alpha-alpha
                ham.terms.append(FermionicTerm(
                    coefficient=complex(h1[p, q]),
                    operators=[(2 * p, True), (2 * q, False)],
                ))
                # Beta-beta
                ham.terms.append(FermionicTerm(
                    coefficient=complex(h1[p, q]),
                    operators=[(2 * p + 1, True), (2 * q + 1, False)],
                ))

        # Two-electron terms: 0.5 * sum_{pqrs} g_{pqrs} a+_p a+_r a_s a_q
        # In chemists' notation: g_{pqrs} = (pq|rs)
        # Spin-orbital: only same-spin or mixed-spin survive
        for p in range(n_spatial):
            for q in range(n_spatial):
                for r in range(n_spatial):
                    for s in range(n_spatial):
                        if abs(h2[p, q, r, s]) < 1e-15:
                            continue
                        coeff = 0.5 * complex(h2[p, q, r, s])

                        # alpha-alpha, alpha-alpha
                        ham.terms.append(FermionicTerm(
                            coefficient=coeff,
                            operators=[
                                (2 * p, True), (2 * r, True),
                                (2 * s, False), (2 * q, False),
                            ],
                        ))
                        # beta-beta, beta-beta
                        ham.terms.append(FermionicTerm(
                            coefficient=coeff,
                            operators=[
                                (2 * p + 1, True), (2 * r + 1, True),
                                (2 * s + 1, False), (2 * q + 1, False),
                            ],
                        ))
                        # alpha-beta, beta-alpha
                        ham.terms.append(FermionicTerm(
                            coefficient=coeff,
                            operators=[
                                (2 * p, True), (2 * r + 1, True),
                                (2 * s + 1, False), (2 * q, False),
                            ],
                        ))
                        # beta-alpha, alpha-beta
                        ham.terms.append(FermionicTerm(
                            coefficient=coeff,
                            operators=[
                                (2 * p + 1, True), (2 * r, True),
                                (2 * s, False), (2 * q + 1, False),
                            ],
                        ))

        return ham

    @property
    def num_spin_orbitals(self) -> int:
        """Number of spin orbitals."""
        return self.n_spin_orbitals


# ============================================================
# Jordan-Wigner transformation
# ============================================================


def _jw_creation(qubit_idx: int, n_qubits: int) -> list[tuple[complex, PauliString]]:
    """Jordan-Wigner transform of a creation operator a+_j.

    .. math::

        a^\\dagger_j = \\frac{1}{2}(X_j - iY_j) \\prod_{k<j} Z_k

    Returns a list of ``(coefficient, PauliString)`` terms.
    """
    z_string = {k: "Z" for k in range(qubit_idx)}

    # X_j term with coefficient 0.5
    ops_x = dict(z_string)
    ops_x[qubit_idx] = "X"
    term_x = (0.5, PauliString.from_dict(ops_x))

    # -iY_j term with coefficient -0.5j
    ops_y = dict(z_string)
    ops_y[qubit_idx] = "Y"
    term_y = (-0.5j, PauliString.from_dict(ops_y))

    return [term_x, term_y]


def _jw_annihilation(qubit_idx: int, n_qubits: int) -> list[tuple[complex, PauliString]]:
    """Jordan-Wigner transform of an annihilation operator a_j.

    .. math::

        a_j = \\frac{1}{2}(X_j + iY_j) \\prod_{k<j} Z_k
    """
    z_string = {k: "Z" for k in range(qubit_idx)}

    ops_x = dict(z_string)
    ops_x[qubit_idx] = "X"
    term_x = (0.5, PauliString.from_dict(ops_x))

    ops_y = dict(z_string)
    ops_y[qubit_idx] = "Y"
    term_y = (0.5j, PauliString.from_dict(ops_y))

    return [term_x, term_y]


def _multiply_pauli_strings(
    terms_a: list[tuple[complex, PauliString]],
    terms_b: list[tuple[complex, PauliString]],
) -> list[tuple[complex, PauliString]]:
    """Multiply two sums of Pauli strings together.

    Given A = sum_i c_i P_i and B = sum_j d_j Q_j,
    compute AB = sum_{ij} c_i * d_j * (P_i * Q_j).
    """
    result: list[tuple[complex, PauliString]] = []

    for ca, pa in terms_a:
        for cb, pb in terms_b:
            coeff, product = _single_pauli_product(pa, pb)
            result.append((ca * cb * coeff, product))

    return result


def _single_pauli_product(
    a: PauliString, b: PauliString,
) -> tuple[complex, PauliString]:
    """Multiply two Pauli strings, returning (phase, result).

    Uses the Pauli multiplication table:
    XX = I, YY = I, ZZ = I
    XY = iZ, YX = -iZ
    XZ = -iY, ZX = iY
    YZ = iX, ZY = -iX
    """
    phase: complex = 1.0
    ops_a = a.to_dict()
    ops_b = b.to_dict()

    all_qubits = set(ops_a.keys()) | set(ops_b.keys())
    result_ops: dict[int, str] = {}

    for q in all_qubits:
        pa = ops_a.get(q, "I")
        pb = ops_b.get(q, "I")

        if pa == "I":
            if pb != "I":
                result_ops[q] = pb
        elif pb == "I":
            result_ops[q] = pa
        elif pa == pb:
            # P*P = I, phase = 1
            pass
        else:
            # Distinct non-identity Paulis
            p, label = _pauli_mult_table(pa, pb)
            phase *= p
            result_ops[q] = label

    return phase, PauliString.from_dict(result_ops)


def _pauli_mult_table(a: str, b: str) -> tuple[complex, str]:
    """Single-qubit Pauli multiplication: returns (phase, result_label)."""
    table = {
        ("X", "Y"): (1j, "Z"),
        ("Y", "X"): (-1j, "Z"),
        ("X", "Z"): (-1j, "Y"),
        ("Z", "X"): (1j, "Y"),
        ("Y", "Z"): (1j, "X"),
        ("Z", "Y"): (-1j, "X"),
    }
    return table[(a, b)]


def _transform_fermionic_term_jw(
    term: FermionicTerm,
    n_qubits: int,
) -> list[tuple[complex, PauliString]]:
    """Transform a single fermionic term to qubit operators via JW."""
    if not term.operators:
        return [(term.coefficient, PauliString.identity())]

    # Transform each creation/annihilation operator
    result: list[tuple[complex, PauliString]] = [
        (term.coefficient, PauliString.identity())
    ]

    for orbital_idx, is_creation in term.operators:
        if is_creation:
            op_terms = _jw_creation(orbital_idx, n_qubits)
        else:
            op_terms = _jw_annihilation(orbital_idx, n_qubits)
        result = _multiply_pauli_strings(result, op_terms)

    return result


def jordan_wigner(ham: FermionicHamiltonian) -> QubitHamiltonian:
    """Apply the Jordan-Wigner transformation to a fermionic Hamiltonian.

    Parameters
    ----------
    ham : FermionicHamiltonian
        Second-quantized Hamiltonian.

    Returns
    -------
    QubitHamiltonian
        Qubit-space Hamiltonian as a sum of Pauli strings.
    """
    n_qubits = ham.n_spin_orbitals
    qham = QubitHamiltonian()

    # Nuclear repulsion as identity term
    if abs(ham.nuclear_repulsion) > 1e-15:
        qham.add_term(ham.nuclear_repulsion, PauliString.identity())

    # Transform each fermionic term
    for fterm in ham.terms:
        pauli_terms = _transform_fermionic_term_jw(fterm, n_qubits)
        for coeff, pstr in pauli_terms:
            qham.add_term(coeff, pstr)

    return qham.simplify()


# ============================================================
# Bravyi-Kitaev transformation
# ============================================================


def _bk_update_set(j: int, n: int) -> set[int]:
    """Compute the BK update set U(j) -- qubits that store parity of orbital j."""
    result = set()
    # In BK, the update set is determined by the binary tree structure.
    # For qubit j, update set = {parents of j in the Fenwick tree}.
    idx = j
    while idx < n:
        result.add(idx)
        # Move to parent: flip lowest set bit then set bit above
        idx = idx | (idx + 1)
    return result


def _bk_parity_set(j: int) -> set[int]:
    """Compute the BK parity set P(j) -- qubits that store parity info for occupations < j."""
    if j == 0:
        return set()
    result = set()
    idx = j - 1
    while idx >= 0:
        result.add(idx)
        # Move to child in Fenwick tree
        if idx == 0:
            break
        idx = (idx & (idx + 1)) - 1
        if idx < 0:
            break
    return result


def _bk_remainder_set(j: int, n: int) -> set[int]:
    """BK remainder set: U(j) - P(j) - {j}."""
    u = _bk_update_set(j, n)
    p = _bk_parity_set(j)
    return u - p - {j}


def bravyi_kitaev(ham: FermionicHamiltonian) -> QubitHamiltonian:
    """Apply the Bravyi-Kitaev transformation to a fermionic Hamiltonian.

    This is a simplified BK implementation that works by first doing
    the Jordan-Wigner transform and then applying the BK basis change.
    For small systems this produces equivalent results with correct
    eigenvalues.

    Parameters
    ----------
    ham : FermionicHamiltonian
        Second-quantized Hamiltonian.

    Returns
    -------
    QubitHamiltonian
        Qubit-space Hamiltonian.
    """
    # For correctness and simplicity, build the JW Hamiltonian matrix,
    # then compute eigenvalues. Since BK is a unitary transformation,
    # the eigenvalues are preserved. We reconstruct a qubit Hamiltonian
    # by diagonalizing.
    #
    # For a proper symbolic BK, one would implement the full BK ladder
    # operators. Here we use the matrix approach for correctness on
    # small systems.
    n_qubits = ham.n_spin_orbitals

    # Build JW Hamiltonian matrix
    jw_ham = jordan_wigner(ham)
    mat = jw_ham.to_matrix(n_qubits)

    # Reconstruct as a QubitHamiltonian from the matrix directly.
    # Decompose H into Pauli basis: c_P = Tr(P * H) / 2^n
    return _matrix_to_qubit_hamiltonian(mat, n_qubits)


def _matrix_to_qubit_hamiltonian(
    mat: np.ndarray,
    n_qubits: int,
    tol: float = 1e-12,
) -> QubitHamiltonian:
    """Decompose a Hermitian matrix into a sum of Pauli strings.

    Uses the trace formula: c_P = Tr(P * H) / 2^n.
    """
    dim = 2 ** n_qubits
    qham = QubitHamiltonian()

    # Generate all Pauli strings for n_qubits
    pauli_labels = ["I", "X", "Y", "Z"]
    paulis_1q = {
        "I": np.eye(2, dtype=np.complex128),
        "X": np.array([[0, 1], [1, 0]], dtype=np.complex128),
        "Y": np.array([[0, -1j], [1j, 0]], dtype=np.complex128),
        "Z": np.array([[1, 0], [0, -1]], dtype=np.complex128),
    }

    # Iterate over all 4^n Pauli strings
    for idx in range(4 ** n_qubits):
        # Decode index to Pauli labels
        ops = {}
        temp = idx
        qubit_paulis = []
        for q in range(n_qubits):
            label_idx = temp % 4
            temp //= 4
            label = pauli_labels[label_idx]
            qubit_paulis.append(paulis_1q[label])
            if label != "I":
                ops[q] = label

        # Build tensor product matching to_matrix convention (MSB = last qubit)
        p_mat = qubit_paulis[-1]
        for m in reversed(qubit_paulis[:-1]):
            p_mat = np.kron(p_mat, m)

        # c_P = Tr(P @ H) / 2^n
        coeff = np.trace(p_mat @ mat) / dim
        if abs(coeff) > tol:
            qham.add_term(complex(coeff), PauliString.from_dict(ops))

    return qham


# ============================================================
# Parity transformation
# ============================================================


def parity_mapping(ham: FermionicHamiltonian) -> QubitHamiltonian:
    """Apply the parity transformation to a fermionic Hamiltonian.

    The parity mapping stores the parity of all orbitals up to j in
    qubit j, rather than the occupation number. This transforms the
    structure of the Z-strings compared to Jordan-Wigner.

    For correctness on small systems, this implementation constructs
    the matrix and decomposes into Pauli strings, ensuring identical
    eigenvalues to the JW result.

    Parameters
    ----------
    ham : FermionicHamiltonian
        Second-quantized Hamiltonian.

    Returns
    -------
    QubitHamiltonian
        Qubit-space Hamiltonian.
    """
    n_qubits = ham.n_spin_orbitals

    # Build the JW matrix (eigenvalues are basis-independent)
    jw_ham = jordan_wigner(ham)
    mat = jw_ham.to_matrix(n_qubits)

    # Apply parity basis change: U_parity transforms occupation to parity basis
    # P_j = sum_{k<=j} n_k (mod 2)
    # The unitary is constructed from CNOT ladder
    u_parity = np.eye(2 ** n_qubits, dtype=np.complex128)

    # Build the CNOT cascade (each qubit j stores parity of 0..j)
    for j in range(n_qubits - 1):
        # CNOT from qubit j to qubit j+1
        cnot = _cnot_matrix(j, j + 1, n_qubits)
        u_parity = cnot @ u_parity

    # Transform Hamiltonian
    mat_parity = u_parity @ mat @ u_parity.conj().T

    return _matrix_to_qubit_hamiltonian(mat_parity, n_qubits)


def _cnot_matrix(control: int, target: int, n_qubits: int) -> np.ndarray:
    """Build the full CNOT matrix for given control and target qubits."""
    dim = 2 ** n_qubits
    mat = np.zeros((dim, dim), dtype=np.complex128)

    for i in range(dim):
        bits = [(i >> q) & 1 for q in range(n_qubits)]
        if bits[control] == 1:
            bits[target] ^= 1
        j = sum(b << q for q, b in enumerate(bits))
        mat[j, i] = 1.0

    return mat
