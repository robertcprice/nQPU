"""Hamiltonian construction for quantum simulation.

Provides Pauli-string-based Hamiltonian construction and standard
model Hamiltonians used in condensed matter and quantum chemistry:

- **PauliOperator**: Single weighted Pauli string (e.g. 1.5 * "XZIY").
- **SparsePauliHamiltonian**: Sum of weighted Pauli strings with efficient
  matrix construction and operator algebra.
- Model constructors: transverse-field Ising, Heisenberg (XXX/XXZ/XYZ),
  Fermi-Hubbard (Jordan-Wigner mapped), and random Pauli Hamiltonians.
- Exact diagonalisation for ground states of small systems.

References:
    - Sachdev, *Quantum Phase Transitions* (Cambridge, 2011)
    - Hubbard, Proc. Roy. Soc. A 276, 238 (1963)
    - Jordan & Wigner, Z. Phys. 47, 631 (1928)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Pauli matrices (2x2, complex128)
# ---------------------------------------------------------------------------

_I = np.eye(2, dtype=np.complex128)
_X = np.array([[0, 1], [1, 0]], dtype=np.complex128)
_Y = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
_Z = np.array([[1, 0], [0, -1]], dtype=np.complex128)

_PAULI_MAP = {"I": _I, "X": _X, "Y": _Y, "Z": _Z}


def _validate_pauli_label(label: str) -> None:
    """Raise ``ValueError`` if *label* contains non-Pauli characters."""
    for ch in label:
        if ch not in _PAULI_MAP:
            raise ValueError(
                f"Invalid Pauli character '{ch}' in label '{label}'. "
                "Allowed characters: I, X, Y, Z"
            )


# ---------------------------------------------------------------------------
# PauliOperator
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PauliOperator:
    """A single weighted Pauli string.

    Parameters
    ----------
    label : str
        Pauli string such as ``"XZIY"`` where each character is one of
        ``I``, ``X``, ``Y``, ``Z``.  The string length equals the number
        of qubits.
    coeff : complex
        Scalar coefficient (default ``1.0``).

    Examples
    --------
    >>> op = PauliOperator("XZ", coeff=0.5)
    >>> op.n_qubits
    2
    >>> op.matrix().shape
    (4, 4)
    """

    label: str
    coeff: complex = 1.0

    def __post_init__(self) -> None:
        _validate_pauli_label(self.label)
        if len(self.label) == 0:
            raise ValueError("Pauli label must be non-empty.")

    # -- properties ----------------------------------------------------------

    @property
    def n_qubits(self) -> int:
        """Number of qubits this operator acts on."""
        return len(self.label)

    @property
    def is_identity(self) -> bool:
        """True if every Pauli in the string is ``I``."""
        return all(ch == "I" for ch in self.label)

    @property
    def weight(self) -> int:
        """Number of non-identity Paulis (Hamming weight)."""
        return sum(1 for ch in self.label if ch != "I")

    # -- matrix construction -------------------------------------------------

    def matrix(self) -> np.ndarray:
        """Build the full 2^n x 2^n matrix representation.

        Returns
        -------
        np.ndarray
            Dense complex matrix of shape ``(2**n, 2**n)``.
        """
        mat = _PAULI_MAP[self.label[0]]
        for ch in self.label[1:]:
            mat = np.kron(mat, _PAULI_MAP[ch])
        return self.coeff * mat

    # -- algebra -------------------------------------------------------------

    def __mul__(self, other: "PauliOperator") -> "PauliOperator":
        """Multiply two Pauli operators (same qubit count).

        Uses the Pauli multiplication table to compute the product
        analytically without building full matrices.
        """
        if not isinstance(other, PauliOperator):
            return NotImplemented
        if self.n_qubits != other.n_qubits:
            raise ValueError(
                f"Qubit count mismatch: {self.n_qubits} vs {other.n_qubits}"
            )

        new_coeff: complex = self.coeff * other.coeff
        new_label: list[str] = []

        for a, b in zip(self.label, other.label):
            c, phase = _pauli_product(a, b)
            new_coeff *= phase
            new_label.append(c)

        return PauliOperator("".join(new_label), new_coeff)

    def __rmul__(self, scalar: complex) -> "PauliOperator":
        """Scalar multiplication from the left."""
        if isinstance(scalar, (int, float, complex)):
            return PauliOperator(self.label, self.coeff * scalar)
        return NotImplemented

    def __neg__(self) -> "PauliOperator":
        return PauliOperator(self.label, -self.coeff)

    def adjoint(self) -> "PauliOperator":
        """Hermitian conjugate.  Pauli strings are Hermitian so this
        just conjugates the coefficient."""
        return PauliOperator(self.label, np.conj(self.coeff))

    def commutator(self, other: "PauliOperator") -> "SparsePauliHamiltonian":
        """Compute [self, other] = self*other - other*self."""
        ab = self * other
        ba = other * self
        terms = []
        if abs(ab.coeff - ba.coeff) > 1e-15:
            if abs(ab.coeff) > 1e-15:
                terms.append(ab)
            if abs(ba.coeff) > 1e-15:
                terms.append(PauliOperator(ba.label, -ba.coeff))
        return SparsePauliHamiltonian(terms if terms else [],
                                       n_qubits=self.n_qubits)

    def trace(self) -> complex:
        """Trace of the operator: Tr(coeff * P1 x P2 x ...).

        The trace of any Pauli matrix except I is zero.  Therefore the
        trace is ``coeff * 2**n`` when the label is all-identity, else 0.
        """
        if self.is_identity:
            return self.coeff * (2 ** self.n_qubits)
        return 0.0 + 0.0j


def _pauli_product(a: str, b: str) -> Tuple[str, complex]:
    """Single-qubit Pauli product: returns (result_char, phase).

    P_a * P_b = phase * P_c where phase is in {1, -1, i, -i}.
    """
    if a == "I":
        return b, 1.0
    if b == "I":
        return a, 1.0
    if a == b:
        return "I", 1.0

    # Cyclic: X->Y->Z->X gives +i
    _CYCLE = {"X": "Y", "Y": "Z", "Z": "X"}
    if _CYCLE[a] == b:
        return _CYCLE[b], 1j
    else:
        return _CYCLE[a], -1j


# ---------------------------------------------------------------------------
# SparsePauliHamiltonian
# ---------------------------------------------------------------------------


@dataclass
class SparsePauliHamiltonian:
    """Sum of weighted Pauli strings: H = sum_k c_k P_k.

    Parameters
    ----------
    terms : list[PauliOperator]
        Pauli terms forming the Hamiltonian.
    n_qubits : int or None
        Number of qubits.  Inferred from terms if not given.
    """

    terms: List[PauliOperator] = field(default_factory=list)
    n_qubits: Optional[int] = None

    def __post_init__(self) -> None:
        if self.terms:
            inferred = self.terms[0].n_qubits
            for t in self.terms:
                if t.n_qubits != inferred:
                    raise ValueError(
                        "All Pauli terms must act on the same number of qubits."
                    )
            if self.n_qubits is None:
                self.n_qubits = inferred
            elif self.n_qubits != inferred:
                raise ValueError(
                    f"Specified n_qubits={self.n_qubits} but terms have "
                    f"n_qubits={inferred}."
                )
        elif self.n_qubits is None:
            self.n_qubits = 0

    # -- helpers -------------------------------------------------------------

    @property
    def dim(self) -> int:
        """Hilbert space dimension 2^n."""
        return 2 ** self.n_qubits if self.n_qubits else 0

    @property
    def num_terms(self) -> int:
        """Number of Pauli terms."""
        return len(self.terms)

    def add_term(self, label: str, coeff: complex = 1.0) -> None:
        """Append a new Pauli term."""
        op = PauliOperator(label, coeff)
        if self.n_qubits and op.n_qubits != self.n_qubits:
            raise ValueError(
                f"Term has {op.n_qubits} qubits but Hamiltonian has "
                f"{self.n_qubits}."
            )
        if self.n_qubits == 0:
            self.n_qubits = op.n_qubits
        self.terms.append(op)

    def simplify(self) -> "SparsePauliHamiltonian":
        """Combine terms with the same label, drop near-zero terms."""
        coeff_map: dict[str, complex] = {}
        for t in self.terms:
            coeff_map[t.label] = coeff_map.get(t.label, 0.0) + t.coeff
        new_terms = [
            PauliOperator(label, coeff)
            for label, coeff in coeff_map.items()
            if abs(coeff) > 1e-15
        ]
        return SparsePauliHamiltonian(new_terms, n_qubits=self.n_qubits)

    # -- matrix construction -------------------------------------------------

    def matrix(self) -> np.ndarray:
        """Build the full dense matrix representation.

        Returns
        -------
        np.ndarray
            Hermitian matrix of shape ``(dim, dim)``.
        """
        dim = self.dim
        if dim == 0:
            return np.array([], dtype=np.complex128).reshape(0, 0)
        mat = np.zeros((dim, dim), dtype=np.complex128)
        for t in self.terms:
            mat += t.matrix()
        return mat

    # -- algebra -------------------------------------------------------------

    def __add__(self, other: "SparsePauliHamiltonian") -> "SparsePauliHamiltonian":
        if not isinstance(other, SparsePauliHamiltonian):
            return NotImplemented
        if self.n_qubits and other.n_qubits and self.n_qubits != other.n_qubits:
            raise ValueError("Cannot add Hamiltonians with different qubit counts.")
        n = self.n_qubits or other.n_qubits
        return SparsePauliHamiltonian(
            list(self.terms) + list(other.terms), n_qubits=n
        )

    def __rmul__(self, scalar: complex) -> "SparsePauliHamiltonian":
        if isinstance(scalar, (int, float, complex)):
            return SparsePauliHamiltonian(
                [PauliOperator(t.label, scalar * t.coeff) for t in self.terms],
                n_qubits=self.n_qubits,
            )
        return NotImplemented

    def __neg__(self) -> "SparsePauliHamiltonian":
        return (-1) * self

    def __sub__(self, other: "SparsePauliHamiltonian") -> "SparsePauliHamiltonian":
        return self + (-other)

    # -- physics -------------------------------------------------------------

    def expectation(self, state: np.ndarray) -> float:
        """Compute <psi|H|psi> for a given state vector.

        Parameters
        ----------
        state : np.ndarray
            State vector of length ``dim``.

        Returns
        -------
        float
            Real-valued energy expectation value.
        """
        state = np.asarray(state, dtype=np.complex128).ravel()
        if len(state) != self.dim:
            raise ValueError(
                f"State dimension {len(state)} doesn't match "
                f"Hamiltonian dimension {self.dim}."
            )
        mat = self.matrix()
        return float(np.real(state.conj() @ mat @ state))

    def eigenvalues(self) -> np.ndarray:
        """Return sorted eigenvalues via exact diagonalisation.

        Suitable only for small systems (<= ~14 qubits).
        """
        mat = self.matrix()
        evals = np.linalg.eigvalsh(mat)
        return evals

    def ground_state(self) -> Tuple[float, np.ndarray]:
        """Compute ground state energy and eigenvector.

        Returns
        -------
        energy : float
            Ground state energy (lowest eigenvalue).
        state : np.ndarray
            Ground state vector (normalised).
        """
        mat = self.matrix()
        evals, evecs = np.linalg.eigh(mat)
        idx = 0  # eigh returns sorted ascending
        return float(evals[idx]), evecs[:, idx]

    def spectral_gap(self) -> float:
        """Energy gap between ground state and first excited state."""
        evals = self.eigenvalues()
        if len(evals) < 2:
            return 0.0
        return float(evals[1] - evals[0])

    def is_hermitian(self, atol: float = 1e-10) -> bool:
        """Check whether the matrix representation is Hermitian."""
        mat = self.matrix()
        return bool(np.allclose(mat, mat.conj().T, atol=atol))

    def commutator(
        self, other: "SparsePauliHamiltonian"
    ) -> "SparsePauliHamiltonian":
        """Compute [self, other] via matrix commutator.

        For small systems this is the most reliable approach.
        """
        a = self.matrix()
        b = other.matrix()
        comm = a @ b - b @ a
        # Reconstruct as Hamiltonian from matrix (decompose into Pauli basis)
        return _matrix_to_pauli_hamiltonian(comm, self.n_qubits)

    def norm(self) -> float:
        """Frobenius norm of the Hamiltonian matrix."""
        mat = self.matrix()
        return float(np.linalg.norm(mat, "fro"))


# ---------------------------------------------------------------------------
# Pauli decomposition helper
# ---------------------------------------------------------------------------


def _matrix_to_pauli_hamiltonian(
    mat: np.ndarray, n_qubits: int
) -> SparsePauliHamiltonian:
    """Decompose a 2^n x 2^n matrix into the Pauli basis.

    Uses Tr(P_k * M) / 2^n to extract coefficients.
    """
    dim = 2 ** n_qubits
    assert mat.shape == (dim, dim)
    labels = ["I", "X", "Y", "Z"]
    terms: list[PauliOperator] = []

    def _generate_labels(n: int):
        if n == 0:
            yield ""
            return
        for rest in _generate_labels(n - 1):
            for ch in labels:
                yield ch + rest

    for label in _generate_labels(n_qubits):
        op = PauliOperator(label, 1.0)
        coeff = np.trace(op.matrix() @ mat) / dim
        if abs(coeff) > 1e-14:
            terms.append(PauliOperator(label, coeff))

    return SparsePauliHamiltonian(terms, n_qubits=n_qubits)


# ---------------------------------------------------------------------------
# Model Hamiltonians
# ---------------------------------------------------------------------------


def ising_model(
    n: int,
    J: float = 1.0,
    h: float = 1.0,
    periodic: bool = False,
) -> SparsePauliHamiltonian:
    """1D transverse-field Ising model.

    H = -J * sum_<ij> Z_i Z_j  -  h * sum_i X_i

    Parameters
    ----------
    n : int
        Number of spins (qubits).
    J : float
        Coupling strength for ZZ interactions.
    h : float
        Transverse field strength.
    periodic : bool
        If True, add periodic boundary (site n-1 coupled to site 0).

    Returns
    -------
    SparsePauliHamiltonian
    """
    if n < 1:
        raise ValueError("Need at least 1 qubit.")
    terms: list[PauliOperator] = []

    # ZZ coupling
    n_bonds = n if periodic else n - 1
    for i in range(n_bonds):
        j = (i + 1) % n
        label = ["I"] * n
        label[i] = "Z"
        label[j] = "Z"
        terms.append(PauliOperator("".join(label), -J))

    # Transverse field
    for i in range(n):
        label = ["I"] * n
        label[i] = "X"
        terms.append(PauliOperator("".join(label), -h))

    return SparsePauliHamiltonian(terms, n_qubits=n)


def heisenberg_model(
    n: int,
    Jx: float = 1.0,
    Jy: float = 1.0,
    Jz: float = 1.0,
    periodic: bool = False,
) -> SparsePauliHamiltonian:
    """1D Heisenberg spin chain.

    H = sum_<ij> [ Jx X_i X_j + Jy Y_i Y_j + Jz Z_i Z_j ]

    Special cases:
      - Jx = Jy = Jz: isotropic XXX model
      - Jx = Jy != Jz: XXZ model
      - All different: XYZ model

    Parameters
    ----------
    n : int
        Number of spins.
    Jx, Jy, Jz : float
        Exchange coupling constants.
    periodic : bool
        Periodic boundary conditions.
    """
    if n < 2:
        raise ValueError("Heisenberg model requires at least 2 spins.")
    terms: list[PauliOperator] = []

    n_bonds = n if periodic else n - 1
    for i in range(n_bonds):
        j = (i + 1) % n
        for pauli, coupling in [("X", Jx), ("Y", Jy), ("Z", Jz)]:
            if abs(coupling) < 1e-15:
                continue
            label = ["I"] * n
            label[i] = pauli
            label[j] = pauli
            terms.append(PauliOperator("".join(label), coupling))

    return SparsePauliHamiltonian(terms, n_qubits=n)


def hubbard_model(
    n_sites: int,
    t: float = 1.0,
    U: float = 2.0,
) -> SparsePauliHamiltonian:
    """1D Fermi-Hubbard model mapped to qubits via Jordan-Wigner.

    H = -t sum_<ij>,s (c^dag_is c_js + h.c.) + U sum_i n_i_up n_i_down

    Uses 2 * n_sites qubits: first n_sites for spin-up, last for spin-down.
    The hopping term under JW becomes:
        -t/2 (X_i X_{i+1} + Y_i Y_{i+1})
    and the on-site interaction:
        U/4 (I - Z_i)(I - Z_{i+n})

    Parameters
    ----------
    n_sites : int
        Number of lattice sites.
    t : float
        Hopping amplitude.
    U : float
        On-site interaction strength.
    """
    if n_sites < 2:
        raise ValueError("Hubbard model requires at least 2 sites.")
    n_qubits = 2 * n_sites
    terms: list[PauliOperator] = []

    # Hopping for each spin channel
    for spin_offset in [0, n_sites]:
        for i in range(n_sites - 1):
            qi = spin_offset + i
            qj = spin_offset + i + 1
            for pauli in ["X", "Y"]:
                label = ["I"] * n_qubits
                label[qi] = pauli
                label[qj] = pauli
                terms.append(PauliOperator("".join(label), -t / 2.0))

    # On-site interaction: U/4 * (I - Z_up)(I - Z_down)
    # = U/4 * (II - Z_up I - I Z_down + Z_up Z_down)
    for i in range(n_sites):
        q_up = i
        q_down = n_sites + i

        # U/4 * I
        label_ii = ["I"] * n_qubits
        terms.append(PauliOperator("".join(label_ii), U / 4.0))

        # -U/4 * Z_up
        label_z_up = ["I"] * n_qubits
        label_z_up[q_up] = "Z"
        terms.append(PauliOperator("".join(label_z_up), -U / 4.0))

        # -U/4 * Z_down
        label_z_down = ["I"] * n_qubits
        label_z_down[q_down] = "Z"
        terms.append(PauliOperator("".join(label_z_down), -U / 4.0))

        # +U/4 * Z_up Z_down
        label_zz = ["I"] * n_qubits
        label_zz[q_up] = "Z"
        label_zz[q_down] = "Z"
        terms.append(PauliOperator("".join(label_zz), U / 4.0))

    return SparsePauliHamiltonian(terms, n_qubits=n_qubits).simplify()


def random_hamiltonian(
    n_qubits: int,
    n_terms: int = 10,
    seed: Optional[int] = None,
) -> SparsePauliHamiltonian:
    """Generate a random Hermitian Pauli Hamiltonian.

    Each term has a random Pauli string with real coefficient drawn
    uniformly from [-1, 1].

    Parameters
    ----------
    n_qubits : int
        Number of qubits.
    n_terms : int
        Number of Pauli terms.
    seed : int or None
        Random seed for reproducibility.
    """
    rng = np.random.default_rng(seed)
    paulis = ["I", "X", "Y", "Z"]
    terms: list[PauliOperator] = []

    for _ in range(n_terms):
        label = "".join(rng.choice(paulis) for _ in range(n_qubits))
        coeff = float(rng.uniform(-1.0, 1.0))
        terms.append(PauliOperator(label, coeff))

    return SparsePauliHamiltonian(terms, n_qubits=n_qubits).simplify()
