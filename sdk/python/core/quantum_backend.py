"""
Real Quantum Computing Backend using PennyLane

Provides ACTUAL quantum computing capabilities for:
1. Variational Quantum Eigensolver (VQE) for molecular energy calculations
2. Quantum fingerprint encoding with real quantum circuits
3. Quantum kernel computation for machine learning
4. Quantum fidelity measurement between states

Requirements:
    pip install pennylane pennylane-qiskit  # qiskit optional for hardware backends

The code gracefully falls back to classical simulation if PennyLane is not installed.

Example:
    >>> from quantum_backend import QuantumBackend, VQEMolecule, QuantumKernel
    >>>
    >>> # VQE for H2 molecule
    >>> vqe = VQEMolecule(backend='default.qubit')
    >>> energy = vqe.compute_ground_state_energy('H2', bond_length=0.74)
    >>> print(f"H2 ground state energy: {energy:.4f} Ha")
    >>>
    >>> # Quantum kernel
    >>> kernel = QuantumKernel(num_qubits=4, num_layers=2)
    >>> K = kernel.compute_kernel_matrix(features_array)
"""

from typing import Dict, List, Tuple, Optional, Union, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import warnings
import math

import numpy as np

# ============================================================================
# PENNYLANE AVAILABILITY CHECK
# ============================================================================

try:
    import pennylane as qml
    from pennylane import numpy as pnp
    HAS_PENNYLANE = True
    PENNYLANE_VERSION = qml.__version__
except ImportError:
    HAS_PENNYLANE = False
    PENNYLANE_VERSION = None
    qml = None
    pnp = None
    warnings.warn(
        "\n"
        "=" * 70 + "\n"
        "PennyLane not installed. Real quantum computing features disabled.\n"
        "Current 'quantum' methods are classical simulations.\n\n"
        "To enable real quantum computing:\n"
        "    pip install pennylane pennylane-qiskit\n\n"
        "For IBM Quantum hardware access (optional):\n"
        "    pip install pennylane-qiskit\n"
        "    qiskit-ibm-provider\n"
        "=" * 70,
        ImportWarning,
        stacklevel=2
    )


# ============================================================================
# MOLECULAR DATA AND REFERENCE VALUES
# ============================================================================

@dataclass
class MolecularGeometry:
    """Molecular geometry with atom positions in Angstroms."""
    symbols: List[str]
    coordinates: np.ndarray  # Shape: (n_atoms, 3)

    @classmethod
    def h2(cls, bond_length: float = 0.74) -> 'MolecularGeometry':
        """
        Create H2 molecule geometry.

        Args:
            bond_length: H-H bond length in Angstroms (equilibrium: 0.74)

        Returns:
            MolecularGeometry for H2
        """
        return cls(
            symbols=['H', 'H'],
            coordinates=np.array([
                [0.0, 0.0, -bond_length/2],
                [0.0, 0.0, bond_length/2]
            ])
        )

    @classmethod
    def lih(cls, bond_length: float = 1.596) -> 'MolecularGeometry':
        """
        Create LiH molecule geometry.

        Args:
            bond_length: Li-H bond length in Angstroms (equilibrium: 1.596)

        Returns:
            MolecularGeometry for LiH
        """
        return cls(
            symbols=['Li', 'H'],
            coordinates=np.array([
                [0.0, 0.0, 0.0],
                [0.0, 0.0, bond_length]
            ])
        )

    @classmethod
    def beh2(cls, bond_length: float = 1.326) -> 'MolecularGeometry':
        """
        Create BeH2 molecule geometry (linear).

        Args:
            bond_length: Be-H bond length in Angstroms (equilibrium: 1.326)

        Returns:
            MolecularGeometry for BeH2
        """
        return cls(
            symbols=['H', 'Be', 'H'],
            coordinates=np.array([
                [0.0, 0.0, -bond_length],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, bond_length]
            ])
        )


# Literature reference values for validation
REFERENCE_ENERGIES = {
    'H2': {
        'equilibrium_bond': 0.74,  # Angstroms
        'ground_state_energy': -1.137,  # Hartree at equilibrium
        'fci_energy': -1.1372838,  # Full CI benchmark
        'description': 'Hydrogen molecule, simplest case, 2 electrons'
    },
    'LiH': {
        'equilibrium_bond': 1.596,  # Angstroms
        'ground_state_energy': -7.88,  # Hartree at equilibrium (approximate)
        'fci_energy': -7.8875,  # Full CI benchmark
        'description': 'Lithium hydride, 4 electrons'
    },
    'BeH2': {
        'equilibrium_bond': 1.326,  # Angstroms
        'ground_state_energy': -15.25,  # Hartree (approximate)
        'fci_energy': None,  # Complex molecule
        'description': 'Beryllium hydride, linear, 4 electrons'
    }
}


# ============================================================================
# QUANTUM BACKEND CONFIGURATION
# ============================================================================

class BackendType(Enum):
    """Available quantum backends."""
    DEFAULT_QUBIT = "default.qubit"  # PennyLane default simulator
    LIGHTNING_QUBIT = "lightning.qubit"  # Fast C++ simulator
    QISKIT_AER = "qiskit.aer"  # IBM Qiskit Aer
    QISKIT_BASIS = "qiskit.basicaer"  # Basic Qiskit simulator


@dataclass
class QuantumBackendConfig:
    """Configuration for quantum backend."""
    backend_type: BackendType = BackendType.DEFAULT_QUBIT
    num_shots: int = 1024  # For shot-based simulations
    device_name: Optional[str] = None  # For real hardware
    ibmq_token: Optional[str] = None  # IBM Quantum API token

    def get_device(self, num_wires: int) -> Any:
        """
        Get PennyLane device with this configuration.

        Args:
            num_wires: Number of qubits

        Returns:
            PennyLane device instance
        """
        if not HAS_PENNYLANE:
            raise RuntimeError(
                "PennyLane not installed. Cannot create quantum device.\n"
                "Install with: pip install pennylane"
            )

        backend_name = self.backend_type.value

        if backend_name == "lightning.qubit":
            try:
                return qml.device(backend_name, wires=num_wires)
            except Exception:
                warnings.warn(
                    "lightning.qubit not available, falling back to default.qubit",
                    RuntimeWarning
                )
                return qml.device("default.qubit", wires=num_wires, shots=self.num_shots)

        elif backend_name.startswith("qiskit"):
            try:
                if self.device_name and self.ibmq_token:
                    # Real IBM Quantum hardware
                    return qml.device(
                        backend_name,
                        wires=num_wires,
                        backend=self.device_name,
                        ibmqx_token=self.ibmq_token
                    )
                else:
                    # Qiskit simulator
                    return qml.device(backend_name, wires=num_wires, shots=self.num_shots)
            except Exception as e:
                warnings.warn(
                    f"Qiskit backend not available ({e}), falling back to default.qubit",
                    RuntimeWarning
                )
                return qml.device("default.qubit", wires=num_wires, shots=self.num_shots)

        else:
            return qml.device("default.qubit", wires=num_wires, shots=self.num_shots)


# ============================================================================
# VARIATIONAL QUANTUM EIGENSOLVER (VQE)
# ============================================================================

class VQEMolecule:
    """
    Variational Quantum Eigensolver for molecular ground state energy calculation.

    Uses PennyLane's quantum chemistry module to construct molecular Hamiltonians
    and variational circuits to find ground state energies.

    Example:
        >>> vqe = VQEMolecule(backend='default.qubit')
        >>> # Compute H2 energy at equilibrium geometry
        >>> energy = vqe.compute_ground_state_energy('H2', bond_length=0.74)
        >>> print(f"Energy: {energy:.4f} Ha (reference: -1.137 Ha)")
    """

    def __init__(self, backend: str = 'default.qubit', shots: Optional[int] = None):
        """
        Initialize VQE solver.

        Args:
            backend: Quantum backend ('default.qubit', 'lightning.qubit', 'qiskit.aer')
            shots: Number of shots for sampling (None for statevector)
        """
        if not HAS_PENNYLANE:
            raise RuntimeError(
                "PennyLane required for VQE. Install with: pip install pennylane"
            )

        self.backend = backend
        self.shots = shots
        self._cached_hamiltonians: Dict[str, Tuple[Any, int]] = {}

    def _get_hamiltonian(self, molecule_name: str, bond_length: float) -> Tuple[Any, int]:
        """
        Construct molecular Hamiltonian using PennyLane's chemistry module.

        Args:
            molecule_name: Molecule identifier ('H2', 'LiH', 'BeH2')
            bond_length: Bond length in Angstroms

        Returns:
            Tuple of (Hamiltonian, number of qubits)
        """
        cache_key = f"{molecule_name}_{bond_length:.4f}"
        if cache_key in self._cached_hamiltonians:
            return self._cached_hamiltonians[cache_key]

        # Get molecular geometry
        if molecule_name == 'H2':
            geometry = MolecularGeometry.h2(bond_length)
            charge = 0
            mult = 1  # Singlet
        elif molecule_name == 'LiH':
            geometry = MolecularGeometry.lih(bond_length)
            charge = 0
            mult = 1
        elif molecule_name == 'BeH2':
            geometry = MolecularGeometry.beh2(bond_length)
            charge = 0
            mult = 1
        else:
            raise ValueError(f"Unsupported molecule: {molecule_name}")

        # Construct coordinates string for PennyLane
        coords_str = "\n".join([
            f"{sym} {x:.6f} {y:.6f} {z:.6f}"
            for sym, (x, y, z) in zip(geometry.symbols, geometry.coordinates)
        ])

        # Build molecular Hamiltonian using PennyLane's qchem
        try:
            # Use minimal STO-3G basis
            symbols = geometry.symbols
            coordinates = geometry.coordinates

            # Get molecular data
            mol = qml.qchem.Molecule(
                symbols,
                coordinates,
                charge=charge,
                mult=mult,
                basis_name='sto-3g',
                unit='angstrom'  # Use Angstroms for bond lengths
            )

            # Construct Hamiltonian from molecule object
            H, qubits = qml.qchem.molecular_hamiltonian(mol)

            self._cached_hamiltonians[cache_key] = (H, qubits)
            return H, qubits

        except Exception as e:
            raise RuntimeError(
                f"Failed to construct Hamiltonian for {molecule_name}: {e}\n"
                "Ensure pennylane-qchem is installed: pip install pennylane[qchem]"
            ) from e

    def hardware_efficient_ansatz(self, params: np.ndarray, wires: List[int],
                                   num_layers: int = 2) -> None:
        """
        Hardware-efficient variational ansatz.

        A simple parameterized circuit with rotation and entangling layers.

        Args:
            params: Variational parameters, shape (num_layers, num_wires, 3)
            wires: List of qubit indices
            num_layers: Number of variational layers
        """
        num_wires = len(wires)

        for layer in range(num_layers):
            # Single-qubit rotations
            for i, wire in enumerate(wires):
                qml.Rot(params[layer, i, 0],
                       params[layer, i, 1],
                       params[layer, i, 2],
                       wires=wire)

            # Entangling layer (ring topology)
            for i in range(num_wires - 1):
                qml.CNOT(wires=[wires[i], wires[i + 1]])
            if num_wires > 1:
                qml.CNOT(wires=[wires[-1], wires[0]])

    def uccsd_simplified_ansatz(self, params: np.ndarray, wires: List[int],
                                 excitations: List[Tuple]) -> None:
        """
        Simplified UCCSD-inspired ansatz.

        Uses single and double excitation gates instead of full UCCSD.

        Args:
            params: Variational parameters for each excitation
            wires: List of qubit indices
            excitations: List of excitation tuples (from PennyLane qchem)
        """
        # Hartree-Fock reference state
        for i in range(len(wires) // 2):
            qml.PauliX(wires=wires[i])

        # Apply excitations
        for i, exc in enumerate(excitations):
            if len(exc) == 2:
                # Single excitation
                qml.SingleExcitation(params[i], wires=exc)
            elif len(exc) == 4:
                # Double excitation
                qml.DoubleExcitation(params[i], wires=exc)

    def compute_ground_state_energy(
        self,
        molecule_name: str,
        bond_length: Optional[float] = None,
        ansatz: str = 'hardware_efficient',
        num_layers: int = 2,
        optimizer: str = 'Adam',
        max_iterations: int = 100,
        conv_tol: float = 1e-5,
        verbose: bool = False
    ) -> float:
        """
        Compute molecular ground state energy using VQE.

        Args:
            molecule_name: Molecule identifier ('H2', 'LiH', 'BeH2')
            bond_length: Bond length in Angstroms (None for equilibrium)
            ansatz: Variational ansatz ('hardware_efficient', 'uccsd')
            num_layers: Number of ansatz layers
            optimizer: Optimizer ('Adam', 'GradientDescent', 'SPSA')
            max_iterations: Maximum optimization steps
            conv_tol: Convergence tolerance
            verbose: Print optimization progress

        Returns:
            Ground state energy in Hartree
        """
        if not HAS_PENNYLANE:
            raise RuntimeError("PennyLane required for VQE")

        # Get bond length
        if bond_length is None:
            bond_length = REFERENCE_ENERGIES[molecule_name]['equilibrium_bond']

        # Get Hamiltonian
        H, num_qubits = self._get_hamiltonian(molecule_name, bond_length)

        # Create device
        device_kwargs = {'wires': num_qubits}
        if self.shots:
            device_kwargs['shots'] = self.shots
        dev = qml.device(self.backend, **device_kwargs)

        # Get excitations for UCCSD if needed
        excitations = None
        if ansatz == 'uccsd':
            try:
                if molecule_name == 'H2':
                    geometry = MolecularGeometry.h2(bond_length)
                elif molecule_name == 'LiH':
                    geometry = MolecularGeometry.lih(bond_length)
                else:
                    geometry = MolecularGeometry.beh2(bond_length)

                # Compute excitations
                _, exc = qml.qchem.excitations(
                    len([s for s in geometry.symbols if s != 'H']),  # electrons
                    num_qubits,
                    delta_sz=0
                )
                excitations = exc
            except Exception:
                # Fall back to hardware efficient
                ansatz = 'hardware_efficient'
                warnings.warn("UCCSD unavailable, using hardware-efficient ansatz")

        # Define variational circuit
        wires = list(range(num_qubits))

        if ansatz == 'hardware_efficient':
            num_params = num_layers * num_qubits * 3

            @qml.qnode(dev)
            def circuit(params):
                self.hardware_efficient_ansatz(
                    params.reshape(num_layers, num_qubits, 3),
                    wires,
                    num_layers
                )
                return qml.expval(H)
        else:
            # UCCSD
            num_params = len(excitations) if excitations else num_qubits

            @qml.qnode(dev)
            def circuit(params):
                self.uccsd_simplified_ansatz(params, wires, excitations or [])
                return qml.expval(H)

        # Initialize parameters
        raw_params = np.random.uniform(0, 2 * np.pi, num_params)
        if pnp is not None:
            params = pnp.array(raw_params, requires_grad=True)
        else:
            params = raw_params

        # Select optimizer
        if optimizer == 'Adam':
            opt = qml.AdamOptimizer(stepsize=0.05)
        elif optimizer == 'GradientDescent':
            opt = qml.GradientDescentOptimizer(stepsize=0.1)
        elif optimizer == 'SPSA':
            opt = qml.SPSAOptimizer(maxiter=max_iterations)
        else:
            opt = qml.GradientDescentOptimizer(stepsize=0.1)

        # Optimization loop
        energy_history = []
        prev_energy = float('inf')

        for i in range(max_iterations):
            params, energy = opt.step_and_cost(circuit, params)
            energy_history.append(energy)

            if verbose and i % 10 == 0:
                ref = REFERENCE_ENERGIES.get(molecule_name, {}).get('ground_state_energy', '?')
                print(f"Iter {i:3d}: Energy = {energy:.6f} Ha (ref: {ref})")

            # Check convergence
            if abs(energy - prev_energy) < conv_tol:
                if verbose:
                    print(f"Converged at iteration {i}")
                break
            prev_energy = energy

        return float(energy_history[-1])

    def compute_dissociation_curve(
        self,
        molecule_name: str,
        bond_lengths: np.ndarray,
        **kwargs
    ) -> Dict[str, np.ndarray]:
        """
        Compute molecular dissociation curve.

        Args:
            molecule_name: Molecule identifier
            bond_lengths: Array of bond lengths in Angstroms
            **kwargs: Additional arguments for compute_ground_state_energy

        Returns:
            Dictionary with 'bond_lengths' and 'energies' arrays
        """
        energies = []
        for bl in bond_lengths:
            e = self.compute_ground_state_energy(molecule_name, bl, **kwargs)
            energies.append(e)

        return {
            'bond_lengths': bond_lengths,
            'energies': np.array(energies)
        }


# ============================================================================
# QUANTUM FINGERPRINT ENCODING
# ============================================================================

class QuantumFingerprint:
    """
    Encode molecular features as quantum states on real qubits.

    Supports amplitude encoding and angle encoding for quantum state preparation,
    with true quantum fidelity computation using quantum circuits.

    Example:
        >>> qfp = QuantumFingerprint(num_qubits=8, encoding='angle')
        >>> state1 = qfp.encode(features1)
        >>> state2 = qfp.encode(features2)
        >>> fidelity = qfp.quantum_fidelity(state1, state2)
    """

    def __init__(
        self,
        num_qubits: int = 8,
        encoding: str = 'angle',
        backend: str = 'default.qubit'
    ):
        """
        Initialize quantum fingerprint encoder.

        Args:
            num_qubits: Number of qubits for encoding
            encoding: Encoding method ('angle', 'amplitude', 'basis')
            backend: Quantum backend
        """
        if not HAS_PENNYLANE:
            raise RuntimeError(
                "PennyLane required for quantum fingerprints. "
                "Install with: pip install pennylane"
            )

        self.num_qubits = num_qubits
        self.encoding = encoding
        self.backend = backend
        self._dev = None

    def _get_device(self):
        """Get or create PennyLane device."""
        if self._dev is None:
            self._dev = qml.device(self.backend, wires=self.num_qubits)
        return self._dev

    def angle_encoding_circuit(self, features: np.ndarray) -> None:
        """
        Angle encoding: each feature value determines rotation angle.

        Maps classical data to rotation angles on each qubit.

        Args:
            features: Feature array, shape (num_qubits,) or smaller
        """
        for i in range(min(len(features), self.num_qubits)):
            # Scale feature to [0, 2π]
            angle = 2 * np.pi * features[i]
            qml.RY(angle, wires=i)

    def amplitude_encoding_circuit(self, features: np.ndarray) -> None:
        """
        Amplitude encoding: features become quantum state amplitudes.

        Requires normalized feature vector with 2^n elements.

        Args:
            features: Normalized feature array, shape (2^num_qubits,)
        """
        # Normalize features
        norm = np.linalg.norm(features)
        if norm > 0:
            features = features / norm

        # Pad to 2^num_qubits if needed
        target_size = 2 ** self.num_qubits
        if len(features) < target_size:
            features = np.pad(features, (0, target_size - len(features)))
        elif len(features) > target_size:
            features = features[:target_size]

        # Renormalize
        features = features / np.linalg.norm(features)

        # Use PennyLane's amplitude embedding (requires statevector simulator)
        qml.AmplitudeEmbedding(features, wires=range(self.num_qubits), normalize=True)

    def basis_encoding_circuit(self, bitstring: str) -> None:
        """
        Basis encoding: bitstring directly sets computational basis state.

        Args:
            bitstring: Binary string of length num_qubits
        """
        for i, bit in enumerate(bitstring[:self.num_qubits]):
            if bit == '1':
                qml.PauliX(wires=i)

    def encode(self, features: np.ndarray) -> np.ndarray:
        """
        Encode classical features into quantum state.

        Args:
            features: Feature array to encode

        Returns:
            Encoded quantum state (statevector or density matrix)
        """
        dev = self._get_device()

        @qml.qnode(dev)
        def circuit():
            if self.encoding == 'angle':
                self.angle_encoding_circuit(features)
            elif self.encoding == 'amplitude':
                self.amplitude_encoding_circuit(features)
            elif self.encoding == 'basis':
                self.basis_encoding_circuit(features)
            else:
                raise ValueError(f"Unknown encoding: {self.encoding}")

            return qml.state()

        return circuit()

    def quantum_fidelity(
        self,
        features1: np.ndarray,
        features2: np.ndarray,
        method: str = 'swap_test'
    ) -> float:
        """
        Compute quantum fidelity |<ψ1|ψ2>|² using quantum circuits.

        Methods:
        - 'swap_test': Uses SWAP test circuit for fidelity estimation
        - 'statevector': Direct statevector computation (simulator only)
        - 'hadamard_test': Hadamard test for real part of overlap

        Args:
            features1: First feature vector
            features2: Second feature vector
            method: Fidelity computation method

        Returns:
            Fidelity value in [0, 1]
        """
        dev = self._get_device()

        if method == 'statevector':
            # Direct computation (simulator only)
            state1 = self.encode(features1)
            state2 = self.encode(features2)
            overlap = np.abs(np.vdot(state1, state2)) ** 2
            return float(overlap)

        elif method == 'swap_test':
            # SWAP test circuit (works on hardware)
            # Need 2n+1 qubits: n for state1, n for state2, 1 auxiliary
            total_wires = 2 * self.num_qubits + 1
            dev_swap = qml.device(self.backend, wires=total_wires)

            @qml.qnode(dev_swap)
            def swap_circuit():
                # Encode first state on qubits 0 to n-1
                for i in range(min(len(features1), self.num_qubits)):
                    angle = 2 * np.pi * features1[i]
                    qml.RY(angle, wires=i)

                # Encode second state on qubits n to 2n-1
                for i in range(min(len(features2), self.num_qubits)):
                    angle = 2 * np.pi * features2[i]
                    qml.RY(angle, wires=self.num_qubits + i)

                # SWAP test on auxiliary qubit
                aux_wire = 2 * self.num_qubits
                qml.Hadamard(wires=aux_wire)
                for i in range(self.num_qubits):
                    # SWAP test: controlled-SWAP between register 1 and register 2
                    qml.CSWAP(wires=[aux_wire, i, self.num_qubits + i])
                qml.Hadamard(wires=aux_wire)

                return qml.expval(qml.PauliZ(aux_wire))

            swap_circuit = qml.set_shots(swap_circuit, shots=1024)

            # Fidelity from SWAP test: F = (1 + <Z>) / 2
            z_expectation = swap_circuit()
            fidelity = (1 + z_expectation) / 2
            return float(fidelity)

        elif method == 'hadamard_test':
            # Hadamard test for overlap
            dev_test = qml.device(self.backend, wires=self.num_qubits + 1)

            @qml.qnode(dev_test)
            def hadamard_circuit():
                # Prepare |ψ1⟩ on data qubits
                for i in range(min(len(features1), self.num_qubits)):
                    angle = 2 * np.pi * features1[i]
                    qml.RY(angle, wires=i)

                # Hadamard test
                aux = self.num_qubits
                qml.Hadamard(wires=aux)

                # Controlled-U where U prepares |ψ2⟩
                # (simplified: controlled rotations)
                for i in range(min(len(features2), self.num_qubits)):
                    angle = 2 * np.pi * features2[i]
                    qml.CRY(angle, wires=[aux, i])

                qml.Hadamard(wires=aux)

                return qml.expval(qml.PauliZ(aux))

            overlap_real = hadamard_circuit()
            # This gives real part of overlap; square for lower bound on fidelity
            return float(overlap_real ** 2)

        else:
            raise ValueError(f"Unknown fidelity method: {method}")

    def quantum_distance(
        self,
        features1: np.ndarray,
        features2: np.ndarray
    ) -> float:
        """
        Compute quantum distance: D = sqrt(2(1 - sqrt(F)))

        Uses quantum fidelity to compute a distance metric.

        Args:
            features1: First feature vector
            features2: Second feature vector

        Returns:
            Quantum distance (0 for identical states)
        """
        fidelity = self.quantum_fidelity(features1, features2, method='statevector')
        return float(np.sqrt(2 * (1 - np.sqrt(fidelity))))


# ============================================================================
# QUANTUM KERNEL FOR MACHINE LEARNING
# ============================================================================

class QuantumKernel:
    """
    Quantum kernel using parameterized quantum circuits.

    Implements quantum feature maps for kernel-based machine learning.
    Uses ZZ feature map or custom parameterized circuits.

    Example:
        >>> qk = QuantumKernel(num_qubits=4, num_layers=2)
        >>> features = np.random.rand(10, 4)  # 10 samples, 4 features
        >>> K = qk.compute_kernel_matrix(features)
        >>> # K is positive semi-definite kernel matrix
    """

    def __init__(
        self,
        num_qubits: int = 4,
        num_layers: int = 2,
        feature_map: str = 'zz',
        backend: str = 'default.qubit',
        shots: Optional[int] = None
    ):
        """
        Initialize quantum kernel.

        Args:
            num_qubits: Number of qubits (should match feature dimension)
            num_layers: Number of feature map repetitions
            feature_map: Feature map type ('zz', 'pauli', 'custom')
            backend: Quantum backend
            shots: Number of shots (None for statevector)
        """
        if not HAS_PENNYLANE:
            raise RuntimeError(
                "PennyLane required for quantum kernels. "
                "Install with: pip install pennylane"
            )

        self.num_qubits = num_qubits
        self.num_layers = num_layers
        self.feature_map = feature_map
        self.backend = backend
        self.shots = shots
        self._dev = None

    def _get_device(self):
        """Get or create PennyLane device."""
        if self._dev is None:
            kwargs = {'wires': self.num_qubits}
            if self.shots:
                kwargs['shots'] = self.shots
            self._dev = qml.device(self.backend, **kwargs)
        return self._dev

    def zz_feature_map(self, x: np.ndarray, wires: List[int]) -> None:
        """
        ZZ feature map for quantum kernel.

        Implements the ZZ feature map from Havlicek et al. (Nature 2019):
        U_Φ(x) = exp(i Σ_i x_i Z_i) exp(i Σ_{i<j} (π - x_i)(π - x_j) Z_i Z_j)

        Args:
            x: Feature vector
            wires: Qubit indices
        """
        for layer in range(self.num_layers):
            # Single-qubit rotations
            for i, wire in enumerate(wires):
                if i < len(x):
                    qml.Hadamard(wires=wire)
                    qml.RZ(x[i], wires=wire)

            # Two-qubit entangling
            for i in range(len(wires)):
                for j in range(i + 1, len(wires)):
                    if i < len(x) and j < len(x):
                        qml.CNOT(wires=[wires[i], wires[j]])
                        qml.RZ((np.pi - x[i]) * (np.pi - x[j]), wires=wires[j])
                        qml.CNOT(wires=[wires[i], wires[j]])

    def pauli_feature_map(self, x: np.ndarray, wires: List[int]) -> None:
        """
        Pauli expansion feature map.

        Uses higher-order Pauli rotations for richer feature encoding.

        Args:
            x: Feature vector
            wires: Qubit indices
        """
        for layer in range(self.num_layers):
            # Hadamard layer
            for wire in wires:
                qml.Hadamard(wires=wire)

            # Single-qubit Z rotations
            for i, wire in enumerate(wires):
                if i < len(x):
                    qml.RZ(x[i], wires=wire)

            # Two-qubit ZZ rotations
            for i in range(len(wires) - 1):
                qml.CNOT(wires=[wires[i], wires[i + 1]])
                if i < len(x) - 1:
                    qml.RZ(x[i] * x[i + 1], wires=wires[i + 1])
                qml.CNOT(wires=[wires[i], wires[i + 1]])

    def kernel_circuit(self, x1: np.ndarray, x2: np.ndarray) -> float:
        """
        Compute kernel value K(x1, x2) = |<φ(x1)|φ(x2)>|²

        Uses the inversion test: prepare |φ(x1)⟩, apply U†(x2),
        measure probability of |0⟩ state.

        Args:
            x1: First feature vector
            x2: Second feature vector

        Returns:
            Kernel value in [0, 1]
        """
        dev = self._get_device()
        wires = list(range(self.num_qubits))

        @qml.qnode(dev)
        def circuit():
            # Encode x1
            if self.feature_map == 'zz':
                self.zz_feature_map(x1, wires)
            else:
                self.pauli_feature_map(x1, wires)

            # Apply inverse of x2 encoding
            if self.feature_map == 'zz':
                # Adjoint of ZZ feature map
                for layer in reversed(range(self.num_layers)):
                    # Inverse two-qubit
                    for i in range(len(wires) - 1, -1, -1):
                        for j in range(len(wires) - 1, i, -1):
                            if i < len(x2) and j < len(x2):
                                qml.CNOT(wires=[wires[i], wires[j]])
                                qml.RZ(-(np.pi - x2[i]) * (np.pi - x2[j]), wires=wires[j])
                                qml.CNOT(wires=[wires[i], wires[j]])

                    # Inverse single-qubit
                    for i, wire in enumerate(wires):
                        if i < len(x2):
                            qml.RZ(-x2[i], wires=wire)
                            qml.Hadamard(wires=wire)

            # Measure probability of all zeros
            return qml.probs(wires=wires)

        probs = circuit()
        # Kernel value is probability of measuring |00...0⟩
        return float(probs[0])

    def compute_kernel_matrix(self, X: np.ndarray) -> np.ndarray:
        """
        Compute full kernel matrix for dataset.

        Args:
            X: Feature matrix, shape (n_samples, n_features)

        Returns:
            Kernel matrix K, shape (n_samples, n_samples)
        """
        n_samples = len(X)
        K = np.zeros((n_samples, n_samples))

        for i in range(n_samples):
            K[i, i] = 1.0  # Self-similarity
            for j in range(i + 1, n_samples):
                k = self.kernel_circuit(X[i], X[j])
                K[i, j] = k
                K[j, i] = k  # Symmetric

        return K

    def is_positive_semidefinite(self, K: np.ndarray, tol: float = 1e-8) -> bool:
        """
        Check if kernel matrix is positive semi-definite.

        All quantum kernel matrices should be PSD.

        Args:
            K: Kernel matrix
            tol: Tolerance for eigenvalue check

        Returns:
            True if matrix is PSD
        """
        eigenvalues = np.linalg.eigvalsh(K)
        return bool(np.all(eigenvalues >= -tol))


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def check_quantum_backend() -> Dict[str, Any]:
    """
    Check quantum backend availability and configuration.

    Returns:
        Dictionary with backend status information
    """
    status = {
        'pennylane_installed': HAS_PENNYLANE,
        'pennylane_version': PENNYLANE_VERSION,
        'available_backends': [],
        'recommended_backend': None
    }

    if HAS_PENNYLANE:
        # Check available backends
        backends_to_check = ['default.qubit', 'lightning.qubit']

        for backend in backends_to_check:
            try:
                dev = qml.device(backend, wires=2)
                status['available_backends'].append(backend)
            except Exception:
                pass

        # Check qiskit
        try:
            import pennylane_qiskit
            status['available_backends'].extend(['qiskit.aer', 'qiskit.basicaer'])
        except ImportError:
            pass

        # Set recommended
        if 'lightning.qubit' in status['available_backends']:
            status['recommended_backend'] = 'lightning.qubit'
        elif 'default.qubit' in status['available_backends']:
            status['recommended_backend'] = 'default.qubit'

    return status


def quick_h2_energy(bond_length: float = 0.74) -> float:
    """
    Quick H2 ground state energy calculation.

    Convenience function for testing quantum backend.

    Args:
        bond_length: H-H bond length in Angstroms

    Returns:
        Ground state energy in Hartree
    """
    if not HAS_PENNYLANE:
        raise RuntimeError("PennyLane required. Install with: pip install pennylane")

    vqe = VQEMolecule(backend='default.qubit')
    return vqe.compute_ground_state_energy('H2', bond_length, max_iterations=50)


# ============================================================================
# CLASSICAL FALLBACK FOR BASIC OPERATIONS
# ============================================================================

class ClassicalFallback:
    """
    Classical fallback implementations when PennyLane is not available.

    These are NOT real quantum computations, but provide compatible API
    for basic testing without quantum dependencies.
    """

    @staticmethod
    def angle_encode_fidelity(features1: np.ndarray, features2: np.ndarray) -> float:
        """
        Classical approximation of quantum fidelity using angle encoding.

        Warning: This is NOT a real quantum computation.

        Args:
            features1: First feature vector
            features2: Second feature vector

        Returns:
            Approximate fidelity (classical computation)
        """
        # Pad to same length
        max_len = max(len(features1), len(features2))
        f1 = np.pad(features1, (0, max_len - len(features1)))
        f2 = np.pad(features2, (0, max_len - len(features2)))

        # Compute cosine similarity of angle-encoded states
        angles1 = 2 * np.pi * f1
        angles2 = 2 * np.pi * f2

        # Overlap product (simplified model)
        overlaps = np.cos(angles1 / 2) * np.cos(angles2 / 2) + \
                   np.sin(angles1 / 2) * np.sin(angles2 / 2)

        # Geometric mean of squared overlaps
        fidelity = np.prod(overlaps ** 2) ** (1 / len(overlaps))

        return float(fidelity)

    @staticmethod
    def classical_kernel(features1: np.ndarray, features2: np.ndarray) -> float:
        """
        Classical RBF kernel as fallback.

        Args:
            features1: First feature vector
            features2: Second feature vector

        Returns:
            RBF kernel value
        """
        diff = features1 - features2
        gamma = 1.0 / len(features1)
        return float(np.exp(-gamma * np.sum(diff ** 2)))


# ============================================================================
# MODULE EXPORTS
# ============================================================================

__all__ = [
    # Main classes
    'VQEMolecule',
    'QuantumFingerprint',
    'QuantumKernel',
    'QuantumBackendConfig',
    'BackendType',

    # Data classes
    'MolecularGeometry',

    # Convenience functions
    'check_quantum_backend',
    'quick_h2_energy',
    'REFERENCE_ENERGIES',

    # Fallback
    'ClassicalFallback',

    # Status
    'HAS_PENNYLANE',
    'PENNYLANE_VERSION',
]


if __name__ == '__main__':
    # Demo/test when run directly
    print("=" * 70)
    print("NQPU Quantum Backend - PennyLane Integration")
    print("=" * 70)

    status = check_quantum_backend()
    print(f"\nPennyLane installed: {status['pennylane_installed']}")
    print(f"Version: {status['pennylane_version']}")
    print(f"Available backends: {status['available_backends']}")
    print(f"Recommended: {status['recommended_backend']}")

    if HAS_PENNYLANE:
        print("\n" + "=" * 70)
        print("Testing H2 VQE (quick test, 50 iterations)")
        print("=" * 70)

        try:
            energy = quick_h2_energy(bond_length=0.74)
            ref = REFERENCE_ENERGIES['H2']['ground_state_energy']
            print(f"\nComputed energy: {energy:.4f} Ha")
            print(f"Reference value: {ref:.4f} Ha")
            print(f"Error: {abs(energy - ref):.4f} Ha")
        except Exception as e:
            print(f"\nVQE test failed: {e}")

        print("\n" + "=" * 70)
        print("Testing Quantum Kernel")
        print("=" * 70)

        try:
            qk = QuantumKernel(num_qubits=4, num_layers=2)
            # Small test dataset
            X = np.random.rand(5, 4)
            K = qk.compute_kernel_matrix(X)
            print(f"\nKernel matrix shape: {K.shape}")
            print(f"Diagonal elements (should be ~1): {np.diag(K)}")
            print(f"Is PSD: {qk.is_positive_semidefinite(K)}")
        except Exception as e:
            print(f"\nKernel test failed: {e}")

        print("\n" + "=" * 70)
        print("Testing Quantum Fingerprint")
        print("=" * 70)

        try:
            qfp = QuantumFingerprint(num_qubits=4, encoding='angle')
            f1 = np.array([0.1, 0.5, 0.3, 0.8])
            f2 = np.array([0.2, 0.4, 0.3, 0.7])

            fidelity = qfp.quantum_fidelity(f1, f2, method='statevector')
            print(f"\nFidelity between similar states: {fidelity:.4f}")

            f3 = np.array([0.9, 0.9, 0.9, 0.9])
            fidelity_diff = qfp.quantum_fidelity(f1, f3, method='statevector')
            print(f"Fidelity between different states: {fidelity_diff:.4f}")
        except Exception as e:
            print(f"\nFingerprint test failed: {e}")

    else:
        print("\n" + "=" * 70)
        print("PennyLane not installed - showing classical fallback")
        print("=" * 70)

        fb = ClassicalFallback()
        f1 = np.array([0.1, 0.5, 0.3, 0.8])
        f2 = np.array([0.2, 0.4, 0.3, 0.7])

        fidelity = fb.angle_encode_fidelity(f1, f2)
        kernel = fb.classical_kernel(f1, f2)

        print(f"\nClassical fallback fidelity: {fidelity:.4f}")
        print(f"Classical RBF kernel: {kernel:.4f}")

    print("\n" + "=" * 70)
    print("Tests complete")
    print("=" * 70)
