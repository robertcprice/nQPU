"""
nQPU-Metal Qiskit Compatibility Layer

Provides a Qiskit backend interface for nQPU-Metal.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Union
import numpy as np

# Try to import Qiskit
try:
    from qiskit.providers import BackendV2, Options
    from qiskit.providers.models import BackendConfiguration
    from qiskit.transpiler import Target
    from qiskit.circuit import QuantumCircuit, Parameter
    from qiskit.result import Result
    from qiskit import QuantumCircuit as QiskitCircuit
    QISKIT_AVAILABLE = True
except ImportError:
    QISKIT_AVAILABLE = False
    BackendV2 = object
    Options = object
    BackendConfiguration = object
    Target = object
    QuantumCircuit = object
    Parameter = object
    Result = object
    QiskitCircuit = object

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'target', 'debug'))

# Try to import the core module
try:
    import nqpu_metal as _nqpu
except ImportError:
    # Try relative import
    try:
        from . import nqpu_metal as _nqpu
    except ImportError:
        _nqpu = None


class NQPUBackend:
    """
    Qiskit backend for nQPU-Metal simulator.
    
    Usage:
        from qiskit import QuantumCircuit
        from nqpu_metal.qiskit_compat import NQPUBackend
        
        backend = NQPUBackend()
        circuit = QuantumCircuit(2)
        circuit.h(0)
        circuit.cx(0, 1)
        circuit.measure_all()
        
        job = backend.run(circuit, shots=1000)
        result = job.result()
        counts = result.get_counts()
    """
    
    _DEFAULT_OPTIONS = {
        'shots': 1024,
        'max_qubits': 24,
        'use_mps': False,
        'mps_bond_dim': 16,
        'readout_error': 0.0,
    }
    
    def __init__(self, **options):
        """Initialize the nQPU-Metal backend."""
        if _nqpu is None:
            raise ImportError("nQPU-Metal core not available. Build with: maturin develop")
        
        if not QISKIT_AVAILABLE:
            raise ImportError("Qiskit is not installed. Install with: pip install qiskit")
        
        self._name = 'nqpu_metal_simulator'
        self._options = {**self._DEFAULT_OPTIONS, **options}
        
    @property
    def name(self) -> str:
        return self._name
    
    @property
    def options(self):
        return type('Options', (), self._options)()
    
    def run(self, circuits: Union[QiskitCircuit, List[QiskitCircuit]], 
            shots: Optional[int] = None, **kwargs) -> 'NQPUJob':
        """Run circuits on nQPU-Metal."""
        if not isinstance(circuits, list):
            circuits = [circuits]
        if shots is None:
            shots = self._options.get('shots', 1024)
        return NQPUJob(self, circuits, shots, **kwargs)


class NQPUJob:
    """Job object for nQPU-Metal."""
    
    def __init__(self, backend: NQPUBackend, circuits: List, shots: int, **kwargs):
        self._backend = backend
        self._circuits = circuits
        self._shots = shots
        self._result = None
        
    def submit(self):
        """Submit and run the job."""
        results = []
        for circuit in self._circuits:
            results.append(self._execute_circuit(circuit))
        
        self._result = type('Result', (), {
            'get_counts': lambda i: results[i] if i < len(results) else {},
            'results': [{'data': {'counts': r}} for r in results]
        })()
        return self
    
    def _execute_circuit(self, circuit: QiskitCircuit) -> Dict[str, int]:
        """Execute a Qiskit circuit."""
        num_qubits = circuit.num_qubits
        use_mps = self._backend._options.get('use_mps', False)
        mps_bond_dim = self._backend._options.get('mps_bond_dim', 16)
        
        # Choose simulator
        if num_qubits > 24 or use_mps:
            sim = _nqpu.MPSSimulator(num_qubits, max_bond_dim=mps_bond_dim)
        else:
            sim = _nqpu.QuantumSimulator(num_qubits)
        
        # Execute gates
        for inst, qargs, cargs in circuit.data:
            if inst.name in ['barrier', 'measure', 'reset']:
                continue
            qubit_indices = [q.index for q in qargs]
            self._apply_gate(sim, inst, qubit_indices)
        
        # Measure
        readout_error = self._backend._options.get('readout_error', 0.0)
        if readout_error > 0:
            samples = _nqpu.simulate_noisy_circuit(num_qubits, readout_error, self._shots)
        else:
            samples = sim.sample_bitstrings(self._shots)
        
        # Reverse bitstrings for Qiskit compatibility
        counts = {}
        for bitstring, count in samples.items():
            reversed_bits = bitstring[::-1]
            counts[reversed_bits] = counts.get(reversed_bits, 0) + count
        
        return counts
    
    def _apply_gate(self, sim, inst, qubit_indices):
        """Apply a Qiskit gate to nQPU-Metal simulator."""
        name = inst.name
        params = inst.params if hasattr(inst, 'params') else []
        
        if name in ['h', 'hadamard']:
            sim.h(qubit_indices[0])
        elif name in ['x', 'not']:
            sim.x(qubit_indices[0])
        elif name == 'y':
            sim.y(qubit_indices[0])
        elif name == 'z':
            sim.z(qubit_indices[0])
        elif name == 's':
            sim.s(qubit_indices[0])
        elif name == 'sdg':
            sim.s(qubit_indices[0])
            sim.z(qubit_indices[0])
        elif name == 't':
            sim.t(qubit_indices[0])
        elif name == 'sx':
            sim.sx(qubit_indices[0])
        elif name in ['rx', 'rx_gate']:
            theta = float(params[0]) if params else 0.0
            sim.rx(qubit_indices[0], theta)
        elif name in ['ry', 'ry_gate']:
            theta = float(params[0]) if params else 0.0
            sim.ry(qubit_indices[0], theta)
        elif name in ['rz', 'rz_gate', 'u1']:
            theta = float(params[0]) if params else 0.0
            sim.rz(qubit_indices[0], theta)
        elif name in ['cx', 'cnot']:
            sim.cx(qubit_indices[0], qubit_indices[1])
        elif name == 'cy':
            sim.cy(qubit_indices[0], qubit_indices[1])
        elif name == 'cz':
            sim.cz(qubit_indices[0], qubit_indices[1])
        elif name == 'swap':
            sim.swap(qubit_indices[0], qubit_indices[1])
        elif name in ['crx', 'crx_gate']:
            theta = float(params[0]) if params else 0.0
            sim.crx(qubit_indices[0], qubit_indices[1], theta)
        elif name in ['cry', 'cry_gate']:
            theta = float(params[0]) if params else 0.0
            sim.cry(qubit_indices[0], qubit_indices[1], theta)
        elif name in ['crz', 'crz_gate']:
            theta = float(params[0]) if params else 0.0
            sim.crz(qubit_indices[0], qubit_indices[1], theta)
        elif name in ['cp', 'cp_gate', 'cu1']:
            theta = float(params[0]) if params else 0.0
            sim.cphase(qubit_indices[0], qubit_indices[1], theta)
        elif name in ['ccx', 'toffoli']:
            sim.toffoli(qubit_indices[0], qubit_indices[1], qubit_indices[2])
        elif name == 'u2':
            phi = float(params[0]) if params else 0.0
            lam = float(params[1]) if len(params) > 1 else 0.0
            sim.rz(qubit_indices[0], phi)
            sim.ry(qubit_indices[0], np.pi/2)
            sim.rz(qubit_indices[0], lam)
        elif name == 'u3':
            theta = float(params[0]) if params else 0.0
            phi = float(params[1]) if len(params) > 1 else 0.0
            lam = float(params[2]) if len(params) > 2 else 0.0
            sim.rz(qubit_indices[0], phi)
            sim.ry(qubit_indices[0], theta)
            sim.rz(qubit_indices[0], lam)
    
    def result(self):
        """Get the result."""
        if self._result is None:
            self.submit()
        return self._result
    
    def status(self):
        """Get job status."""
        return 'DONE' if self._result else 'RUNNING'


def qiskit_to_nqpu(circuit: QiskitCircuit, shots: int = 1024, 
                   use_mps: bool = False) -> Dict[str, int]:
    """Convert Qiskit circuit to nQPU-Metal results."""
    backend = NQPUBackend(use_mps=use_mps)
    job = backend.run(circuit, shots=shots)
    result = job.result()
    return result.get_counts(0)


def get_backend(**options) -> NQPUBackend:
    """Get nQPU-Metal backend."""
    return NQPUBackend(**options)
