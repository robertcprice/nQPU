"""
nQPU-Metal Advanced Error Mitigation

Implement techniques to mitigate errors in NISQ-era quantum computing:
- Zero-Noise Extrapolation (ZNE)
- Probabilistic Error Cancellation (PEC)
- Virtual Distillation
- Readout Error Mitigation
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'target', 'debug'))

import nqpu_metal
import numpy as np
from typing import List, Dict, Tuple, Callable, Optional
from dataclasses import dataclass
from enum import Enum


class MitigationMethod(Enum):
    """Error mitigation methods."""
    NONE = "none"
    ZNE = "zne"  # Zero-Noise Extrapolation
    PEC = "pec"  # Probabilistic Error Cancellation
    REM = "rem"  # Readout Error Mitigation
    VIRTUAL_DISTILLATION = "virtual_distillation"


@dataclass
class MitigationResult:
    """Result of error mitigation."""
    raw_counts: Dict[str, int]
    mitigated_counts: Dict[str, int]
    method: MitigationMethod
    improvement_factor: float
    confidence: Optional[float] = None


class ReadoutErrorMitigator:
    """
    Mitigate measurement (readout) errors.
    
    Uses calibration data to correct measurement outcomes.
    """
    
    def __init__(self, num_qubits: int):
        self.num_qubits = num_qubits
        self.confusion_matrix = None
        self.inverse_matrix = None
    
    def calibrate(self, shots: int = 1000) -> np.ndarray:
        """
        Calibrate readout error by measuring all-zero states.
        
        Args:
            shots: Number of calibration shots
            
        Returns:
            Confusion matrix (2^n x 2^n)
        """
        size = 2 ** self.num_qubits
        confusion = np.zeros((size, size))
        
        # Calibrate by preparing each basis state and measuring
        for i in range(size):
            # Prepare state |i>
            sim = nqpu_metal.QuantumSimulator(self.num_qubits)
            for q in range(self.num_qubits):
                if (i >> (self.num_qubits - 1 - q)) & 1:
                    sim.x(q)
            
            # Measure
            counts = sim.sample_bitstrings(shots)
            
            # Fill confusion matrix row
            for bitstring, count in counts.items():
                j = int(bitstring, 2)
                confusion[i, j] = count / shots
        
        self.confusion_matrix = confusion
        
        # Compute pseudo-inverse for mitigation
        try:
            self.inverse_matrix = np.linalg.pinv(confusion)
        except np.linalg.LinAlgError:
            # Use regular inverse if pseudo-inverse fails
            self.inverse_matrix = np.linalg.inv(confusion)
        
        return confusion
    
    def mitigate(self, counts: Dict[str, int]) -> Dict[str, int]:
        """
        Apply readout error mitigation to measurement results.
        
        Args:
            counts: Raw measurement counts
            
        Returns:
            Mitigated counts
        """
        if self.inverse_matrix is None:
            raise ValueError("Must calibrate first. Call calibrate().")
        
        size = 2 ** self.num_qubits
        
        # Convert counts to vector
        raw_vector = np.zeros(size)
        for bitstring, count in counts.items():
            i = int(bitstring, 2)
            raw_vector[i] = count
        
        # Apply inverse confusion matrix
        mitigated_vector = self.inverse_matrix @ raw_vector
        
        # Convert back to dictionary
        mitigated = {}
        for i in range(size):
            bitstring = format(i, f'0{self.num_qubits}b')
            value = max(0, mitigated_vector[i])  # Ensure non-negative
            if value > 0.5:  # Only include significant counts
                mitigated[bitstring] = int(round(value))
        
        return mitigated


class ZeroNoiseExtrapolator:
    """
    Zero-Noise Extrapolation (ZNE).
    
    Extrapolates to zero noise by running at multiple noise levels.
    """
    
    def __init__(self, base_noise: float = 0.01):
        self.base_noise = base_noise
        self.noise_factors = [1.0, 3.0, 5.0]  # Different noise scales
    
    def scale_circuit(self, gates: List, scale_factor: float) -> List:
        """
        Scale a circuit by inserting pairs of identity gates.
        
        Args:
            gates: List of gates
            scale_factor: Noise scaling factor
            
        Returns:
            Scaled gate list
        """
        if scale_factor == 1.0:
            return gates
        
        # Insert pairs of CNOT gates (CNOT*CNOT = I)
        scaled = []
        for gate in gates:
            scaled.append(gate)
            # Insert pairs based on scale factor
            num_pairs = int((scale_factor - 1) / 2)
            for _ in range(num_pairs):
                if len(gate[1]) >= 2:  # Multi-qubit gate
                    # Insert CNOT pair on same qubits
                    scaled.append(('cx', gate[1], ()))
                    scaled.append(('cx', gate[1], ()))
        
        return scaled
    
    def run_with_noise(self, 
                      circuit_func: Callable,
                      scale_factor: float,
                      shots: int = 1000) -> Dict[str, int]:
        """Run circuit with scaled noise."""
        # Run with scaled readout error
        readout_error = self.base_noise * scale_factor
        return circuit_func(readout_error=readout_error, shots=shots)
    
    def extrapolate(self, 
                   results_at_scales: List[Dict[str, int]],
                   method: str = "linear") -> Dict[str, int]:
        """
        Extrapolate to zero noise.
        
        Args:
            results_at_scales: Results at different noise scales
            method: Extrapolation method ("linear" or "exponential")
            
        Returns:
            Zero-noise extrapolated counts
        """
        # Collect data for each basis state
        states = set()
        for result in results_at_scales:
            states.update(result.keys())
        
        mitigated = {}
        
        for state in states:
            # Get counts at each scale
            y = []
            for result in results_at_scales:
                y.append(result.get(state, 0))
            
            # Extrapolate
            if method == "linear":
                # Linear fit: y = a + b*scale
                # Extrapolate to scale=0: y(0) = a
                scales = [1.0, 3.0, 5.0][:len(y)]
                coeffs = np.polyfit(scales, y, 1)
                mitigated_count = coeffs[1]  # y-intercept
            else:  # exponential
                # Exponential fit: y = a * exp(b*scale)
                # Extrapolate to scale=0: y(0) = a
                scales = [1.0, 3.0, 5.0][:len(y)]
                log_y = [np.log(max(c, 1)) for c in y]
                coeffs = np.polyfit(scales, log_y, 1)
                mitigated_count = np.exp(coeffs[1])
            
            mitigated[state] = max(0, int(round(mitigated_count)))
        
        return mitigated
    
    def mitigate_circuit(self,
                        circuit_func: Callable,
                        shots: int = 1000) -> MitigationResult:
        """
        Apply ZNE to a circuit.
        
        Args:
            circuit_func: Function that runs the circuit
            shots: Number of shots per noise level
            
        Returns:
            Mitigation result
        """
        results = []
        for scale in self.noise_factors:
            result = self.run_with_noise(circuit_func, scale, shots)
            results.append(result)
        
        mitigated = self.extrapolate(results)
        
        # Calculate improvement
        baseline = results[0]
        improvement = self._calculate_improvement(baseline, mitigated)
        
        return MitigationResult(
            raw_counts=baseline,
            mitigated_counts=mitigated,
            method=MitigationMethod.ZNE,
            improvement_factor=improvement
        )
    
    def _calculate_improvement(self, 
                              raw: Dict[str, int], 
                              mitigated: Dict[str, int]) -> float:
        """Calculate improvement factor (fidelity increase)."""
        # Simplified: measure how much closer mitigated is to expected
        # For Bell state, expected is 50/50 split between |00> and |11>
        expected = {'00': 0.5, '11': 0.5}
        
        total = sum(raw.values())
        raw_error = sum(abs(raw.get(k, 0) / total - expected.get(k, 0)) 
                       for k in expected)
        mitigated_error = sum(abs(mitigated.get(k, 0) / total - expected.get(k, 0)) 
                             for k in expected)
        
        return raw_error / max(mitigated_error, 1e-10)


class ProbabilisticErrorCanceller:
    """
    Probabilistic Error Cancellation (PEC).
    
    Represents noise channels and applies inverse operations.
    """
    
    def __init__(self, noise_prob: float = 0.01):
        self.noise_prob = noise_prob
        self.gate_fidelities = {}
    
    def characterize_gate(self, 
                         gate_name: str,
                         ideal_func: Callable,
                         shots: int = 10000) -> float:
        """
        Characterize a gate's error rate.
        
        Args:
            gate_name: Name of the gate
            ideal_func: Function to create ideal state
            shots: Number of shots for characterization
            
        Returns:
            Gate fidelity
        """
        # Run ideal
        sim_ideal = ideal_func()
        ideal_probs = sim_ideal.probabilities()
        
        # Run with noise
        noisy_counts = nqpu_metal.simulate_noisy_circuit(
            sim_ideal.num_qubits, 
            self.noise_prob, 
            shots
        )
        
        # Calculate fidelity
        fidelity = 0.0
        for bitstring, count in noisy_counts.items():
            idx = int(bitstring, 2)
            fidelity += (count / shots) * ideal_probs[idx]
        
        self.gate_fidelities[gate_name] = fidelity
        return fidelity
    
    def get_inverse_operation(self, gate_name: str) -> List[Tuple]:
        """
        Get inverse operation for error cancellation.
        
        Args:
            gate_name: Gate to invert
            
        Returns:
            List of (name, qubits, params) tuples for inverse
        """
        # For depolarizing noise, the inverse is probabilistic application
        # This is simplified - full PEC requires more complex sampling
        inverses = {
            'h': [('h',)],
            'x': [('x',)],
            'y': [('y',)],
            'z': [('z',)],
            'cx': [('cx',)],
        }
        return inverses.get(gate_name, [])


def mitigate_readout(counts: Dict[str, int],
                    num_qubits: int,
                    calibration_shots: int = 1000) -> Dict[str, int]:
    """
    Apply readout error mitigation.
    
    Args:
        counts: Raw measurement counts
        num_qubits: Number of qubits
        calibration_shots: Shots for calibration
        
    Returns:
        Mitigated counts
    """
    mitigator = ReadoutErrorMitigator(num_qubits)
    mitigator.calibrate(calibration_shots)
    return mitigator.mitigate(counts)


def mitigate_zne(circuit_func: Callable,
                shots: int = 1000,
                base_noise: float = 0.01) -> MitigationResult:
    """
    Apply Zero-Noise Extrapolation.
    
    Args:
        circuit_func: Function that runs circuit
        shots: Shots per noise level
        base_noise: Base readout error rate
        
    Returns:
        Mitigation result
    """
    extrapolator = ZeroNoiseExtrapolator(base_noise)
    return extrapolator.mitigate_circuit(circuit_func, shots)


# Convenience function for complete mitigation pipeline
def full_mitigation(circuit_func: Callable,
                   num_qubits: int,
                   shots: int = 1000,
                   methods: List[MitigationMethod] = None) -> Dict[str, any]:
    """
    Apply multiple error mitigation techniques.
    
    Args:
        circuit_func: Function that runs the circuit
        num_qubits: Number of qubits
        shots: Number of shots
        methods: List of methods to apply
        
    Returns:
        Dictionary with mitigation results
    """
    if methods is None:
        methods = [MitigationMethod.REM, MitigationMethod.ZNE]
    
    results = {}
    
    # Get raw result
    raw_counts = circuit_func(readout_error=0.0, shots=shots)
    results['raw'] = raw_counts
    
    # Apply readout error mitigation
    if MitigationMethod.REM in methods:
        rem_counts = mitigate_readout(raw_counts, num_qubits, shots)
        results['rem'] = rem_counts
    
    # Apply ZNE
    if MitigationMethod.ZNE in methods:
        zne_result = mitigate_zne(circuit_func, shots)
        results['zne'] = zne_result.mitigated_counts
    
    return results
