"""
nQPU-Metal Circuit Optimizer

Provides gate optimization and circuit compilation passes
to reduce gate count and improve simulation performance.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'target', 'debug'))

import nqpu_metal
from typing import List, Tuple, Dict, Optional, Callable
from dataclasses import dataclass
from enum import Enum


class OptimizationLevel(Enum):
    """Optimization levels."""
    NONE = 0
    LOW = 1
    MEDIUM = 2
    HIGH = 3


@dataclass
class Gate:
    """Represent a single gate in a circuit."""
    name: str
    qubits: Tuple[int, ...]
    params: Tuple[float, ...] = ()

    def __repr__(self):
        if self.params:
            return f"{self.name}({','.join(f'{p:.3f}' for p in self.params)}) on {self.qubits}"
        return f"{self.name} on {self.qubits}"

    def __eq__(self, other):
        if not isinstance(other, Gate):
            return False
        return (self.name == other.name and 
                self.qubits == other.qubits and 
                self.params == other.params)


class CircuitOptimizer:
    """
    Optimize quantum circuits to reduce gate count and depth.
    
    Optimizations include:
    - Gate cancellation (X*X = I)
    - Rotation merging (Rz(a)*Rz(b) = Rz(a+b))
    - Hadamard optimization
    - CNOT optimization
    - Single-qubit gate fusion
    """
    
    def __init__(self, optimization_level: OptimizationLevel = OptimizationLevel.MEDIUM):
        self.level = optimization_level
        self.original_gates = []
        self.optimized_gates = []
        self.stats = {}
    
    def optimize_circuit(self, gates: List[Gate]) -> List[Gate]:
        """
        Optimize a list of gates.
        
        Args:
            gates: List of gates to optimize
            
        Returns:
            Optimized list of gates
        """
        self.original_gates = gates.copy()
        self.optimized_gates = gates.copy()
        
        # Apply optimization passes based on level
        if self.level.value >= OptimizationLevel.LOW.value:
            self._cancel_single_qubit_gates()
            self._merge_rotations()
        
        if self.level.value >= OptimizationLevel.MEDIUM.value:
            self._cancel_cnot_pairs()
            self._optimize_hadamards()
            self._remove_identity_gates()
        
        if self.level.value >= OptimizationLevel.HIGH.value:
            self._fuse_single_qubit_gates()
            self._optimize_measurement_gates()
        
        # Calculate statistics
        self._calculate_stats()
        
        return self.optimized_gates
    
    def _cancel_single_qubit_gates(self):
        """Cancel adjacent single-qubit gates (X*X=I, H*H=I, etc.)."""
        cancellations = {
            ('x', 'x'), ('y', 'y'), ('z', 'z'),
            ('h', 'h'), ('s', 's'), ('t', 't'),
        }
        
        new_gates = []
        i = 0
        while i < len(self.optimized_gates):
            if i + 1 < len(self.optimized_gates):
                g1, g2 = self.optimized_gates[i], self.optimized_gates[i + 1]
                # Check if gates operate on same qubit
                if (len(g1.qubits) == 1 and len(g2.qubits) == 1 and
                    g1.qubits == g2.qubits and
                    (g1.name, g2.name) in cancellations):
                    # Cancel both
                    i += 2
                    continue
            new_gates.append(self.optimized_gates[i])
            i += 1
        self.optimized_gates = new_gates
    
    def _merge_rotations(self):
        """Merge consecutive rotation gates on the same qubit and axis."""
        mergeable = {
            'rx', 'ry', 'rz', 'phase',
            'crx', 'cry', 'crz', 'cphase'
        }
        
        new_gates = []
        i = 0
        while i < len(self.optimized_gates):
            gate = self.optimized_gates[i]
            
            if i + 1 < len(self.optimized_gates):
                next_gate = self.optimized_gates[i + 1]
                
                # Check if same rotation gate on same qubits
                if (gate.name == next_gate.name and 
                    gate.name in mergeable and
                    gate.qubits == next_gate.qubits):
                    # Merge angles
                    new_angle = gate.params[0] + next_gate.params[0]
                    # Wrap to [0, 2π)
                    new_angle = new_angle % (2 * 3.14159265359)
                    # Only keep if not approximately zero
                    if abs(new_angle) > 1e-10:
                        new_gates.append(Gate(gate.name, gate.qubits, (new_angle,)))
                    i += 2
                    continue
            
            new_gates.append(gate)
            i += 1
        
        self.optimized_gates = new_gates
    
    def _cancel_cnot_pairs(self):
        """Cancel adjacent CNOT gates on the same qubits."""
        new_gates = []
        i = 0
        while i < len(self.optimized_gates):
            if i + 1 < len(self.optimized_gates):
                g1, g2 = self.optimized_gates[i], self.optimized_gates[i + 1]
                if (g1.name in ['cx', 'cnot'] and g2.name in ['cx', 'cnot'] and
                    g1.qubits == g2.qubits):
                    # Cancel both CNOTs
                    i += 2
                    continue
            new_gates.append(self.optimized_gates[i])
            i += 1
        self.optimized_gates = new_gates
    
    def _optimize_hadamards(self):
        """Optimize H*Z*H = X patterns."""
        # Replace HZH with X
        new_gates = []
        i = 0
        while i < len(self.optimized_gates):
            if i + 2 < len(self.optimized_gates):
                g1, g2, g3 = (self.optimized_gates[i], 
                               self.optimized_gates[i + 1],
                               self.optimized_gates[i + 2])
                if (g1.name == 'h' and g2.name == 'z' and g3.name == 'h' and
                    len(g1.qubits) == 1 and g1.qubits == g2.qubits == g3.qubits):
                    # Replace with X
                    new_gates.append(Gate('x', g1.qubits))
                    i += 3
                    continue
            new_gates.append(self.optimized_gates[i])
            i += 1
        self.optimized_gates = new_gates
    
    def _remove_identity_gates(self):
        """Remove gates that result in identity."""
        # Remove Rz(0), Ry(0), etc.
        new_gates = []
        for gate in self.optimized_gates:
            is_identity = False
            if gate.params:
                if abs(gate.params[0]) < 1e-10:
                    is_identity = True
            if not is_identity:
                new_gates.append(gate)
        self.optimized_gates = new_gates
    
    def _fuse_single_qubit_gates(self):
        """Fuse consecutive single-qubit gates into a single unitary."""
        # For each qubit, find runs of single-qubit gates and fuse them
        by_qubit = {}
        
        for i, gate in enumerate(self.optimized_gates):
            if len(gate.qubits) == 1:
                q = gate.qubits[0]
                if q not in by_qubit:
                    by_qubit[q] = []
                by_qubit[q].append((i, gate))
        
        # For each qubit with multiple single-qubit gates in a row, fuse
        # This is a simplified version - full implementation would compute
        # the actual unitary product
        pass  # Placeholder for full implementation
    
    def _optimize_measurement_gates(self):
        """Remove redundant measurements."""
        # Keep only the last measurement on each qubit
        measured = set()
        new_gates = []
        
        # Process in reverse
        for gate in reversed(self.optimized_gates):
            if gate.name == 'measure':
                qubits = gate.qubits
                # Only keep if not already measured
                if not any(q in measured for q in qubits):
                    new_gates.append(gate)
                    for q in qubits:
                        measured.add(q)
            else:
                new_gates.append(gate)
        
        self.optimized_gates = list(reversed(new_gates))
    
    def _calculate_stats(self):
        """Calculate optimization statistics."""
        self.stats = {
            'original_gates': len(self.original_gates),
            'optimized_gates': len(self.optimized_gates),
            'gates_removed': len(self.original_gates) - len(self.optimized_gates),
            'reduction_percent': 100 * (1 - len(self.optimized_gates) / max(len(self.original_gates), 1)),
        }
    
    def get_stats(self) -> Dict[str, float]:
        """Get optimization statistics."""
        return self.stats
    
    def print_stats(self):
        """Print optimization statistics."""
        print(f"\nCircuit Optimization Statistics:")
        print(f"  Original gates: {self.stats['original_gates']}")
        print(f"  Optimized gates: {self.stats['optimized_gates']}")
        print(f"  Gates removed: {self.stats['gates_removed']}")
        print(f"  Reduction: {self.stats['reduction_percent']:.1f}%")


def optimize_circuit(level: OptimizationLevel = OptimizationLevel.MEDIUM) -> CircuitOptimizer:
    """
    Create a circuit optimizer.
    
    Args:
        level: Optimization level
        
    Returns:
        CircuitOptimizer instance
    """
    return CircuitOptimizer(level)


class CircuitCompiler:
    """
    Compile high-level circuits to nQPU-Metal native gates.
    
    Handles:
    - Gate decomposition (multi-qubit to 1- and 2-qubit gates)
    - Basis gate conversion
    - Qubit mapping
    """
    
    # Supported basis gates
    BASIS_GATES = {
        'h', 'x', 'y', 'z', 's', 't', 'sx',
        'rx', 'ry', 'rz', 'phase',
        'cx', 'cy', 'cz', 'swap', 'cphase',
        'crx', 'cry', 'crz',
        'toffoli', 'measure'
    }
    
    def __init__(self, basis_gates: Optional[set] = None):
        self.basis_gates = basis_gates or self.BASIS_GATES.copy()
    
    def compile_gate(self, gate: Gate) -> List[Gate]:
        """
        Compile a gate to the target basis.
        
        Args:
            gate: Gate to compile
            
        Returns:
            List of basis gates
        """
        # Already in basis
        if gate.name in self.basis_gates:
            return [gate]
        
        # Decompose multi-qubit gates
        decompositions = {
            'mcx': self._decompose_mcx,
            'mcy': self._decompose_mcy,
            'mcz': self._decompose_mcz,
            'swap': self._decompose_swap,
        }
        
        if gate.name in decompositions:
            return decompositions[gate.name](gate)
        
        # Unknown gate - return as-is
        return [gate]
    
    def _decompose_mcx(self, gate: Gate) -> List[Gate]:
        """Decompose multi-controlled X using Toffoli decomposition."""
        controls = list(gate.qubits[:-1])
        target = gate.qubits[-1]
        
        if len(controls) == 2:
            # Already a Toffoli
            return [gate]
        elif len(controls) == 1:
            # CNOT
            return [Gate('cx', (controls[0], target))]
        else:
            # Decompose using Toffoli gates
            # Simplified: use sequential Toffoli with ancilla
            result = []
            n = len(controls)
            
            # This is a simplified decomposition
            # Full implementation would use proper ancilla management
            for i in range(n - 1):
                result.append(Gate('toffoli', (controls[i], controls[i + 1], controls[i + 1])))
            
            result.append(Gate('toffoli', (controls[-2], controls[-1], target)))
            
            # Uncompute
            for i in range(n - 2, -1, -1):
                result.append(Gate('toffoli', (controls[i], controls[i + 1], controls[i + 1])))
            
            return result
    
    def _decompose_mcy(self, gate: Gate) -> List[Gate]:
        """Decompose multi-controlled Y."""
        # Y = S† * X * S (up to global phase)
        controls = list(gate.qubits[:-1])
        target = gate.qubits[-1]
        
        result = []
        # Add S gates around MCX
        result.append(Gate('s', (target,)))
        result.extend(self._decompose_mcx(Gate('mcx', gate.qubits)))
        # Add S† (which is S * Z)
        result.append(Gate('s', (target,)))
        result.append(Gate('z', (target,)))
        
        return result
    
    def _decompose_mcz(self, gate: Gate) -> List[Gate]:
        """Decompose multi-controlled Z."""
        # CZ = H(target) * MCX * H(target)
        controls = list(gate.qubits[:-1])
        target = gate.qubits[-1]
        
        result = []
        result.append(Gate('h', (target,)))
        result.extend(self._decompose_mcx(Gate('mcx', gate.qubits)))
        result.append(Gate('h', (target,)))
        
        return result
    
    def _decompose_swap(self, gate: Gate) -> List[Gate]:
        """Decompose SWAP into CNOTs."""
        q1, q2 = gate.qubits
        return [
            Gate('cx', (q1, q2)),
            Gate('cx', (q2, q1)),
            Gate('cx', (q1, q2)),
        ]


def compile_and_optimize(gates: List[Gate],
                        opt_level: OptimizationLevel = OptimizationLevel.MEDIUM,
                        basis_gates: Optional[set] = None) -> Tuple[List[Gate], Dict]:
    """
    Compile and optimize a circuit.
    
    Args:
        gates: List of gates
        opt_level: Optimization level
        basis_gates: Target basis gate set
        
    Returns:
        Tuple of (optimized gates, statistics)
    """
    # First compile to basis
    compiler = CircuitCompiler(basis_gates)
    compiled = []
    for gate in gates:
        compiled.extend(compiler.compile_gate(gate))
    
    # Then optimize
    optimizer = CircuitOptimizer(opt_level)
    optimized = optimizer.optimize_circuit(compiled)
    
    stats = {
        'original': len(gates),
        'compiled': len(compiled),
        'optimized': len(optimized),
        'total_reduction': 100 * (1 - len(optimized) / max(len(gates), 1)),
    }
    stats.update(optimizer.get_stats())
    
    return optimized, stats


# Convenience functions
def optimize_gates(gate_list: list, level: int = 2) -> list:
    """
    Optimize a list of gates.
    
    Args:
        gate_list: List of (name, qubits, params) tuples
        level: Optimization level (0-3)
        
    Returns:
        Optimized list of gates
    """
    opt_level = OptimizationLevel(min(level, 3))
    optimizer = CircuitOptimizer(opt_level)
    
    # Convert to Gate objects
    gates = []
    for item in gate_list:
        if isinstance(item, Gate):
            gates.append(item)
        elif isinstance(item, tuple):
            name = item[0]
            qubits = tuple(item[1]) if isinstance(item[1], list) else item[1]
            params = item[2] if len(item) > 2 else ()
            gates.append(Gate(name, qubits, params))
    
    optimized = optimizer.optimize_circuit(gates)
    return optimized


def gate_count_reduction(original: list, optimized: list) -> float:
    """Calculate gate count reduction percentage."""
    return 100 * (1 - len(optimized) / max(len(original), 1))
