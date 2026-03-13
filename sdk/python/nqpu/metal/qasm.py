"""
nQPU-Metal QASM (Quantum Assembly) Transcription

Import/export circuits in OpenQASM 2.0 and 3.0 formats.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'target', 'debug'))

import nqpu_metal
import re
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass


@dataclass
class QASMGate:
    """Represent a gate in QASM format."""
    name: str
    qubits: List[str]
    params: List[float] = None
    
    def __str__(self):
        if self.params:
            params_str = ', '.join(f'{p:.10f}' for p in self.params)
            return f"{self.name}({params_str}) {', '.join(self.qubits)}"
        return f"{self.name} {', '.join(self.qubits)}"


class QASMImporter:
    """Import QASM circuits into nQPU-Metal."""
    
    def __init__(self):
        self.qubit_map = {}
        self.clbit_map = {}
        self.qubit_counter = 0
        self.clbit_counter = 0
    
    def import_qasm(self, qasm_str: str) -> nqpu_metal.QuantumSimulator:
        """
        Import a QASM string and create a simulator.
        
        Args:
            qasm_str: QASM circuit string
            
        Returns:
            QuantumSimulator with the circuit applied
        """
        # Parse header
        header_match = re.search(r'OPENQASM\s+(\d+\.\d+);', qasm_str)
        if header_match:
            version = header_match.group(1)
        
        # Find qreg and creg declarations
        qreg_matches = re.findall(r'qreg\s+(\w+)\[(\d+)\];', qasm_str)
        creg_matches = re.findall(r'creg\s+(\w+)\[(\d+)\];', qasm_str)
        
        # Count total qubits
        total_qubits = sum(int(m[1]) for m in qreg_matches)
        
        # Create simulator
        sim = nqpu_metal.QuantumSimulator(total_qubits)
        
        # Build qubit map
        qubit_idx = 0
        for name, size in qreg_matches:
            for i in range(int(size)):
                self.qubit_map[f"{name}[{i}]"] = qubit_idx
                qubit_idx += 1
        
        # Parse gates
        gate_pattern = r'(\w+)(?:\(([^)]+)\))?\s+([^;]+);'
        for match in re.finditer(gate_pattern, qasm_str):
            gate_name = match.group(1)
            params_str = match.group(2)
            args_str = match.group(3)
            
            # Parse parameters
            params = []
            if params_str:
                params = [float(p.strip()) for p in params_str.split(',')]
            
            # Parse qubit arguments
            args = [a.strip() for a in args_str.split(',')]
            qubits = []
            for arg in args:
                if arg in self.qubit_map:
                    qubits.append(self.qubit_map[arg])
                elif '[' in arg:
                    # Direct index like q[0]
                    match = re.match(r'(\w+)\[(\d+)\]', arg)
                    if match:
                        name = match.group(1)
                        idx = int(match.group(2))
                        key = f"{name}[{idx}]"
                        if key in self.qubit_map:
                            qubits.append(self.qubit_map[key])
            
            # Apply gate
            self._apply_gate(sim, gate_name, qubits, params)
        
        return sim
    
    def _apply_gate(self, sim, name: str, qubits: List[int], params: List[float]):
        """Apply a parsed gate to the simulator."""
        # Single qubit gates
        if name == 'h' or name == 'H':
            sim.h(qubits[0])
        elif name == 'x' or name == 'X':
            sim.x(qubits[0])
        elif name == 'y' or name == 'Y':
            sim.y(qubits[0])
        elif name == 'z' or name == 'Z':
            sim.z(qubits[0])
        elif name == 's' or name == 'S':
            sim.s(qubits[0])
        elif name == 'sdg':
            sim.s(qubits[0])
            sim.z(qubits[0])
        elif name == 't' or name == 'T':
            sim.t(qubits[0])
        elif name == 'tdg':
            sim.t(qubits[0])
            sim.z(qubits[0])
        elif name == 'rx':
            if params:
                sim.rx(qubits[0], params[0])
        elif name == 'ry':
            if params:
                sim.ry(qubits[0], params[0])
        elif name == 'rz':
            if params:
                sim.rz(qubits[0], params[0])
        elif name == 'u1':
            if params:
                sim.rz(qubits[0], params[0])
        elif name == 'u2':
            if len(params) >= 2:
                sim.rz(qubits[0], params[0])
                sim.ry(qubits[0], 3.14159/2)
                sim.rz(qubits[0], params[1])
        elif name == 'u3':
            if len(params) >= 3:
                sim.rz(qubits[0], params[1])
                sim.ry(qubits[0], params[0])
                sim.rz(qubits[0], params[2])
        # Two qubit gates
        elif name == 'cx' or name == 'CX' or name == 'CNOT':
            sim.cx(qubits[0], qubits[1])
        elif name == 'cy':
            sim.cy(qubits[0], qubits[1])
        elif name == 'cz':
            sim.cz(qubits[0], qubits[1])
        elif name == 'swap' or name == 'SWAP':
            sim.swap(qubits[0], qubits[1])
        elif name == 'crx':
            if params and len(qubits) >= 2:
                sim.crx(qubits[0], qubits[1], params[0])
        elif name == 'cry':
            if params and len(qubits) >= 2:
                sim.cry(qubits[0], qubits[1], params[0])
        elif name == 'crz':
            if params and len(qubits) >= 2:
                sim.crz(qubits[0], qubits[1], params[0])
        elif name == 'cp' or name == 'cu1':
            if params and len(qubits) >= 2:
                sim.cphase(qubits[0], qubits[1], params[0])
        elif name == 'ccx' or name == 'CCX':
            if len(qubits) >= 3:
                sim.toffoli(qubits[0], qubits[1], qubits[2])
        # Measurement
        elif name == 'measure' or name == 'measure':
            # Handled separately in full QASM
            pass
        else:
            # Unknown gate - try as-is
            pass


class QASMExporter:
    """Export nQPU-Metal circuits to QASM format."""
    
    def __init__(self, version: str = "2.0"):
        self.version = version
    
    def export_simulator(self, sim: nqpu_metal.QuantumSimulator,
                         qreg_name: str = "q",
                         include_header: bool = True) -> str:
        """
        Export a simulator's circuit to QASM.
        
        Note: This exports the circuit that was APPLIED to create the state.
        For full circuit export, you need to track gates separately.
        
        Args:
            sim: Quantum simulator
            qreg_name: Quantum register name
            include_header: Include QASM header
            
        Returns:
            QASM string
        """
        n_qubits = sim.num_qubits
        
        lines = []
        if include_header:
            lines.append(f'OPENQASM {self.version};')
            lines.append(f'include "qelib1.inc";')
            lines.append(f'qreg {qreg_name}[{n_qubits}];')
            lines.append(f'creg c[{n_qubits}];')
        
        return '\n'.join(lines)
    
    def export_gates(self, gates: List[Tuple], 
                     n_qubits: int,
                     qreg_name: str = "q") -> str:
        """
        Export a list of gates to QASM.
        
        Args:
            gates: List of (name, qubits, params) tuples
            n_qubits: Total number of qubits
            qreg_name: Quantum register name
            
        Returns:
            QASM string
        """
        lines = [
            f'OPENQASM {self.version};',
            f'include "qelib1.inc";',
            f'qreg {qreg_name}[{n_qubits}];',
            f'creg c[{n_qubits}];',
        ]
        
        for gate in gates:
            if isinstance(gate, tuple):
                name = gate[0]
                qubits = gate[1]
                params = gate[2] if len(gate) > 2 else None
                
                # Format qubit arguments
                if isinstance(qubits, (list, tuple)):
                    qargs = ', '.join(f'{qreg_name}[{q}]' for q in qubits)
                else:
                    qargs = f'{qreg_name}[{qubits}]'
                
                # Format gate
                if params:
                    params_str = ', '.join(f'{p}' for p in params)
                    lines.append(f'{name}({params_str}) {qargs};')
                else:
                    lines.append(f'{name} {qargs};')
        
        return '\n'.join(lines)


def from_qasm(qasm_str: str) -> nqpu_metal.QuantumSimulator:
    """
    Import a QASM circuit.
    
    Args:
        qasm_str: QASM circuit string
        
    Returns:
        QuantumSimulator with the circuit applied
        
    Example:
        qasm = '''
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[2];
        h q[0];
        cx q[0],q[1];
        '''
        sim = from_qasm(qasm)
    """
    importer = QASMImporter()
    return importer.import_qasm(qasm_str)


def to_qasm(sim: nqpu_metal.QuantumSimulator,
            gates: List[tuple] = None,
            n_qubits: int = None) -> str:
    """
    Export to QASM format.
    
    Args:
        sim: Quantum simulator
        gates: Optional list of gates to export
        n_qubits: Number of qubits (defaults to sim.num_qubits)
        
    Returns:
        QASM string
    """
    exporter = QASMExporter()
    
    if n_qubits is None:
        n_qubits = sim.num_qubits
    
    if gates:
        return exporter.export_gates(gates, n_qubits)
    else:
        return exporter.export_simulator(sim)


# Convenience function
def load_qasm(filename: str) -> nqpu_metal.QuantumSimulator:
    """Load a QASM file."""
    with open(filename, 'r') as f:
        qasm_str = f.read()
    return from_qasm(qasm_str)


def save_qasm(sim: nqpu_metal.QuantumSimulator,
              filename: str,
              gates: List[tuple] = None):
    """Save a circuit to a QASM file."""
    qasm_str = to_qasm(sim, gates)
    with open(filename, 'w') as f:
        f.write(qasm_str)
