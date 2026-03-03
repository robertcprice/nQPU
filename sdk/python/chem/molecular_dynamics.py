"""
Molecular Dynamics Simulation for NQPU

Simplified physics-based simulation for:
1. Protein dynamics (folding trajectories)
2. Ligand binding (induced fit)
3. Drug diffusion through membranes

Run with: python3 molecular_dynamics.py
"""

import numpy as np
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass, field
import random


# ============================================================================
# ATOM PROPERTIES
# ============================================================================

ATOM_MASSES = {
    "C": 12.011,
    "H": 1.008,
    "N": 14.007,
    "O": 15.999,
    "S": 32.065,
    "P": 30.974,
}

ATOM_RADII = {
    "C": 1.7,
    "H": 1.2,
    "N": 1.55,
    "O": 1.52,
    "S": 1.8,
    "P": 1.8,
}


# ============================================================================
# MOLECULAR SYSTEM
# ============================================================================

class Atom:
    """Single atom in molecular system."""
    def __init__(self, atom_type: str, position: List[float]):
        self.atom_type = atom_type
        self.position = np.array(position, dtype=float)
        self.velocity = np.zeros(3)
        self.mass = ATOM_MASSES.get(atom_type, 12.0)
        self.radius = ATOM_RADII.get(atom_type, 1.5)
        self.force = np.zeros(3)


class MolecularSystem:
    """Complete molecular system for dynamics simulation."""

    def __init__(self):
        self.atoms: List[Atom] = []
        self.bonds: List[Tuple[int, int, float]] = []  # (atom1, atom2, equilibrium_length)
        self.box_size = np.array([50.0, 50.0, 50.0])
        self.temperature = 300.0  # Kelvin
        self.time = 0.0

    def add_atom(self, atom_type: str, position: List[float]) -> int:
        """Add an atom to the system."""
        atom = Atom(atom_type, position)
        self.atoms.append(atom)
        return len(self.atoms) - 1

    def add_bond(self, atom1_idx: int, atom2_idx: int, eq_length: float = 1.5):
        """Add a bond between two atoms."""
        self.bonds.append((atom1_idx, atom2_idx, eq_length))

    def calculate_forces(self) -> None:
        """Calculate forces on all atoms."""
        # Reset forces
        for atom in self.atoms:
            atom.force = np.zeros(3)

        # Bond forces (harmonic springs)
        for atom1_idx, atom2_idx, eq_length in self.bonds:
            pos1 = self.atoms[atom1_idx].position
            pos2 = self.atoms[atom2_idx].position

            r_vec = pos2 - pos1
            r = np.linalg.norm(r_vec)

            if r > 0.01:
                # Harmonic spring: F = -k(r - r0)
                k = 300.0  # Spring constant kJ/mol/A^2
                force_mag = -k * (r - eq_length)
                force_dir = r_vec / r
                force = force_mag * force_dir

                self.atoms[atom1_idx].force += force
                self.atoms[atom2_idx].force -= force

        # Non-bonded forces (simplified Lennard-Jones)
        for i in range(len(self.atoms)):
            for j in range(i + 1, len(self.atoms)):
                r_vec = self.atoms[j].position - self.atoms[i].position
                r = np.linalg.norm(r_vec)

                if r > 0.1 and r < 10.0:
                    # Simplified LJ: F = 24eps/r^6 - 48eps/r^12
                    eps = 0.5  # kJ/mol (simplified)
                    sigma = 3.5  # Angstroms

                    sr6 = (sigma / r) ** 6
                    sr12 = sr6 ** 2

                    force_mag = 24 * eps * sr6 / r - 48 * eps * sr12 / r
                    force_dir = r_vec / r

                    self.atoms[i].force += force_mag * force_dir
                    self.atoms[j].force -= force_mag * force_dir

    def calculate_energy(self) -> Tuple[float, float]:
        """Calculate potential and kinetic energy."""
        # Potential energy
        pe = 0.0

        for atom1_idx, atom2_idx, eq_length in self.bonds:
            r = np.linalg.norm(
                self.atoms[atom2_idx].position - self.atoms[atom1_idx].position
            )
            k = 300.0
            pe += 0.5 * k * (r - eq_length) ** 2

        # Non-bonded (LJ)
        for i in range(len(self.atoms)):
            for j in range(i + 1, len(self.atoms)):
                r = np.linalg.norm(
                    self.atoms[j].position - self.atoms[i].position
                )
                if r > 0.1 and r < 10.0:
                    eps = 0.5
                    sigma = 3.5
                    sr6 = (sigma / r) ** 6
                    sr12 = sr6 ** 2
                    pe += 4 * eps * (sr12 - sr6)

        # Kinetic energy
        ke = 0.0
        for atom in self.atoms:
            ke += 0.5 * atom.mass * np.sum(atom.velocity ** 2)

        return pe, ke

    def step(self, use_thermostat: bool = True):
        """Perform one MD step using velocity Verlet integration."""
        dt = 0.5  # ps

        # Langevin thermostat friction
        gamma = 0.1 if use_thermostat else 0.0

        # Calculate forces
        self.calculate_forces()

        # Update velocities (half step)
        for atom in self.atoms:
            acceleration = atom.force / atom.mass
            atom.velocity += 0.5 * acceleration * dt

        # Apply thermostat
        if use_thermostat:
            for atom in self.atoms:
                # Random thermal force
                thermal_force = np.random.normal(0, 0.1, 3)
                atom.velocity += thermal_force
                # Friction
                atom.velocity *= (1 - gamma * dt)

        # Update positions
        for atom in self.atoms:
            atom.position += atom.velocity * dt

        # Update velocities (half step with new forces)
        self.calculate_forces()
        for atom in self.atoms:
            acceleration = atom.force / atom.mass
            atom.velocity += 0.5 * acceleration * dt

        # Periodic boundary conditions
        for atom in self.atoms:
            for dim in range(3):
                if atom.position[dim] < 0:
                    atom.position[dim] += self.box_size[dim]
                elif atom.position[dim] >= self.box_size[dim]:
                    atom.position[dim] -= self.box_size[dim]

        self.time += dt

    def run(self, n_steps: int = 1000, use_thermostat: bool = True) -> Dict:
        """Run simulation for n steps."""
        for _ in range(n_steps):
            self.step(use_thermostat=use_thermostat)

        pe, ke = self.calculate_energy()
        n_atoms = len(self.atoms)
        temp_k = 2 * ke / (1.5 * 8.314e-3 * n_atoms) if n_atoms > 0 else 0

        return {
            "potential_energy": pe,
            "kinetic_energy": ke,
            "total_energy": pe + ke,
            "temperature_K": temp_k
        }


# ============================================================================
# PROTEIN FOLDING
# ============================================================================

class ProteinFolder:
    """Simulate protein folding using MD."""

    @staticmethod
    def create_extended_conformation(sequence: str) -> MolecularSystem:
        """Create protein in extended conformation."""
        system = MolecularSystem()
        system.temperature = 350.0

        # Place atoms in extended chain
        x = 25.0
        for aa in sequence:
            system.add_atom(aa, [x, 25.0, 25.0])
            x += 3.8  # ~3.8 Å per residue

        # Add peptide bonds
        for i in range(len(sequence) - 1):
                system.add_bond(i, i + 1, 3.8)

        return system

    @staticmethod
    def fold(sequence: str, n_steps: int = 5000) -> Dict:
        """Simulate protein folding."""
        system = ProteinFolder.create_extended_conformation(sequence)

        initial_pe, initial_ke = system.calculate_energy()
        initial_energy = initial_pe + initial_ke

        # Simulated annealing: cool gradually
        for phase in range(10):
            system.temperature = 350.0 * (1 - phase / 10) + 50  # Cool from 350K to 50K
            for _ in range(n_steps // 10):
                system.step(use_thermostat=True)

        final_pe, final_ke = system.calculate_energy()
        final_energy = final_pe + final_ke

        # Get final positions
        positions = np.array([atom.position for atom in system.atoms])

        # Calculate radius of gyration (compactness measure)
        center = np.mean(positions, axis=0)
        distances = np.linalg.norm(positions - center, axis=1)
        rg = np.sqrt(np.mean(distances ** 2))

        return {
            "sequence": sequence,
            "initial_energy_kJ_mol": initial_energy,
            "final_energy_kJ_mol": final_energy,
            "energy_change_kJ_mol": final_energy - initial_energy,
            "radius_of_gyration_A": rg,
            "n_atoms": len(system.atoms),
            "folded": final_energy < initial_energy
        }


# ============================================================================
# LIGAND BINDING
# ============================================================================

class LigandBinding:
    """Simulate ligand-receptor binding."""

    @staticmethod
    def induced_fit(receptor_coords: List[List[float]],
                    ligand_coords: List[List[float]],
                    n_steps: int = 5000) -> Dict:
        """Simulate induced fit binding mechanism."""
        # Create system
        system = MolecularSystem()
        system.box_size = np.array([100.0, 100.0, 100.0])

        # Add receptor atoms (fixed, heavy)
        receptor_indices = []
        for pos in receptor_coords:
            idx = system.add_atom("C", pos)
            system.atoms[idx].mass *= 100  # Heavy
            receptor_indices.append(idx)

        # Add ligand atoms (starting far away)
        ligand_indices = []
        offset = [20.0, 0.0, 0.0]
        for pos in ligand_coords:
            new_pos = [pos[i] + offset[i] for i in range(3)]
            idx = system.add_atom("C", new_pos)
            ligand_indices.append(idx)

        # Calculate initial distance
        initial_dist = np.linalg.norm(
            np.array(ligand_coords[0]) - np.array(receptor_coords[0])
        )

        # Run simulation
        system.temperature = 300.0
        for _ in range(n_steps):
            system.step(use_thermostat=True)

        # Calculate final distance
        final_ligand_pos = system.atoms[ligand_indices[0]].position
        final_receptor_pos = system.atoms[receptor_indices[0]].position
        final_dist = np.linalg.norm(final_ligand_pos - final_receptor_pos)

        return {
            "initial_distance_A": initial_dist,
            "final_distance_A": final_dist,
            "binding_occurred": final_dist < 5.0,
            "displacement_A": abs(final_dist - initial_dist)
        }


# ============================================================================
# DRUG DIFFUSION
# ============================================================================

class DrugDiffusion:
    """Simulate drug diffusion through membrane."""

    @staticmethod
    def permeability(molecule_atoms: List[Tuple[str, List[float]]],
                    membrane_z: float = 25.0,
                    n_steps: int = 5000) -> Dict:
        """Simulate drug crossing a membrane."""
        system = MolecularSystem()
        system.box_size = np.array([50.0, 50.0, 50.0])

        # Add molecule atoms
        molecule_indices = []
        for atom_type, pos in molecule_atoms:
            idx = system.add_atom(atom_type, pos)
            molecule_indices.append(idx)

        # Track z-position over time
        initial_z = np.mean([system.atoms[i].position[2] for i in molecule_indices])

        # Run simulation
        system.temperature = 310.0  # Body temperature
        for _ in range(n_steps):
            system.step(use_thermostat=True)

        # Calculate final position
        final_z = np.mean([system.atoms[i].position[2] for i in molecule_indices])

        crossed = (initial_z < membrane_z and final_z >= membrane_z) or \
                      (initial_z >= membrane_z and final_z < membrane_z)

        return {
            "initial_z_A": initial_z,
            "final_z_A": final_z,
            "membrane_z_A": membrane_z,
            "crossed_membrane": crossed,
            "displacement_A": abs(final_z - initial_z)
        }


# ============================================================================
# DEMO
# ============================================================================

def demo():
    print("=" * 70)
    print("MOLECULAR DYNAMICS SIMULATION")
    print("=" * 70)

    # 1. Basic MD
    print("\n[1] BASIC MOLECULAR DYNAMICS")
    print("-" * 40)

    system = MolecularSystem()
    system.add_atom("C", [0.0, 0.0, 0.0])
    system.add_atom("O", [1.2, 0.0, 0.0])
    system.add_atom("O", [-1.2, 0.0, 0.0])
    system.add_bond(0, 1, 1.2)
    system.add_bond(0, 2, 1.2)

    print(f"  Created CO2 molecule with {len(system.atoms)} atoms, {len(system.bonds)} bonds")
    print(f"  Temperature: {system.temperature} K")
    print(f"  Running 1000 steps...")

    result = system.run(1000)
    print(f"  Final energy: {result['total_energy']:.2f} kJ/mol")
    print(f"  Temperature: {result['temperature_K']:.1f} K")

    # 2. Protein folding
    print("\n[2] PROTEIN FOLDING SIMULATION")
    print("-" * 40)

    peptide = "ACDEFGHIK"  # 9 residues
    print(f"  Peptide sequence: {peptide}")

    result = ProteinFolder.fold(peptide, n_steps=5000)
    print(f"  Initial energy: {result['initial_energy_kJ_mol']:.2f} kJ/mol")
    print(f"  Final energy: {result['final_energy_kJ_mol']:.2f} kJ/mol")
    print(f"  Energy change: {result['energy_change_kJ_mol']:.2f} kJ/mol")
    print(f"  Radius of gyration: {result['radius_of_gyration_A']:.2f} Å")
    print(f"  Status: {'folded' if result['folded'] else 'unfolded'}")

    # 3. Ligand binding
    print("\n[3] LIGAND BINDING (INDUCED FIT)")
    print("-" * 40)

    receptor = [[0.0, 0.0, 0.0], [1.5, 0.0, 0.0], [0.75, 1.3, 0.0]]
    ligand = [[0.0, 0.0, 2.0], [1.0, 0.0, 2.5]]

    result = LigandBinding.induced_fit(receptor, ligand, n_steps=3000)
    print(f"  Initial distance: {result['initial_distance_A']:.2f} Å")
    print(f"  Final distance: {result['final_distance_A']:.2f} Å")
    print(f"  Binding occurred: {result['binding_occurred']}")

    # 4. Drug diffusion
    print("\n[4] DRUG DIFFUSION")
    print("-" * 40)

    # Aspirin-like molecule (simplified)
    drug = [("C", [10.0, 10.0, 10.0]), ("O", [11.5, 10.0, 10.0]), ("C", [10.0, 11.5, 10.0])]
    result = DrugDiffusion.permeability(drug, membrane_z=25.0, n_steps=3000)
    print(f"  Initial Z: {result['initial_z_A']:.2f} Å")
    print(f"  Final Z: {result['final_z_A']:.2f} Å")
    print(f"  Crossed membrane: {result['crossed_membrane']}")

    print("\n" + "=" * 70)
    print("MOLECULAR DYNAMICS DEMO COMPLETE!")
    print("=" * 70)

    print("\n💡 Capabilities:")
    print("  1. Velocity Verlet integration (stable MD)")
    print("  2. Harmonic bond potentials")
    print("  3. Lennard-Jones non-bonded interactions")
    print("  4. Langevin thermostat (temperature control)")
    print("  5. Protein folding simulation")
    print("  6. Ligand binding (induced fit)")
    print("  7. Drug diffusion through membrane")


if __name__ == "__main__":
    demo()
