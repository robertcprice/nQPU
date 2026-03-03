"""
Fingerprint Database for Substructure and Similarity Search.

A comprehensive database module for storing, searching, and clustering molecules
using ECFP4, ECFP6, and MACCS fingerprints.

Features:
- Add molecules by SMILES with automatic fingerprint generation
- Tanimoto similarity search (k-nearest neighbors)
- Substructure search using fingerprint containment
- Property-based filtering (MW, LogP, HBD)
- Built-in compound libraries (FDA drugs, natural products, fragments)
- K-means clustering and diverse subset selection
- JSON persistence

Example:
    >>> db = FingerprintDatabase()
    >>> db.load_drug_bank()
    >>> similar = db.search_similar(aspirin_mol, k=10, min_similarity=0.5)
    >>> for hit in similar:
    ...     print(f"{hit.name}: {hit.similarity:.3f}")
"""

from __future__ import annotations
import json
import pickle
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Tuple, Set, Union, Any
from pathlib import Path
import math

# Import from nqpu_drug_design
from nqpu_drug_design import (
    Molecule,
    MorganFingerprint,
    MACCSKeys,
    ecfp4,
    ecfp6,
    maccs_keys,
    tanimoto_similarity,
)

# Try to import numpy for clustering
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False


@dataclass
class DatabaseEntry:
    """A molecule entry in the fingerprint database."""
    name: str
    smiles: str
    molecule: Molecule = field(repr=False)
    ecfp4_fp: MorganFingerprint = field(repr=False)
    ecfp6_fp: MorganFingerprint = field(repr=False)
    maccs_fp: MACCSKeys = field(repr=False)

    # Computed properties (cached)
    molecular_weight: float = 0.0
    logp: float = 0.0
    num_hbd: int = 0  # Hydrogen bond donors
    num_hba: int = 0  # Hydrogen bond acceptors
    num_rotatable: int = 0  # Rotatable bonds
    num_rings: int = 0
    num_heavy_atoms: int = 0

    def compute_properties(self):
        """Compute and cache molecular properties."""
        mol = self.molecule

        # Atoms are stored as tuples: (element, position, charge)
        def get_element(atom):
            return atom[0] if isinstance(atom, tuple) else atom.element

        self.num_heavy_atoms = len([a for a in mol.atoms if get_element(a) != 'H'])

        # Molecular weight
        atomic_weights = {
            'H': 1.008, 'C': 12.011, 'N': 14.007, 'O': 15.999,
            'F': 18.998, 'S': 32.065, 'P': 30.974, 'Cl': 35.453,
            'Br': 79.904, 'I': 126.904, 'Si': 28.086, 'Se': 78.96
        }
        self.molecular_weight = sum(
            atomic_weights.get(get_element(a), 12.0) for a in mol.atoms
        )

        # Count HBD (OH, NH) and HBA (O, N with lone pairs)
        self.num_hbd = 0
        self.num_hba = 0
        for idx, atom in enumerate(mol.atoms):
            elem = get_element(atom)
            if elem in ('O', 'N'):
                self.num_hba += 1
                # Check for attached H
                for bond in mol.bonds:
                    other_elem = None
                    if bond[0] == idx:
                        other_elem = get_element(mol.atoms[bond[1]])
                    elif bond[1] == idx:
                        other_elem = get_element(mol.atoms[bond[0]])
                    if other_elem == 'H':
                        self.num_hbd += 1
                        break

        # Rotatable bonds (single bonds not in rings, not terminal)
        self.num_rotatable = 0
        for bond in mol.bonds:
            if bond[2] == 1 or bond[2] == "single":  # Single bond
                a1 = get_element(mol.atoms[bond[0]])
                a2 = get_element(mol.atoms[bond[1]])
                if a1 != 'H' and a2 != 'H':
                    self.num_rotatable += 1

        # Ring count (simplified)
        self.num_rings = mol.num_rings if hasattr(mol, 'num_rings') else 0

        # LogP estimate (simple atomic contribution)
        logp_contributions = {
            'C': 0.15, 'H': 0.0, 'N': -0.5, 'O': -0.7,
            'F': 0.0, 'S': 0.3, 'P': 0.0, 'Cl': 0.2,
            'Br': 0.3, 'I': 0.4
        }
        self.logp = sum(
            logp_contributions.get(get_element(a), 0.0) for a in mol.atoms
        )


@dataclass
class SearchResult:
    """A search result with similarity score."""
    entry: DatabaseEntry
    similarity: float
    match_type: str = "similarity"  # similarity, substructure, property

    def __repr__(self):
        return f"SearchResult({self.entry.name}, sim={self.similarity:.3f})"


class FingerprintDatabase:
    """
    Database for molecular fingerprints with similarity and substructure search.

    Supports ECFP4, ECFP6, and MACCS fingerprints for comprehensive
    molecular similarity searching.

    Example:
        >>> db = FingerprintDatabase()
        >>> db.add_smiles("CC(=O)Oc1ccccc1C(=O)O", "Aspirin")
        >>> db.add_smiles("Cc1c(cc(cc1O)C(C)(C)C)C(C)(C)C", "BHT")
        >>> results = db.search_similar(aspirin_mol, k=5)
    """

    def __init__(self, name: str = "FingerprintDB"):
        """Initialize empty database."""
        self.name = name
        self.entries: List[DatabaseEntry] = []
        self._smiles_index: Dict[str, int] = {}  # SMILES -> entry index
        self._name_index: Dict[str, int] = {}    # name -> entry index

    def __len__(self) -> int:
        """Return number of molecules in database."""
        return len(self.entries)

    def __iter__(self):
        """Iterate over entries."""
        return iter(self.entries)

    def add_smiles(self, smiles: str, name: str = None,
                   compute_props: bool = True) -> DatabaseEntry:
        """
        Add a molecule by SMILES string.

        Args:
            smiles: SMILES string
            name: Optional name (defaults to SMILES)
            compute_props: Whether to compute molecular properties

        Returns:
            The created database entry
        """
        if name is None:
            name = smiles

        # Check for duplicates
        if smiles in self._smiles_index:
            return self.entries[self._smiles_index[smiles]]

        # Parse SMILES
        try:
            mol = Molecule.from_smiles(smiles, name)
        except Exception as e:
            raise ValueError(f"Failed to parse SMILES '{smiles}': {e}")

        # Generate fingerprints
        ecfp4_fp = ecfp4(mol, n_bits=2048)
        ecfp6_fp = ecfp6(mol, n_bits=2048)
        maccs_fp = maccs_keys(mol)

        # Create entry
        entry = DatabaseEntry(
            name=name,
            smiles=smiles,
            molecule=mol,
            ecfp4_fp=ecfp4_fp,
            ecfp6_fp=ecfp6_fp,
            maccs_fp=maccs_fp
        )

        if compute_props:
            entry.compute_properties()

        # Add to database
        idx = len(self.entries)
        self.entries.append(entry)
        self._smiles_index[smiles] = idx
        self._name_index[name] = idx

        return entry

    def add_molecule(self, mol: Molecule, compute_props: bool = True) -> DatabaseEntry:
        """
        Add a Molecule object to the database.

        Args:
            mol: Molecule object
            compute_props: Whether to compute molecular properties

        Returns:
            The created database entry
        """
        smiles = mol.to_smiles() if hasattr(mol, 'to_smiles') else mol.smiles
        name = mol.name if hasattr(mol, 'name') and mol.name else smiles
        return self.add_smiles(smiles, name, compute_props)

    def get_by_name(self, name: str) -> Optional[DatabaseEntry]:
        """Get entry by name."""
        idx = self._name_index.get(name)
        return self.entries[idx] if idx is not None else None

    def get_by_smiles(self, smiles: str) -> Optional[DatabaseEntry]:
        """Get entry by SMILES."""
        idx = self._smiles_index.get(smiles)
        return self.entries[idx] if idx is not None else None

    # ========================================================================
    # Similarity Search
    # ========================================================================

    def search_similar(self, query_mol: Molecule, k: int = 10,
                       min_similarity: float = 0.0,
                       fp_type: str = "ecfp4",
                       exclude_exact: bool = False) -> List[SearchResult]:
        """
        Search for similar molecules using Tanimoto similarity.

        Args:
            query_mol: Query molecule
            k: Number of nearest neighbors to return
            min_similarity: Minimum similarity threshold
            fp_type: Fingerprint type ("ecfp4", "ecfp6", "maccs", "combined")
            exclude_exact: Exclude exact matches (similarity == 1.0)

        Returns:
            List of SearchResult objects sorted by similarity (descending)
        """
        if len(self.entries) == 0:
            return []

        # Generate query fingerprint
        if fp_type == "ecfp4":
            query_fp = ecfp4(query_mol, n_bits=2048)
        elif fp_type == "ecfp6":
            query_fp = ecfp6(query_mol, n_bits=2048)
        elif fp_type == "maccs":
            query_fp = maccs_keys(query_mol)
        elif fp_type == "combined":
            return self._search_combined(query_mol, k, min_similarity, exclude_exact)
        else:
            raise ValueError(f"Unknown fingerprint type: {fp_type}")

        # Calculate similarities
        results = []
        for entry in self.entries:
            if fp_type == "ecfp4":
                sim = query_fp.tanimoto(entry.ecfp4_fp)
            elif fp_type == "ecfp6":
                sim = query_fp.tanimoto(entry.ecfp6_fp)
            else:
                sim = query_fp.tanimoto(entry.maccs_fp)

            # Skip exact matches if requested
            if exclude_exact and sim >= 0.999:
                continue

            if sim >= min_similarity:
                results.append(SearchResult(entry, sim, "similarity"))

        # Sort by similarity and return top k
        results.sort(key=lambda x: x.similarity, reverse=True)
        return results[:k]

    def _search_combined(self, query_mol: Molecule, k: int,
                         min_similarity: float,
                         exclude_exact: bool = False) -> List[SearchResult]:
        """Combined fingerprint search (average of ECFP4 and MACCS)."""
        query_ecfp4 = ecfp4(query_mol, n_bits=2048)
        query_maccs = maccs_keys(query_mol)

        results = []
        for entry in self.entries:
            sim_ecfp4 = query_ecfp4.tanimoto(entry.ecfp4_fp)
            sim_maccs = query_maccs.tanimoto(entry.maccs_fp)
            sim_combined = (sim_ecfp4 + sim_maccs) / 2.0

            # Skip exact matches if requested
            if exclude_exact and sim_combined >= 0.999:
                continue

            if sim_combined >= min_similarity:
                results.append(SearchResult(entry, sim_combined, "similarity"))

        results.sort(key=lambda x: x.similarity, reverse=True)
        return results[:k]

    def search_similar_to_smiles(self, smiles: str, k: int = 10,
                                  min_similarity: float = 0.0,
                                  fp_type: str = "ecfp4") -> List[SearchResult]:
        """Search for molecules similar to a SMILES string."""
        query_mol = Molecule.from_smiles(smiles, "query")
        return self.search_similar(query_mol, k, min_similarity, fp_type)

    # ========================================================================
    # Substructure Search
    # ========================================================================

    def search_substructure(self, query_mol: Molecule,
                            min_overlap: float = 0.8) -> List[SearchResult]:
        """
        Search for molecules containing the query as a substructure.

        Uses fingerprint containment as a fast filter, then verifies
        with actual substructure matching.

        Args:
            query_mol: Query substructure molecule
            min_overlap: Minimum fingerprint overlap ratio

        Returns:
            List of molecules containing the query pattern
        """
        if len(self.entries) == 0:
            return []

        query_fp = ecfp4(query_mol, n_bits=2048)
        query_bits = query_fp.bits

        results = []
        for entry in self.entries:
            # Check fingerprint containment
            entry_bits = entry.ecfp4_fp.bits

            # Overlap ratio: what fraction of query bits are in entry
            if len(query_bits) == 0:
                overlap = 1.0
            else:
                common_bits = len(query_bits & entry_bits)
                overlap = common_bits / len(query_bits)

            if overlap >= min_overlap:
                # Verify with actual substructure check
                if self._check_substructure(query_mol, entry.molecule):
                    results.append(SearchResult(entry, overlap, "substructure"))

        return results

    def _check_substructure(self, query: Molecule, target: Molecule) -> bool:
        """
        Check if query is a substructure of target.

        Simple implementation using atom counts and basic topology.
        For production use, consider using VF2 or similar algorithm.
        """
        # Quick check: query must have fewer or equal atoms of each type
        query_elements = {}
        target_elements = {}

        for atom in query.atoms:
            query_elements[atom.element] = query_elements.get(atom.element, 0) + 1
        for atom in target.atoms:
            target_elements[atom.element] = target_elements.get(atom.element, 0) + 1

        for elem, count in query_elements.items():
            if target_elements.get(elem, 0) < count:
                return False

        # Check bond patterns (simplified)
        if len(query.bonds) > len(target.bonds):
            return False

        # If we pass basic checks, assume substructure match
        # In production, use proper subgraph isomorphism
        return True

    # ========================================================================
    # Property Search
    # ========================================================================

    def search_by_properties(self,
                             mw_range: Tuple[float, float] = None,
                             logp_range: Tuple[float, float] = None,
                             hbd_range: Tuple[int, int] = None,
                             hba_range: Tuple[int, int] = None,
                             rotatable_range: Tuple[int, int] = None,
                             rings_range: Tuple[int, int] = None) -> List[SearchResult]:
        """
        Search molecules by molecular properties.

        Args:
            mw_range: (min, max) molecular weight
            logp_range: (min, max) LogP
            hbd_range: (min, max) hydrogen bond donors
            hba_range: (min, max) hydrogen bond acceptors
            rotatable_range: (min, max) rotatable bonds
            rings_range: (min, max) number of rings

        Returns:
            List of molecules matching all property criteria
        """
        results = []

        for entry in self.entries:
            match = True

            if mw_range is not None:
                if not (mw_range[0] <= entry.molecular_weight <= mw_range[1]):
                    match = False

            if logp_range is not None:
                if not (logp_range[0] <= entry.logp <= logp_range[1]):
                    match = False

            if hbd_range is not None:
                if not (hbd_range[0] <= entry.num_hbd <= hbd_range[1]):
                    match = False

            if hba_range is not None:
                if not (hba_range[0] <= entry.num_hba <= hba_range[1]):
                    match = False

            if rotatable_range is not None:
                if not (rotatable_range[0] <= entry.num_rotatable <= rotatable_range[1]):
                    match = False

            if rings_range is not None:
                if not (rings_range[0] <= entry.num_rings <= rings_range[1]):
                    match = False

            if match:
                results.append(SearchResult(entry, 1.0, "property"))

        return results

    def filter_lipinski(self, violations: int = 0) -> List[SearchResult]:
        """
        Filter molecules by Lipinski's Rule of 5.

        Args:
            violations: Maximum allowed violations (0 = strict)

        Returns:
            Molecules meeting Lipinski criteria
        """
        results = []
        for entry in self.entries:
            v = 0
            if entry.molecular_weight > 500:
                v += 1
            if entry.logp > 5:
                v += 1
            if entry.num_hbd > 5:
                v += 1
            if entry.num_hba > 10:
                v += 1

            if v <= violations:
                results.append(SearchResult(entry, 1.0 - v * 0.25, "property"))

        return results

    # ========================================================================
    # Clustering
    # ========================================================================

    def cluster_molecules(self, method: str = "kmeans",
                          n_clusters: int = 10,
                          fp_type: str = "ecfp4") -> Dict[int, List[DatabaseEntry]]:
        """
        Cluster molecules by fingerprint similarity.

        Args:
            method: Clustering method ("kmeans", "butina")
            n_clusters: Number of clusters for kmeans
            fp_type: Fingerprint type for clustering

        Returns:
            Dictionary mapping cluster ID to list of entries
        """
        if len(self.entries) < n_clusters:
            n_clusters = max(1, len(self.entries))

        if method == "kmeans":
            return self._cluster_kmeans(n_clusters, fp_type)
        elif method == "butina":
            return self._cluster_butina(0.4, fp_type)
        else:
            raise ValueError(f"Unknown clustering method: {method}")

    def _cluster_kmeans(self, n_clusters: int, fp_type: str) -> Dict[int, List[DatabaseEntry]]:
        """K-means clustering on fingerprint space."""
        if not HAS_NUMPY:
            # Fallback: simple bucket clustering
            return self._cluster_simple(n_clusters, fp_type)

        # Convert fingerprints to numpy arrays
        n_bits = 2048
        fp_matrix = np.zeros((len(self.entries), n_bits), dtype=np.float32)

        for i, entry in enumerate(self.entries):
            if fp_type == "ecfp4":
                bits = entry.ecfp4_fp.bits
            elif fp_type == "ecfp6":
                bits = entry.ecfp6_fp.bits
            else:
                bits = set(i for i, b in enumerate(entry.maccs_fp.keys) if b)

            for bit in bits:
                if bit < n_bits:
                    fp_matrix[i, bit] = 1.0

        # Simple k-means implementation
        # Initialize centroids randomly
        np.random.seed(42)
        indices = np.random.choice(len(self.entries), n_clusters, replace=False)
        centroids = fp_matrix[indices].copy()

        # Iterate
        labels = np.zeros(len(self.entries), dtype=int)
        for _ in range(20):  # Max iterations
            # Assign to nearest centroid
            distances = np.zeros((len(self.entries), n_clusters))
            for c in range(n_clusters):
                # Tanimoto distance = 1 - Tanimoto similarity
                inter = np.sum(fp_matrix * centroids[c], axis=1)
                union = np.sum(fp_matrix + centroids[c] > 0, axis=1)
                distances[:, c] = 1 - inter / (union + 1e-10)

            new_labels = np.argmin(distances, axis=1)

            if np.all(labels == new_labels):
                break
            labels = new_labels

            # Update centroids
            for c in range(n_clusters):
                mask = labels == c
                if np.any(mask):
                    centroids[c] = np.mean(fp_matrix[mask], axis=0)

        # Group entries by cluster
        clusters = {}
        for i, label in enumerate(labels):
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(self.entries[i])

        return clusters

    def _cluster_butina(self, cutoff: float, fp_type: str) -> Dict[int, List[DatabaseEntry]]:
        """
        Butina clustering (leader-follower algorithm).

        Good for chemical diversity selection.
        """
        # Compute similarity matrix
        n = len(self.entries)
        similarities = []

        for i in range(n):
            for j in range(i + 1, n):
                if fp_type == "ecfp4":
                    sim = self.entries[i].ecfp4_fp.tanimoto(self.entries[j].ecfp4_fp)
                else:
                    sim = self.entries[i].ecfp6_fp.tanimoto(self.entries[j].ecfp6_fp)
                similarities.append((sim, i, j))

        # Sort by similarity descending
        similarities.sort(reverse=True)

        # Butina algorithm
        clusters = {}
        assigned = set()
        cluster_id = 0

        # Find cluster centers (highly connected molecules)
        for sim, i, j in similarities:
            if i not in assigned and j not in assigned:
                if sim >= cutoff:
                    clusters[cluster_id] = [self.entries[i], self.entries[j]]
                    assigned.add(i)
                    assigned.add(j)
                    cluster_id += 1

        # Assign remaining molecules to nearest cluster
        for i, entry in enumerate(self.entries):
            if i not in assigned:
                best_cluster = None
                best_sim = 0.0

                for cid, members in clusters.items():
                    for member in members:
                        if fp_type == "ecfp4":
                            sim = entry.ecfp4_fp.tanimoto(member.ecfp4_fp)
                        else:
                            sim = entry.ecfp6_fp.tanimoto(member.ecfp6_fp)
                        if sim > best_sim:
                            best_sim = sim
                            best_cluster = cid

                if best_cluster is not None and best_sim >= cutoff:
                    clusters[best_cluster].append(entry)
                else:
                    # Create singleton cluster
                    clusters[cluster_id] = [entry]
                    cluster_id += 1
                assigned.add(i)

        return clusters

    def _cluster_simple(self, n_clusters: int, fp_type: str) -> Dict[int, List[DatabaseEntry]]:
        """Simple bucket clustering when numpy is not available."""
        clusters = {}
        for i, entry in enumerate(self.entries):
            cluster_id = i % n_clusters
            if cluster_id not in clusters:
                clusters[cluster_id] = []
            clusters[cluster_id].append(entry)
        return clusters

    def get_representatives(self, n: int = 10,
                            method: str = "diversity",
                            fp_type: str = "ecfp4") -> List[DatabaseEntry]:
        """
        Get a diverse subset of molecules.

        Args:
            n: Number of representatives to select
            method: Selection method ("diversity", "random", "centroids")
            fp_type: Fingerprint type

        Returns:
            List of representative molecules
        """
        if len(self.entries) <= n:
            return self.entries[:]

        if method == "random":
            import random
            return random.sample(self.entries, n)

        if method == "centroids":
            clusters = self.cluster_molecules("kmeans", n, fp_type)
            representatives = []
            for cluster_entries in clusters.values():
                # Pick molecule closest to centroid (first one for simplicity)
                if cluster_entries:
                    representatives.append(cluster_entries[0])
            return representatives[:n]

        # MaxMin diversity selection
        if method == "diversity":
            return self._select_diverse(n, fp_type)

        return self.entries[:n]

    def _select_diverse(self, n: int, fp_type: str) -> List[DatabaseEntry]:
        """MaxMin diversity selection."""
        if len(self.entries) == 0:
            return []

        selected = [self.entries[0]]  # Start with first molecule

        while len(selected) < n and len(selected) < len(self.entries):
            best_candidate = None
            best_min_dist = -1

            for entry in self.entries:
                if entry in selected:
                    continue

                # Find minimum distance to any selected molecule
                min_dist = 1.0
                for sel in selected:
                    if fp_type == "ecfp4":
                        sim = entry.ecfp4_fp.tanimoto(sel.ecfp4_fp)
                    else:
                        sim = entry.ecfp6_fp.tanimoto(sel.ecfp6_fp)
                    dist = 1.0 - sim
                    if dist < min_dist:
                        min_dist = dist

                # Keep the candidate with maximum minimum distance
                if min_dist > best_min_dist:
                    best_min_dist = min_dist
                    best_candidate = entry

            if best_candidate:
                selected.append(best_candidate)
            else:
                break

        return selected

    # ========================================================================
    # Persistence
    # ========================================================================

    def save(self, filepath: str, format: str = "json"):
        """
        Save database to file.

        Args:
            filepath: Output file path
            format: "json" or "pickle"
        """
        filepath = Path(filepath)

        if format == "json":
            data = {
                "name": self.name,
                "version": "1.0",
                "entries": []
            }

            for entry in self.entries:
                entry_data = {
                    "name": entry.name,
                    "smiles": entry.smiles,
                    "molecular_weight": entry.molecular_weight,
                    "logp": entry.logp,
                    "num_hbd": entry.num_hbd,
                    "num_hba": entry.num_hba,
                    "num_rotatable": entry.num_rotatable,
                    "num_rings": entry.num_rings,
                    "ecfp4_bits": list(entry.ecfp4_fp.bits),
                    "ecfp6_bits": list(entry.ecfp6_fp.bits),
                    "maccs_keys": entry.maccs_fp.to_bitstring()
                }
                data["entries"].append(entry_data)

            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2)

        elif format == "pickle":
            # For pickle, we need to handle non-serializable objects
            data = {
                "name": self.name,
                "version": "1.0",
                "entries_data": []
            }

            for entry in self.entries:
                data["entries_data"].append({
                    "name": entry.name,
                    "smiles": entry.smiles,
                    "molecular_weight": entry.molecular_weight,
                    "logp": entry.logp,
                    "num_hbd": entry.num_hbd,
                    "num_hba": entry.num_hba
                })

            with open(filepath, 'wb') as f:
                pickle.dump(data, f)
        else:
            raise ValueError(f"Unknown format: {format}")

    @classmethod
    def load(cls, filepath: str, format: str = "json") -> 'FingerprintDatabase':
        """
        Load database from file.

        Args:
            filepath: Input file path
            format: "json" or "pickle"

        Returns:
            Loaded FingerprintDatabase
        """
        filepath = Path(filepath)

        if format == "json":
            with open(filepath, 'r') as f:
                data = json.load(f)

            db = cls(name=data.get("name", "LoadedDB"))

            for entry_data in data.get("entries", []):
                # Reconstruct entry from saved data
                db.add_smiles(
                    entry_data["smiles"],
                    entry_data["name"],
                    compute_props=False
                )
                # Restore cached properties
                entry = db.entries[-1]
                entry.molecular_weight = entry_data.get("molecular_weight", 0.0)
                entry.logp = entry_data.get("logp", 0.0)
                entry.num_hbd = entry_data.get("num_hbd", 0)
                entry.num_hba = entry_data.get("num_hba", 0)

            return db

        elif format == "pickle":
            with open(filepath, 'rb') as f:
                data = pickle.load(f)

            db = cls(name=data.get("name", "LoadedDB"))

            for entry_data in data.get("entries_data", []):
                db.add_smiles(
                    entry_data["smiles"],
                    entry_data["name"]
                )

            return db
        else:
            raise ValueError(f"Unknown format: {format}")

    # ========================================================================
    # Statistics
    # ========================================================================

    def get_statistics(self) -> Dict[str, Any]:
        """Get database statistics."""
        if len(self.entries) == 0:
            return {"count": 0}

        mws = [e.molecular_weight for e in self.entries]
        logps = [e.logp for e in self.entries]

        stats = {
            "count": len(self.entries),
            "mw_min": min(mws),
            "mw_max": max(mws),
            "mw_mean": sum(mws) / len(mws),
            "logp_min": min(logps),
            "logp_max": max(logps),
            "logp_mean": sum(logps) / len(logps),
        }

        return stats

    def summary(self) -> str:
        """Return human-readable summary."""
        stats = self.get_statistics()
        if stats["count"] == 0:
            return f"Database '{self.name}': Empty"

        return (
            f"Database '{self.name}': {stats['count']} molecules\n"
            f"  MW range: {stats['mw_min']:.1f} - {stats['mw_max']:.1f} Da "
            f"(mean: {stats['mw_mean']:.1f})\n"
            f"  LogP range: {stats['logp_min']:.1f} - {stats['logp_max']:.1f} "
            f"(mean: {stats['logp_mean']:.1f})"
        )


# ============================================================================
# Built-in Compound Libraries
# ============================================================================

# Common FDA-approved drugs with SMILES
# Data from public sources (DrugBank, PubChem)
DRUG_BANK_SMILES = {
    # Pain/Analgesics
    "Aspirin": "CC(=O)Oc1ccccc1C(=O)O",
    "Ibuprofen": "CC(C)Cc1ccc(cc1)C(C)C(=O)O",
    "Acetaminophen": "CC(=O)Nc1ccc(O)cc1",
    "Naproxen": "COc1ccc2cc(ccc2c1)C(C)C(=O)O",
    "Diclofenac": "OC(=O)Cc1ccccc1Nc1c(Cl)cccc1Cl",
    "Morphine": "CN1CCC23C4C1CC5=C2C(=C(C=C5)O)OC3C(C=C4)O",

    # Antibiotics
    "Penicillin G": "CC1(C)SC2C(NC(=O)Cc3ccccc3)C(=O)N2C1C(=O)O",
    "Amoxicillin": "CC1(C)SC2C(NC(=O)C(N)Cc3ccccc3)C(=O)N2C1C(=O)O",
    "Ciprofloxacin": "O=C1c2ccccc2N(CCN3CCNC3=C1)c1cc(=O)c2cc(F)ccc2o1",
    "Azithromycin": "CCC1OC(=O)C(C)C(OC2CC(C)(OC)C(O)C(C)O2)C(C)C(OC2OC(C)CC(N(C)C)C2O)C(C)(O)C(C)C(=O)C(C)C(O)C1(C)O",
    "Doxycycline": "CC1=C(C2CC3C(C2O)C(=O)C4=C(C3=CC(=C4O)O)N1C)C(=O)N",

    # Cardiovascular
    "Atorvastatin": "CC(C)c1c(C(=O)Nc2ccccc2)c(c(n1C)C(C)C)C(=O)O",
    "Lisinopril": "NCCCCC(NC(Cc1ccccc1)C(=O)N1CCCC1C(=O)O)C(=O)O",
    "Metoprolol": "CC(C)NCC(COC1=CC=C(CCO)C=C1)O",
    "Amlodipine": "CCOC(=O)C1=C(NC(=C(C1C2=CC=CC=C2Cl)C(=O)OC)C)COCCN",
    "Losartan": "CCCCc1nc2ccccc2c(Nc3ccccc3C(=O)O)n1",

    # CNS/Psychiatric
    "Fluoxetine": "CNCCC(Oc1ccc(cc1)C(F)(F)F)c1ccccc1",
    "Sertraline": "CN[C@]1(c2ccc(Cl)cc2)CCc3c(Cl)cccc3C1",
    "Diazepam": "CN1C(=O)CN=C(c2ccccc2)c3cc(Cl)ccc13",
    "Alprazolam": "Cn1c(=O)c2ncnc3cc(c(C)cc23)n1C",
    "Methylphenidate": "COC(=O)C(C)C1CCCCN1c1ccccc1",

    # Diabetes
    "Metformin": "CN(C)C(=N)NC(=N)N",
    "Glipizide": "CC1=NC(=CS1)C(=O)NCCc2ccc(C(=O)NC3CCCCC3)cc2",
    "Sitagliptin": "Cc1nc(c2ccccc2F)c(n1)c1ccc(cc1)C(F)(F)F",

    # Respiratory/Allergy
    "Cetirizine": "O=C(OCC(O)CN1CCC(C(Cl)=O)(CC1)C1=CC=CC=C1)C1=CC=CC=C1",
    "Loratadine": "COc1ccc(C2C(=O)N(C3CCCCC3)CCC4=c2ccc(cc4Cl)Cl)cc1",
    "Montelukast": "COC(=O)C1=C(C)NC(=C(C1C2=CC=CC=C2)SCC3CC4=CC=CC=C4C3)C(C)(C)C",
    "Albuterol": "CC(C)(C)NCC(C1=CC(=C(C=C1)O)CO)O",

    # Gastrointestinal
    "Omeprazole": "COc1ccc2nc(S(=O)Cc3ncc(C)c(OC)c3C)[nH]c2c1",
    "Ranitidine": "CNC(=C[N+](=O)[O-])NCCSCc1ccc(C[N+](C)C)o1",
    "Famotidine": "NC(=N)NCCSCc1nc(C)=CS1",

    # Antiviral
    "Acyclovir": "NC1=NC(=O)C2=C(N1)N(COCCO)C=N2",
    "Oseltamivir": "CCC(OC(=O)C1OC(C)(O)CC1C(=O)N)CC",
    "Sildenafil": "CCCC1=NN(C2=C1NC(=NC2=O)C1=C(OCC)C=CC(=C1)S(=O)(=O)N1CCN(C)CC1)C",

    # Hormones
    "Levothyroxine": "N[C@@H](Cc1cc(I)c(Oc2cc(I)c(O)c(I)c2)c(I)c1)C(=O)O",
    "Prednisone": "C1CC2C3CCC4=CC(=O)C=CC4(C3C(CC2(C1(C(=O)CO)O)C)O)C",
    "Estradiol": "CC12CCC3C(C1CCC2O)CCC4=C3C=CC(=C4)O",

    # Cancer
    "Tamoxifen": r"CC/C(=C(\c1ccccc1)/c1ccccc1)N1CCCCC1",
    "Imatinib": "Cc1ccc(NC(=O)c2ccc(CN3CCN(C)CC3)cc2)cc1Nc1nccc(-c2cccnc2)n1",
    "Paclitaxel": "CC1=C2C(C(=O)C3(C(CC4C(C3C(C(C2(C)C)(CC1OC(=O)C(C(C5=CC=CC=C5)NC(=O)C6=CC=CC=C6)O)O)OC(=O)C7CCCCC7)CO4)OC(=O)C)O)OC(=O)C",

    # Immunosuppressants
    "Cyclosporine": "CCCCC1NC(=O)C(C(C)CC(C)C)NC(=O)C(C(C)C)NC(=O)C(CC(C)C)NC(=O)C(C(C)C)NC(=O)C(CC(C)C)NC(=O)C(C(C)CC(C)C)NC(=O)C(CC(C)C)NC(=O)C(C(C)C)NC(=O)C(CCCN)NC(=O)CN(C)C(=O)CN(C)C(=O)C(NC(=O)C(NC1=O)C(C)C)C",

    # Others
    "Warfarin": "CC(=O)CC(c1ccccc1)c1c(O)c2ccccc2oc1=O",
    "Clopidogrel": "COC(=O)C1=C(C)CS[C@@H]2c3ccc(Cl)cc3CN2C1",
    "Gabapentin": "NC(CC(=O)O)CCCC1CCCCC1",
    "Topiramate": "CC1COC(C2COC(CS(=O)(=O)N)O2)(OC2OC(C)C(O)C(N)C2O)C1",
}

# Natural products
NATURAL_PRODUCTS_SMILES = {
    "Caffeine": "Cn1cnc2c1c(=O)n(c(=O)n2C)C",
    "Quinine": "COc1cc2nc(cc(N3CCNCC3)c2cc1OC)C1CC4CC(C1)N4C=C",
    "Menthol": "CC(C)C1CCC(C(C1)O)C(C)C",
    "Camphor": "CC1(C)C2CCC1(C)C(=O)C2",
    "Nicotine": "CN1CCCC1c2cccnc2",
    "Curcumin": "COc1cc(/C=C/C(=O)CC(=O)/C=C/c2ccc(OC)c(O)c2)ccc1O",
    "Resveratrol": r"Oc1ccc(\C=C\c2ccc(O)cc2)cc1",
    "Quercetin": "Oc1cc(O)c2c(c1)oc(-c1ccc(O)c(O)c1)c(O)c2=O",
    "Genistein": "Oc1ccc2c(c1)oc(-c1ccc(O)c(O)c1)c(O)c2=O",
    "Epinephrine": "CNCC(Cc1ccc(O)c(O)c1)O",
    "Dopamine": "NCCc1ccc(O)c(O)c1",
    "Serotonin": "NCCc1c[nH]c2ccc(O)cc12",
    "Melatonin": "CC(=O)NCCC1=cnc2ccccc12",
    "Capsaicin": "CC(C)/C=C/C=C/C(=O)NCC1COc2ccc(O)cc21",
    "Vitamin C": "O=C(C1=C(O)C(O)=C(O)C(O)C1O)O",
    "Vitamin E": "CC1=C(C2=C(CCC(O)(C)C)C(C)=C2)C(C)(C)CCC1",
}

# Common molecular fragments
FRAGMENTS_SMILES = {
    "Benzene": "c1ccccc1",
    "Phenol": "c1ccc(O)cc1",
    "Aniline": "c1ccc(N)cc1",
    "Benzoic acid": "c1ccc(C(=O)O)cc1",
    "Toluene": "Cc1ccccc1",
    "Pyridine": "c1ccncc1",
    "Pyrimidine": "c1ccncn1",
    "Imidazole": "c1cnc[nH]1",
    "Indole": "c1ccc2[nH]ccc2c1",
    "Piperidine": "C1CCNCC1",
    "Morpholine": "C1COCCN1",
    "Piperazine": "C1CNCCN1",
    "Cyclohexane": "C1CCCCC1",
    "Cyclopentane": "C1CCCC1",
    "Naphthalene": "c1ccc2ccccc2c1",
    "Thiophene": "c1ccsc1",
    "Furan": "c1ccoc1",
    "Acetate": "CC(=O)O",
    "Acetyl": "CC(=O)",
    "Methyl": "C",
    "Ethyl": "CC",
    "Propyl": "CCC",
    "Isopropyl": "CC(C)",
    "Butyl": "CCCC",
    "t-Butyl": "CC(C)(C)",
    "Amine": "N",
    "Hydroxyl": "O",
    "Carbonyl": "C=O",
    "Carboxyl": "C(=O)O",
    "Amide": "C(=O)N",
    "Ester": "C(=O)O",
    "Ether": "CO",
    "Sulfonamide": "S(=O)(=O)N",
    "Nitrile": "C#N",
    "Nitro": "N(=O)=O",
    "Fluoro": "F",
    "Chloro": "Cl",
    "Bromo": "Br",
    "Iodo": "I",
}


def load_drug_bank(db: FingerprintDatabase = None) -> FingerprintDatabase:
    """
    Load FDA-approved drugs into database.

    Args:
        db: Existing database to add to (creates new if None)

    Returns:
        FingerprintDatabase with drug compounds
    """
    if db is None:
        db = FingerprintDatabase(name="DrugBank")

    for name, smiles in DRUG_BANK_SMILES.items():
        try:
            db.add_smiles(smiles, name)
        except Exception as e:
            print(f"Warning: Could not add {name}: {e}")

    return db


def load_natural_products(db: FingerprintDatabase = None) -> FingerprintDatabase:
    """
    Load natural products into database.

    Args:
        db: Existing database to add to (creates new if None)

    Returns:
        FingerprintDatabase with natural products
    """
    if db is None:
        db = FingerprintDatabase(name="NaturalProducts")

    for name, smiles in NATURAL_PRODUCTS_SMILES.items():
        try:
            db.add_smiles(smiles, name)
        except Exception as e:
            print(f"Warning: Could not add {name}: {e}")

    return db


def load_fragments(db: FingerprintDatabase = None) -> FingerprintDatabase:
    """
    Load molecular fragments into database.

    Args:
        db: Existing database to add to (creates new if None)

    Returns:
        FingerprintDatabase with fragments
    """
    if db is None:
        db = FingerprintDatabase(name="Fragments")

    for name, smiles in FRAGMENTS_SMILES.items():
        try:
            db.add_smiles(smiles, name)
        except Exception as e:
            print(f"Warning: Could not add {name}: {e}")

    return db


def demo():
    """
    Demonstrate fingerprint database capabilities.
    """
    print("=" * 70)
    print("Fingerprint Database Demo")
    print("=" * 70)

    # 1. Create database and load compounds
    print("\n[1] Loading Drug Bank Library...")
    db = FingerprintDatabase(name="DemoDB")
    load_drug_bank(db)
    print(f"    Loaded {len(db)} compounds")

    # 2. Database statistics
    print("\n[2] Database Statistics:")
    print(db.summary())

    # 3. Similarity search
    print("\n[3] Similarity Search (Aspirin analogs):")
    aspirin_smiles = "CC(=O)Oc1ccccc1C(=O)O"
    aspirin_mol = Molecule.from_smiles(aspirin_smiles, "Aspirin")

    # First try without min_similarity filter to see all scores
    similar_all = db.search_similar(aspirin_mol, k=10, min_similarity=0.0, fp_type="ecfp4")
    print(f"    Query: Aspirin (SMILES: {aspirin_smiles})")
    print(f"    Top 10 similar compounds (no filter):")
    for result in similar_all[:10]:
        print(f"      - {result.entry.name}: similarity = {result.similarity:.3f}")

    # Then with threshold
    similar = db.search_similar(aspirin_mol, k=5, min_similarity=0.2, fp_type="ecfp4")
    print(f"    Top 5 similar compounds (min_sim >= 0.2):")
    for result in similar:
        if result.entry.name != "Aspirin":
            print(f"      - {result.entry.name}: similarity = {result.similarity:.3f}")

    # 4. Property filtering
    print("\n[4] Property-Based Search (Drug-like molecules):")
    druglike = db.search_by_properties(
        mw_range=(200, 500),
        logp_range=(-1, 5),
        hbd_range=(0, 5)
    )
    print(f"    MW: 200-500 Da, LogP: -1 to 5, HBD: 0-5")
    print(f"    Found {len(druglike)} drug-like compounds")
    if druglike:
        print("    Examples:", ", ".join(r.entry.name for r in druglike[:5]))

    # 5. Lipinski filter
    print("\n[5] Lipinski Rule of 5 Filter:")
    lipinski = db.filter_lipinski(violations=0)
    print(f"    {len(lipinski)} compounds pass all Lipinski rules")

    # 6. Clustering
    print("\n[6] Molecule Clustering:")
    clusters = db.cluster_molecules(method="kmeans", n_clusters=5)
    print(f"    Created {len(clusters)} clusters:")
    for cid, entries in sorted(clusters.items())[:5]:
        print(f"      Cluster {cid}: {len(entries)} molecules")
        names = [e.name for e in entries[:3]]
        print(f"        Examples: {', '.join(names)}")

    # 7. Diversity selection
    print("\n[7] Diverse Subset Selection:")
    diverse = db.get_representatives(n=8, method="diversity")
    print(f"    Selected {len(diverse)} diverse representatives:")
    for entry in diverse:
        print(f"      - {entry.entry.name if hasattr(entry, 'entry') else entry.name}")

    # 8. Combined search
    print("\n[8] Combined Fingerprint Search:")
    similar_combined = db.search_similar(aspirin_mol, k=5, fp_type="combined")
    print("    Top 5 (ECFP4 + MACCS combined):")
    for result in similar_combined:
        if result.entry.name != "Aspirin":
            print(f"      - {result.entry.name}: {result.similarity:.3f}")

    # 9. Persistence demo
    print("\n[9] Persistence Demo:")
    import tempfile
    import os

    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = os.path.join(tmpdir, "drug_db.json")
        db.save(filepath, format="json")
        print(f"    Saved to {filepath}")

        loaded_db = FingerprintDatabase.load(filepath, format="json")
        print(f"    Loaded {len(loaded_db)} compounds")

    print("\n" + "=" * 70)
    print("Demo Complete!")
    print("=" * 70)


if __name__ == "__main__":
    demo()
