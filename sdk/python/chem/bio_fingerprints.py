"""
Bio-Conditioned Fingerprints for NQPU Drug Design

⚠️  EXPERIMENTAL / RESEARCH CODE ⚠️

This module is exploratory research code. The practical use case is unclear.

WHAT IT DOES:
Integrates molecular fingerprints with biological state vectors to create
bio-state-aware molecular representations.

THE PROBLEM:
Drugs don't have "bio states" - patients do. A drug's fingerprint shouldn't
change based on a patient's heart rate or dopamine level. This module
explores whether bio-state context could be useful for:
- Personalized medicine screening (find drugs suited for a patient's state)
- Understanding drug-body interactions
- Research on context-dependent drug activity

CURRENT STATUS:
- No validated use case
- Pattern definitions are arbitrary
- Needs domain expert input

Bio State Vector (12-dim):
- cardiac_phase_sin, cardiac_phase_cos (2 dims)
- coherence
- dopamine, serotonin, acetylcholine, norepinephrine (4 dims)
- somatic_comfort, somatic_arousal (2 dims)
- neural_stress
- encoding_boost
- vagal_tone

Example:
    >>> from bio_fingerprints import BioConditionedFingerprint, BioMolecularEncoder
    >>> from nqpu_drug_design import Molecule, ecfp4
    >>>
    >>> mol = Molecule.from_smiles('CC(=O)Oc1ccccc1C(=O)O', 'Aspirin')
    >>> bio_state = [0.0, 1.0, 0.8, 0.5, 0.5, 0.5, 0.5, 0.7, 0.3, 0.2, 1.0, 0.6]
    >>>
    >>> bio_fp = BioConditionedFingerprint.from_molecule(mol, bio_state)
    >>> embedding = BioMolecularEncoder().encode(mol, bio_state)
"""

from typing import List, Dict, Optional, Union, Tuple, Set
from dataclasses import dataclass
import numpy as np

try:
    from nqpu_drug_design import (
        Molecule, MorganFingerprint, MACCSKeys,
        ecfp4, ecfp6, maccs_keys
    )
    HAS_NQPU = True
except ImportError:
    HAS_NQPU = False


# ============================================================================
# BIO-RELEVANT MOLECULAR PATTERNS
# ============================================================================

# Pattern definitions: (name, SMILES pattern, bits to boost)
CARDIAC_PATTERNS = {
    # Beta-blockers: aryloxypropanolamine pattern
    'beta_blocker': {
        'patterns': ['OCC(N)COc1ccccc1', 'OCC(O)CN', 'aryloxypropanolamine'],
        'boost_bits': [100, 200, 300, 400, 500],
        'description': 'Beta-adrenergic blockers (propranolol-like)'
    },
    # Calcium channel blockers: dihydropyridine pattern
    'ccb': {
        'patterns': ['n1cc(ccc1C(=O)O)C', 'dihydropyridine'],
        'boost_bits': [150, 250, 350],
        'description': 'Calcium channel blockers (amlodipine-like)'
    },
    # ACE inhibitors: carboxyalkyl pattern
    'ace_inhibitor': {
        'patterns': ['C(CC(=O)O)NC(=O)', 'proline-like'],
        'boost_bits': [175, 275, 375],
        'description': 'ACE inhibitors (lisinopril-like)'
    }
}

NEURAL_PATTERNS = {
    # Serotonergic
    'serotonergic': {
        'patterns': ['c1ccc2[nH]ccc2c1', 'indole', 'tryptamine'],
        'boost_bits': [120, 220, 320, 420],
        'description': 'Serotonin receptor ligands'
    },
    # Dopaminergic
    'dopaminergic': {
        'patterns': ['c1cc(O)c(cc1CCN)', 'catechol', 'dopamine-like'],
        'boost_bits': [130, 230, 330, 430],
        'description': 'Dopamine receptor ligands'
    },
    # GABAergic
    'gabaergic': {
        'patterns': ['C(CC(=O)O)N', 'GABA', 'benzodiazepine'],
        'boost_bits': [140, 240, 340, 440],
        'description': 'GABA receptor modulators'
    },
    # Cholinergic
    'cholinergic': {
        'patterns': ['C(CO)N(C)C', 'choline', 'acetylcholine'],
        'boost_bits': [145, 245, 345, 445],
        'description': 'Acetylcholine receptor ligands'
    }
}

SOMATIC_PATTERNS = {
    # Anti-inflammatory (NSAID)
    'nsaid': {
        'patterns': ['CC(=O)Oc1ccccc1C(=O)O', 'carboxylate', 'arylacetic'],
        'boost_bits': [110, 210, 310, 410],
        'description': 'NSAIDs (aspirin, ibuprofen-like)'
    },
    # Corticosteroid
    'corticosteroid': {
        'patterns': ['C1CCC2C(C1)CCC1C2CCC2(C)C1CCC2(O)C(=O)CO', 'steroid'],
        'boost_bits': [160, 260, 360, 460],
        'description': 'Corticosteroids (prednisone-like)'
    },
    # Muscle relaxant
    'muscle_relaxant': {
        'patterns': ['C1CN(CCN1)C(=O)O', 'glycine-like'],
        'boost_bits': [155, 255, 355, 455],
        'description': 'Muscle relaxants'
    }
}


# ============================================================================
# BIO STATE ENCODER
# ============================================================================

@dataclass
class BioState:
    """Biological state vector with named components."""
    cardiac_phase: float = 0.0  # 0-2π
    coherence: float = 0.5      # 0-1
    dopamine: float = 0.5       # 0-1
    serotonin: float = 0.5      # 0-1
    acetylcholine: float = 0.5  # 0-1
    norepinephrine: float = 0.5 # 0-1
    somatic_comfort: float = 0.5   # 0-1
    somatic_arousal: float = 0.5   # 0-1
    neural_stress: float = 0.0     # 0-1
    encoding_boost: float = 1.0    # 0-2
    vagal_tone: float = 0.5        # 0-1

    def to_vector(self) -> np.ndarray:
        """Convert to 12-dim vector with sin/cos cardiac encoding."""
        return np.array([
            np.sin(self.cardiac_phase),  # cardiac_phase_sin
            np.cos(self.cardiac_phase),  # cardiac_phase_cos
            self.coherence,
            self.dopamine,
            self.serotonin,
            self.acetylcholine,
            self.norepinephrine,
            self.somatic_comfort,
            self.somatic_arousal,
            self.neural_stress,
            self.encoding_boost,
            self.vagal_tone,
        ], dtype=np.float32)

    @classmethod
    def from_vector(cls, vec: np.ndarray) -> 'BioState':
        """Create from 12-dim vector."""
        if len(vec) != 12:
            raise ValueError(f"Expected 12-dim vector, got {len(vec)}")

        # Extract cardiac phase from sin/cos
        cardiac_phase = np.arctan2(vec[0], vec[1])

        return cls(
            cardiac_phase=cardiac_phase,
            coherence=float(np.clip(vec[2], 0, 1)),
            dopamine=float(np.clip(vec[3], 0, 1)),
            serotonin=float(np.clip(vec[4], 0, 1)),
            acetylcholine=float(np.clip(vec[5], 0, 1)),
            norepinephrine=float(np.clip(vec[6], 0, 1)),
            somatic_comfort=float(np.clip(vec[7], 0, 1)),
            somatic_arousal=float(np.clip(vec[8], 0, 1)),
            neural_stress=float(np.clip(vec[9], 0, 1)),
            encoding_boost=float(np.clip(vec[10], 0, 2)),
            vagal_tone=float(np.clip(vec[11], 0, 1)),
        )


# ============================================================================
# BIO-CONDITIONED FINGERPRINT
# ============================================================================

class BioConditionedFingerprint:
    """
    Molecular fingerprint modulated by biological state.

    The base fingerprint (ECFP4, MACCS) is modulated by bio state
    to emphasize or de-emphasize certain molecular features based on
    the current physiological state.

    Example:
        >>> mol = Molecule.from_smiles('CC(=O)Oc1ccccc1C(=O)O', 'Aspirin')
        >>> bio_state = BioState(coherence=0.9, somatic_comfort=0.8)
        >>> bio_fp = BioConditionedFingerprint.from_molecule(mol, bio_state)
    """

    def __init__(self, bits: Set[int], weights: Dict[int, float], n_bits: int = 2048):
        """
        Initialize bio-conditioned fingerprint.

        Args:
            bits: Set of active bit positions
            weights: Bit position -> weight modulation
            n_bits: Total fingerprint length
        """
        self.bits = bits
        self.weights = weights
        self.n_bits = n_bits

    @classmethod
    def from_molecule(cls, mol: 'Molecule', bio_state: Union[BioState, np.ndarray, List[float]],
                      base_fp: str = 'ecfp4', n_bits: int = 2048,
                      conditioning: str = 'unified') -> 'BioConditionedFingerprint':
        """
        Create bio-conditioned fingerprint from molecule.

        Args:
            mol: Input molecule
            bio_state: BioState, 12-dim vector, or list
            base_fp: Base fingerprint type ('ecfp4', 'ecfp6', 'maccs')
            n_bits: Fingerprint length
            conditioning: Conditioning strategy ('unified', 'cardiac', 'neural', 'somatic')

        Returns:
            BioConditionedFingerprint instance
        """
        if not HAS_NQPU:
            raise ImportError("nqpu_drug_design not available")

        # Normalize bio state
        if isinstance(bio_state, BioState):
            bio_vec = bio_state.to_vector()
        elif isinstance(bio_state, list):
            bio_vec = np.array(bio_state, dtype=np.float32)
        else:
            bio_vec = bio_state

        # Generate base fingerprint
        if base_fp == 'ecfp4':
            fp = ecfp4(mol, n_bits=n_bits)
        elif base_fp == 'ecfp6':
            fp = ecfp6(mol, n_bits=n_bits)
        elif base_fp == 'maccs':
            fp = maccs_keys(mol)
            n_bits = 166
        else:
            fp = ecfp4(mol, n_bits=n_bits)

        # Apply bio-conditioning
        weights = cls._compute_weights(fp, bio_vec, conditioning)

        return cls(fp.bits, weights, n_bits)

    @classmethod
    def _compute_weights(cls, fp, bio_vec: np.ndarray, conditioning: str) -> Dict[int, float]:
        """Compute bio-conditioned weights for each bit."""
        weights = {}

        # Base weight for all bits
        base_weight = 1.0

        # Get bio state components
        coherence = bio_vec[2]
        dopamine = bio_vec[3]
        serotonin = bio_vec[4]
        acetylcholine = bio_vec[5]
        norepinephrine = bio_vec[6]
        somatic_comfort = bio_vec[7]
        somatic_arousal = bio_vec[8]
        neural_stress = bio_vec[9]
        vagal_tone = bio_vec[11]

        for bit in fp.bits:
            weight = base_weight

            if conditioning == 'unified':
                # Combined modulation based on all systems
                # Coherence boosts all weights slightly
                weight *= (1.0 + 0.2 * coherence)

                # Neural state affects certain bit ranges
                if bit % 256 < 128:  # First half of each 256-bit block
                    weight *= (1.0 + 0.1 * (dopamine + serotonin) / 2)
                else:
                    weight *= (1.0 + 0.1 * (acetylcholine + norepinephrine) / 2)

                # Somatic state affects bits related to functional groups
                if bit % 64 < 32:
                    weight *= (1.0 + 0.15 * somatic_comfort - 0.1 * somatic_arousal)

                # Stress slightly suppresses overall signal
                weight *= (1.0 - 0.1 * neural_stress)

                # Vagal tone adds subtle modulation
                weight *= (1.0 + 0.05 * vagal_tone)

            elif conditioning == 'cardiac':
                # Cardiac-focused modulation
                cardiac_mod = coherence * vagal_tone
                weight *= (1.0 + 0.3 * cardiac_mod)

                # Boost cardiac-relevant patterns
                for pattern_name, pattern_info in CARDIAC_PATTERNS.items():
                    if bit in pattern_info['boost_bits']:
                        weight *= (1.0 + 0.5 * coherence)

            elif conditioning == 'neural':
                # Neural-focused modulation
                neuromod_avg = (dopamine + serotonin + acetylcholine + norepinephrine) / 4
                weight *= (1.0 + 0.3 * neuromod_avg)

                # Boost neural-relevant patterns
                for pattern_name, pattern_info in NEURAL_PATTERNS.items():
                    if bit in pattern_info['boost_bits']:
                        weight *= (1.0 + 0.5 * neuromod_avg)

            elif conditioning == 'somatic':
                # Somatic-focused modulation
                somatic_balance = somatic_comfort - somatic_arousal + 0.5
                weight *= (1.0 + 0.2 * somatic_balance)

                # Boost somatic-relevant patterns
                for pattern_name, pattern_info in SOMATIC_PATTERNS.items():
                    if bit in pattern_info['boost_bits']:
                        weight *= (1.0 + 0.5 * somatic_comfort)

            weights[bit] = weight

        return weights

    def weighted_tanimoto(self, other: 'BioConditionedFingerprint') -> float:
        """Compute weighted Tanimoto similarity."""
        # Intersection with weights
        common_bits = self.bits & other.bits
        weighted_intersection = sum(
            min(self.weights.get(b, 1.0), other.weights.get(b, 1.0))
            for b in common_bits
        )

        # Union with weights
        all_bits = self.bits | other.bits
        weighted_union = sum(
            max(self.weights.get(b, 1.0), other.weights.get(b, 1.0))
            for b in all_bits
        )

        if weighted_union == 0:
            return 0.0

        return weighted_intersection / weighted_union

    def to_weighted_vector(self) -> np.ndarray:
        """Convert to weighted numpy vector."""
        vec = np.zeros(self.n_bits, dtype=np.float32)
        for bit in self.bits:
            vec[bit] = self.weights.get(bit, 1.0)
        return vec


# ============================================================================
# BIO MOLECULAR ENCODER
# ============================================================================

class BioMolecularEncoder:
    """
    Combines molecular fingerprint with bio state for unified embeddings.

    Produces embeddings suitable for ML models, including Bio-LoRA adapters.

    Example:
        >>> encoder = BioMolecularEncoder(embedding_dim=256)
        >>> mol = Molecule.from_smiles('CCO', 'Ethanol')
        >>> bio_state = BioState(coherence=0.9)
        >>> embedding = encoder.encode(mol, bio_state)
    """

    def __init__(self, embedding_dim: int = 256, fp_type: str = 'ecfp4'):
        """
        Initialize encoder.

        Args:
            embedding_dim: Output embedding dimension
            fp_type: Fingerprint type ('ecfp4', 'ecfp6', 'maccs')
        """
        self.embedding_dim = embedding_dim
        self.fp_type = fp_type

        # Random projection matrix for dimension reduction
        np.random.seed(42)  # Reproducible
        self._proj_matrix = None  # Created on first use

    def encode(self, mol: 'Molecule', bio_state: Union[BioState, np.ndarray, List[float]]) -> np.ndarray:
        """
        Encode molecule with bio state to unified embedding.

        Args:
            mol: Input molecule
            bio_state: BioState or 12-dim vector

        Returns:
            Embedding vector of shape (embedding_dim,)
        """
        # Create bio-conditioned fingerprint
        bio_fp = BioConditionedFingerprint.from_molecule(
            mol, bio_state, base_fp=self.fp_type
        )

        # Get weighted fingerprint vector
        fp_vec = bio_fp.to_weighted_vector()

        # Project to embedding dimension
        if self._proj_matrix is None:
            input_dim = len(fp_vec)
            self._proj_matrix = np.random.randn(input_dim, self.embedding_dim).astype(np.float32)
            self._proj_matrix /= np.sqrt(input_dim)  # Xavier initialization

        embedding = fp_vec @ self._proj_matrix

        # Normalize
        norm = np.linalg.norm(embedding)
        if norm > 1e-10:
            embedding = embedding / norm

        return embedding

    def encode_batch(self, molecules: List['Molecule'],
                     bio_states: List[Union[BioState, np.ndarray, List[float]]]) -> np.ndarray:
        """
        Encode batch of molecules with bio states.

        Args:
            molecules: List of molecules
            bio_states: List of bio states (one per molecule)

        Returns:
            Embedding matrix of shape (n_molecules, embedding_dim)
        """
        embeddings = []
        for mol, bio_state in zip(molecules, bio_states):
            emb = self.encode(mol, bio_state)
            embeddings.append(emb)
        return np.stack(embeddings, axis=0)


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def bio_similarity(mol1: 'Molecule', mol2: 'Molecule',
                   bio_state: Union[BioState, np.ndarray, List[float]],
                   conditioning: str = 'unified') -> float:
    """
    Compute bio-conditioned similarity between two molecules.

    Args:
        mol1: First molecule
        mol2: Second molecule
        bio_state: Shared bio state for both molecules
        conditioning: Conditioning strategy

    Returns:
        Bio-conditioned similarity score
    """
    bio_fp1 = BioConditionedFingerprint.from_molecule(mol1, bio_state, conditioning=conditioning)
    bio_fp2 = BioConditionedFingerprint.from_molecule(mol2, bio_state, conditioning=conditioning)
    return bio_fp1.weighted_tanimoto(bio_fp2)


# ============================================================================
# MODULE EXPORTS
# ============================================================================

__all__ = [
    # Classes
    'BioState',
    'BioConditionedFingerprint',
    'BioMolecularEncoder',

    # Functions
    'bio_similarity',

    # Pattern definitions
    'CARDIAC_PATTERNS',
    'NEURAL_PATTERNS',
    'SOMATIC_PATTERNS',
]


# ============================================================================
# TESTS
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Bio-Conditioned Fingerprints Tests")
    print("=" * 60)

    if not HAS_NQPU:
        print("ERROR: nqpu_drug_design not available")
        exit(1)

    # Test BioState
    print("\n1. BioState Tests:")
    bio = BioState(coherence=0.9, dopamine=0.7, somatic_comfort=0.8)
    vec = bio.to_vector()
    print(f"   BioState -> vector: {vec[:5]}...")
    bio2 = BioState.from_vector(vec)
    print(f"   Vector -> BioState: coherence={bio2.coherence:.2f}")

    # Test BioConditionedFingerprint
    print("\n2. BioConditionedFingerprint Tests:")
    mol = Molecule.from_smiles('CC(=O)Oc1ccccc1C(=O)O', 'Aspirin')

    bio_high = BioState(coherence=0.9, somatic_comfort=0.9)
    bio_low = BioState(coherence=0.2, somatic_comfort=0.2)

    bio_fp_high = BioConditionedFingerprint.from_molecule(mol, bio_high)
    bio_fp_low = BioConditionedFingerprint.from_molecule(mol, bio_low)

    print(f"   High coherence weights sum: {sum(bio_fp_high.weights.values()):.2f}")
    print(f"   Low coherence weights sum: {sum(bio_fp_low.weights.values()):.2f}")

    # Test different conditionings
    for cond in ['unified', 'cardiac', 'neural', 'somatic']:
        bio_fp = BioConditionedFingerprint.from_molecule(mol, bio_high, conditioning=cond)
        print(f"   {cond} conditioning: {len(bio_fp.weights)} weighted bits")

    # Test BioMolecularEncoder
    print("\n3. BioMolecularEncoder Tests:")
    encoder = BioMolecularEncoder(embedding_dim=128)
    embedding = encoder.encode(mol, bio_high)
    print(f"   Embedding shape: {embedding.shape}")
    print(f"   Embedding norm: {np.linalg.norm(embedding):.3f}")

    # Test batch encoding
    mols = [
        Molecule.from_smiles('CCO', 'Ethanol'),
        Molecule.from_smiles('CCC', 'Propane'),
    ]
    bio_states = [bio_high, bio_low]
    batch_emb = encoder.encode_batch(mols, bio_states)
    print(f"   Batch embedding shape: {batch_emb.shape}")

    # Test bio_similarity
    print("\n4. Bio-Conditioned Similarity Tests:")
    mol1 = Molecule.from_smiles('CC(=O)Oc1ccccc1C(=O)O', 'Aspirin')
    mol2 = Molecule.from_smiles('OC1=CC=CC=C1C(=O)O', 'Salicylic')

    sim = bio_similarity(mol1, mol2, bio_high)
    print(f"   Aspirin vs Salicylic (high coherence): {sim:.3f}")

    sim = bio_similarity(mol1, mol2, bio_low)
    print(f"   Aspirin vs Salicylic (low coherence): {sim:.3f}")

    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)
