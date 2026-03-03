"""
DNA/RNA and Organism Simulation Module for NQPU

Provides tools for:
- DNA sequence generation and manipulation
- RNA transcription and translation
- Protein synthesis simulation
- Single-cell organism genome modeling
- Plant genome basics

Example:
    >>> from dna_rna_organism import DNA, Organism
    >>> dna = DNA.random(length=1000)
    >>> rna = dna.transcribe()
    >>> protein = rna.translate()
    >>> organism = Organism.genome_minimal(dna)
"""

from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass, field
from enum import Enum
import random
from collections import Counter


# ============================================================================
# GENETIC CODE AND CONSTANTS
# ============================================================================

# Standard genetic code (codon -> amino acid)
CODON_TABLE = {
    # Phenylalanine
    'UUU': 'F', 'UUC': 'F',
    # Leucine
    'UUA': 'L', 'UUG': 'L', 'CUU': 'L', 'CUC': 'L', 'CUA': 'L', 'CUG': 'L',
    # Isoleucine
    'AUU': 'I', 'AUC': 'I', 'AUA': 'I',
    # Methionine (Start)
    'AUG': 'M',
    # Valine
    'GUU': 'V', 'GUC': 'V', 'GUA': 'V', 'GUG': 'V',
    # Serine
    'UCU': 'S', 'UCC': 'S', 'UCA': 'S', 'UCG': 'S', 'AGU': 'S', 'AGC': 'S',
    # Proline
    'CCU': 'P', 'CCC': 'P', 'CCA': 'P', 'CCG': 'P',
    # Threonine
    'ACU': 'T', 'ACC': 'T', 'ACA': 'T', 'ACG': 'T',
    # Alanine
    'GCU': 'A', 'GCC': 'A', 'GCA': 'A', 'GCG': 'A',
    # Tyrosine
    'UAU': 'Y', 'UAC': 'Y',
    # Histidine
    'CAU': 'H', 'CAC': 'H',
    # Glutamine
    'CAA': 'Q', 'CAG': 'Q',
    # Asparagine
    'AAU': 'N', 'AAC': 'N',
    # Lysine
    'AAA': 'K', 'AAG': 'K',
    # Aspartic acid
    'GAU': 'D', 'GAC': 'D',
    # Glutamic acid
    'GAA': 'E', 'GAG': 'E',
    # Cysteine
    'UGU': 'C', 'UGC': 'C',
    # Tryptophan
    'UGG': 'W',
    # Arginine
    'CGU': 'R', 'CGC': 'R', 'CGA': 'R', 'CGG': 'R', 'AGA': 'R', 'AGG': 'R',
    # Glycine
    'GGU': 'G', 'GGC': 'G', 'GGA': 'G', 'GGG': 'G',
    # Stop codons
    'UAA': '*', 'UAG': '*', 'UGA': '*',
}

# Amino acid full names
AMINO_ACID_NAMES = {
    'A': 'Alanine', 'R': 'Arginine', 'N': 'Asparagine', 'D': 'Aspartic acid',
    'C': 'Cysteine', 'E': 'Glutamic acid', 'Q': 'Glutamine', 'G': 'Glycine',
    'H': 'Histidine', 'I': 'Isoleucine', 'L': 'Leucine', 'K': 'Lysine',
    'M': 'Methionine', 'F': 'Phenylalanine', 'P': 'Proline', 'S': 'Serine',
    'T': 'Threonine', 'W': 'Tryptophan', 'Y': 'Tyrosine', 'V': 'Valine',
    '*': 'Stop'
}

# Amino acid properties
AMINO_ACID_PROPERTIES = {
    'A': {'hydrophobic': True, 'polar': False, 'charged': False, 'mw': 89.09},
    'R': {'hydrophobic': False, 'polar': True, 'charged': True, 'mw': 174.20},
    'N': {'hydrophobic': False, 'polar': True, 'charged': False, 'mw': 132.12},
    'D': {'hydrophobic': False, 'polar': True, 'charged': True, 'mw': 133.10},
    'C': {'hydrophobic': True, 'polar': False, 'charged': False, 'mw': 121.15},
    'E': {'hydrophobic': False, 'polar': True, 'charged': True, 'mw': 147.13},
    'Q': {'hydrophobic': False, 'polar': True, 'charged': False, 'mw': 146.15},
    'G': {'hydrophobic': True, 'polar': False, 'charged': False, 'mw': 75.07},
    'H': {'hydrophobic': False, 'polar': True, 'charged': True, 'mw': 155.16},
    'I': {'hydrophobic': True, 'polar': False, 'charged': False, 'mw': 131.17},
    'L': {'hydrophobic': True, 'polar': False, 'charged': False, 'mw': 131.17},
    'K': {'hydrophobic': False, 'polar': True, 'charged': True, 'mw': 146.19},
    'M': {'hydrophobic': True, 'polar': False, 'charged': False, 'mw': 149.21},
    'F': {'hydrophobic': True, 'polar': False, 'charged': False, 'mw': 165.19},
    'P': {'hydrophobic': False, 'polar': False, 'charged': False, 'mw': 115.13},
    'S': {'hydrophobic': False, 'polar': True, 'charged': False, 'mw': 105.09},
    'T': {'hydrophobic': False, 'polar': True, 'charged': False, 'mw': 119.12},
    'W': {'hydrophobic': True, 'polar': False, 'charged': False, 'mw': 204.23},
    'Y': {'hydrophobic': False, 'polar': True, 'charged': False, 'mw': 181.19},
    'V': {'hydrophobic': True, 'polar': False, 'charged': False, 'mw': 117.15},
    '*': {'hydrophobic': False, 'polar': False, 'charged': False, 'mw': 0},
}

# Essential genes for minimal organism (based on Mycoplasma genitalium)
MINIMAL_GENOME_GENES = [
    'dnaA',    # DNA replication initiator
    'dnaB',    # DNA helicase
    'dnaC',    # DNA primase
    'dnaE',    # DNA polymerase III
    'dnaN',    # DNA polymerase sliding clamp
    'gyrA',    # DNA gyrase subunit A
    'gyrB',    # DNA gyrase subunit B
    'rpoA',    # RNA polymerase alpha
    'rpoB',    # RNA polymerase beta
    'rpoC',    # RNA polymerase beta prime
    'rpsA',    # Ribosomal protein S1
    'rpsB',    # Ribosomal protein S2
    'rpsC',    # Ribosomal protein S3
    'rpsD',    # Ribosomal protein S4
    'rpsE',    # Ribosomal protein S5
    'rpsG',    # Ribosomal protein S7
    'rpsH',    # Ribosomal protein S8
    'rpsI',    # Ribosomal protein S9
    'rpsJ',    # Ribosomal protein S10
    'rpsK',    # Ribosomal protein S11
    'rpsL',    # Ribosomal protein S12
    'rpsM',    # Ribosomal protein S13
    'rpsN',    # Ribosomal protein S14
    'rpsO',    # Ribosomal protein S15
    'rpsP',    # Ribosomal protein S16
    'rpsQ',    # Ribosomal protein S17
    'rpsR',    # Ribosomal protein S18
    'rpsS',    # Ribosomal protein S19
    'rpsT',    # Ribosomal protein S20
    'rplA',    # Ribosomal protein L1
    'rplB',    # Ribosomal protein L2
    'rplC',    # Ribosomal protein L3
    'rplD',    # Ribosomal protein L4
    'rplE',    # Ribosomal protein L5
    'rplF',    # Ribosomal protein L6
    'rplK',    # Ribosomal protein L11
    'rplL',    # Ribosomal protein L7/L12
    'rplM',    # Ribosomal protein L13
    'rplN',    # Ribosomal protein L14
    'rplO',    # Ribosomal protein L15
    'rplP',    # Ribosomal protein L16
    'rplQ',    # Ribosomal protein L17
    'rplR',    # Ribosomal protein L18
    'rplS',    # Ribosomal protein L19
    'rplT',    # Ribosomal protein L20
    'rplU',    # Ribosomal protein L21
    'rplV',    # Ribosomal protein L22
    'rplW',    # Ribosomal protein L23
    'rplX',    # Ribosomal protein L24
    'rplY',    # Ribosomal protein L25
    'infA',    # Translation initiation factor IF-1
    'infB',    # Translation initiation factor IF-2
    'infC',    # Translation initiation factor IF-3
    'tufA',    # Elongation factor Tu
    'fusA',    # Elongation factor G
    'prfA',    # Peptide chain release factor 1
    'prfB',    # Peptide chain release factor 2
    'atpA',    # ATP synthase alpha
    'atpB',    # ATP synthase beta
    'atpC',    # ATP synthase gamma
    'atpD',    # ATP synthase delta
    'atpE',    # ATP synthase epsilon
    'atpF',    # ATP synthase B
    'atpH',    # ATP synthase A
    'secA',    # Protein translocase
    'secY',    # Protein translocase
    'secE',    # Protein translocase
    'ffs',     # 4.5S RNA
    'tsaD',    # tRNA modification
    'pyrG',    # CTP synthase
    'purA',    # Adenylosuccinate synthase
    'purB',    # Adenylosuccinate lyase
    'purC',    # Phosphoribosylaminoimidazole-succinocarboxamide synthase
    'purD',    # Phosphoribosylamine--glycine ligase
    'purE',    # Phosphoribosylaminoimidazole carboxylase
    'purF',    # Amidophosphoribosyltransferase
    'purH',    # Bifunctional purine biosynthesis protein
    'purK',    # Phosphoribosylaminoimidazole carboxylase
    'purL',    # Phosphoribosylformylglycinamidine synthase
    'purM',    # Phosphoribosylaminoimidazole synthetase
    'purN',    # Phosphoribosylglycinamide formyltransferase
    'glyA',    # Serine hydroxymethyltransferase
    'glyQ',    # Glycyl-tRNA synthetase alpha
    'glyS',    # Glycyl-tRNA synthetase beta
    'alaS',    # Alanyl-tRNA synthetase
    'argS',    # Arginyl-tRNA synthetase
    'asnS',    # Asparaginyl-tRNA synthetase
    'aspS',    # Aspartyl-tRNA synthetase
    'cysS',    # Cysteinyl-tRNA synthetase
    'glnS',    # Glutaminyl-tRNA synthetase
    'gluS',    # Glutamyl-tRNA synthetase
    'hisS',    # Histidyl-tRNA synthetase
    'ileS',    # Isoleucyl-tRNA synthetase
    'leuS',    # Leucyl-tRNA synthetase
    'lysS',    # Lysyl-tRNA synthetase
    'metS',    # Methionyl-tRNA synthetase
    'pheS',    # Phenylalanyl-tRNA synthetase alpha
    'pheT',    # Phenylalanyl-tRNA synthetase beta
    'proS',    # Prolyl-tRNA synthetase
    'serS',    # Seryl-tRNA synthetase
    'thrS',    # Threonyl-tRNA synthetase
    'trpS',    # Tryptophanyl-tRNA synthetase
    'tyrS',    # Tyrosyl-tRNA synthetase
    'valS',    # Valyl-tRNA synthetase
    'fmt',     # Methionyl-tRNA formyltransferase
    'def',     # Peptide deformylase
    'map',     # Methionine aminopeptidase
    'murA',    # Cell wall biosynthesis
    'murB',    # Cell wall biosynthesis
    'murC',    # Cell wall biosynthesis
    'murD',    # Cell wall biosynthesis
    'murE',    # Cell wall biosynthesis
    'murF',    # Cell wall biosynthesis
    'murG',    # Cell wall biosynthesis
    'mraY',    # Cell wall biosynthesis
    'ftsZ',    # Cell division
    'ftsA',    # Cell division
    'ftsW',    # Cell division
    'ftsQ',    # Cell division
    'ftsL',    # Cell division
    'pbpA',    # Penicillin-binding protein
    'pbpB',    # Penicillin-binding protein
    'rpmA',    # Ribosomal protein L27
    'rpmB',    # Ribosomal protein L28
    'rpmC',    # Ribosomal protein L29
    'rpmD',    # Ribosomal protein L30
    'rpmE',    # Ribosomal protein L31
    'rpmF',    # Ribosomal protein L32
    'rpmG',    # Ribosomal protein L33
    'rpmH',    # Ribosomal protein L34
    'rpmI',    # Ribosomal protein L35
    'rpmJ',    # Ribosomal protein L36
]

# Gene -> approximate protein length (codons)
GENE_LENGTHS = {gene: random.randint(150, 600) for gene in MINIMAL_GENOME_GENES}


class OrganismType(Enum):
    """Types of organisms."""
    BACTERIA = "bacteria"
    ARCHAEA = "archaea"
    EUKARYOTE = "eukaryote"
    PLANT = "plant"
    FUNGUS = "fungus"
    PROTOZOAN = "protozoan"


class GenomeType(Enum):
    """Types of genomes."""
    DNA = "dna"
    RNA = "rna"
    SINGLE_STRANDED = "ss"
    DOUBLE_STRANDED = "ds"


# ============================================================================
# DNA CLASS
# ============================================================================

@dataclass
class DNA:
    """
    Represents a DNA sequence.

    DNA is stored as a string of nucleotides (A, T, G, C).
    Supports transcription to RNA and various analyses.
    """
    sequence: str
    name: str = "Unknown"
    circular: bool = False  # Circular (plasmid/bacterial) vs linear

    @classmethod
    def random(cls, length: int = 1000, gc_content: float = 0.5, name: str = "Random") -> "DNA":
        """Generate a random DNA sequence with specified GC content."""
        nucleotides = []
        for _ in range(length):
            if random.random() < gc_content:
                nucleotides.append(random.choice(['G', 'C']))
            else:
                nucleotides.append(random.choice(['A', 'T']))
        return cls(''.join(nucleotides), name)

    @classmethod
    def from_sequence(cls, sequence: str, name: str = "Unknown", circular: bool = False) -> "DNA":
        """Create DNA from a sequence string (cleans input)."""
        clean = ''.join(c.upper() for c in sequence if c.upper() in 'ATGC')
        return cls(clean, name, circular)

    @property
    def length(self) -> int:
        """Length of the DNA sequence."""
        return len(self.sequence)

    @property
    def complement(self) -> str:
        """Return the complementary strand."""
        complement_map = {'A': 'T', 'T': 'A', 'G': 'C', 'C': 'G'}
        return ''.join(complement_map[base] for base in self.sequence)

    @property
    def reverse_complement(self) -> str:
        """Return the reverse complement strand."""
        return self.complement[::-1]

    def gc_content(self) -> float:
        """Calculate GC content as a fraction."""
        if len(self.sequence) == 0:
            return 0.0
        gc = sum(1 for base in self.sequence if base in 'GC')
        return gc / len(self.sequence)

    def nucleotide_counts(self) -> Dict[str, int]:
        """Count each nucleotide."""
        return Counter(self.sequence)

    def transcribe(self) -> "RNA":
        """
        Transcribe DNA to mRNA.

        DNA -> RNA: T -> U, others stay same
        """
        rna_seq = self.sequence.replace('T', 'U')
        return RNA(rna_seq, f"{self.name}_mRNA")

    def find_orfs(self, min_length: int = 100) -> List[Tuple[int, int, str]]:
        """
        Find open reading frames (ORFs) in all six frames.

        Returns list of (start, end, frame) tuples.
        """
        orfs = []

        # Search both strands
        sequences = [
            (self.sequence, '+'),
            (self.reverse_complement, '-')
        ]

        for seq, strand in sequences:
            for frame in range(3):
                i = frame
                while i < len(seq) - 2:
                    codon = seq[i:i+3]
                    if codon == 'ATG':  # Start codon
                        # Look for stop codon
                        j = i + 3
                        while j < len(seq) - 2:
                            stop_codon = seq[j:j+3]
                            if stop_codon in ('TAA', 'TAG', 'TGA'):
                                if j - i >= min_length:
                                    orfs.append((i, j + 3, f"{strand}{frame}"))
                                break
                            j += 3
                        i = j if j > i else i + 3
                    else:
                        i += 3

        return orfs

    def find_restriction_sites(self, enzyme: str) -> List[int]:
        """Find restriction enzyme recognition sites."""
        enzymes = {
            'EcoRI': 'GAATTC',
            'BamHI': 'GGATCC',
            'HindIII': 'AAGCTT',
            'XbaI': 'TCTAGA',
            'SalI': 'GTCGAC',
            'PstI': 'CTGCAG',
            'KpnI': 'GGTACC',
            'SacI': 'GAGCTC',
        }

        if enzyme not in enzymes:
            raise ValueError(f"Unknown enzyme: {enzyme}")

        site = enzymes[enzyme]
        positions = []
        start = 0
        while True:
            pos = self.sequence.find(site, start)
            if pos == -1:
                break
            positions.append(pos)
            start = pos + 1

        return positions

    def mutate(self, mutation_rate: float = 0.01) -> "DNA":
        """Apply random point mutations."""
        bases = list(self.sequence)
        for i in range(len(bases)):
            if random.random() < mutation_rate:
                original = bases[i]
                alternatives = [b for b in 'ATGC' if b != original]
                bases[i] = random.choice(alternatives)
        return DNA(''.join(bases), f"{self.name}_mutant", self.circular)

    def hamming_distance(self, other: "DNA") -> int:
        """Calculate Hamming distance between two sequences."""
        if len(self.sequence) != len(other.sequence):
            raise ValueError("Sequences must be same length")
        return sum(a != b for a, b in zip(self.sequence, other.sequence))

    def __len__(self) -> int:
        return len(self.sequence)

    def __str__(self) -> str:
        preview = self.sequence[:50] + "..." if len(self.sequence) > 50 else self.sequence
        return f"DNA({self.name}, {len(self)}bp): {preview}"


# ============================================================================
# RNA CLASS
# ============================================================================

@dataclass
class RNA:
    """
    Represents an RNA sequence.

    RNA is stored as a string of nucleotides (A, U, G, C).
    Supports translation to protein.
    """
    sequence: str
    name: str = "Unknown"
    rna_type: str = "mRNA"  # mRNA, tRNA, rRNA, etc.

    @classmethod
    def random(cls, length: int = 1000, name: str = "Random") -> "RNA":
        """Generate a random RNA sequence."""
        nucleotides = [random.choice(['A', 'U', 'G', 'C']) for _ in range(length)]
        return cls(''.join(nucleotides), name)

    @classmethod
    def from_sequence(cls, sequence: str, name: str = "Unknown") -> "RNA":
        """Create RNA from a sequence string."""
        clean = ''.join(c.upper() for c in sequence if c.upper() in 'AUGC')
        return cls(clean, name)

    @property
    def length(self) -> int:
        return len(self.sequence)

    def translate(self, start: int = 0, frame: int = 0) -> "Protein":
        """
        Translate RNA to protein using the standard genetic code.

        Args:
            start: Starting position in the sequence
            frame: Reading frame (0, 1, or 2)

        Returns:
            Protein sequence
        """
        protein_seq = []
        pos = start + frame

        while pos < len(self.sequence) - 2:
            codon = self.sequence[pos:pos+3]
            if len(codon) < 3:
                break

            amino_acid = CODON_TABLE.get(codon, 'X')  # X for unknown codons

            if amino_acid == '*':  # Stop codon
                break

            protein_seq.append(amino_acid)
            pos += 3

        return Protein(''.join(protein_seq), f"{self.name}_protein")

    def translate_all_orfs(self, min_length: int = 30) -> List["Protein"]:
        """Translate all open reading frames."""
        proteins = []

        for frame in range(3):
            # Find start codon
            i = frame
            while i < len(self.sequence) - 2:
                codon = self.sequence[i:i+3]
                if codon == 'AUG':  # Start codon
                    protein = self.translate(i, 0)
                    if len(protein.sequence) >= min_length:
                        protein.name = f"{self.name}_ORF{len(proteins)+1}"
                        proteins.append(protein)
                    i += 3
                else:
                    i += 3

        return proteins

    def reverse_transcribe(self) -> DNA:
        """Reverse transcribe RNA to DNA."""
        dna_seq = self.sequence.replace('U', 'T')
        return DNA(dna_seq, f"{self.name}_cDNA")

    def __len__(self) -> int:
        return len(self.sequence)

    def __str__(self) -> str:
        preview = self.sequence[:50] + "..." if len(self.sequence) > 50 else self.sequence
        return f"RNA({self.name}, {len(self)}nt): {preview}"


# ============================================================================
# PROTEIN CLASS
# ============================================================================

@dataclass
class Protein:
    """
    Represents a protein sequence (amino acids).
    """
    sequence: str
    name: str = "Unknown"

    @property
    def length(self) -> int:
        return len(self.sequence)

    def molecular_weight(self) -> float:
        """Calculate approximate molecular weight in Daltons."""
        # Sum of amino acid weights minus water (18.015) per peptide bond
        mw = sum(AMINO_ACID_PROPERTIES.get(aa, {'mw': 110}).get('mw', 110)
                 for aa in self.sequence)
        # Subtract water for peptide bonds
        mw -= 18.015 * (len(self.sequence) - 1) if len(self.sequence) > 1 else 0
        return mw

    def amino_acid_counts(self) -> Dict[str, int]:
        """Count each amino acid."""
        return Counter(self.sequence)

    def hydrophobicity(self) -> float:
        """Calculate average hydrophobicity."""
        if len(self.sequence) == 0:
            return 0.0
        hydrophobic = sum(1 for aa in self.sequence
                         if AMINO_ACID_PROPERTIES.get(aa, {}).get('hydrophobic', False))
        return hydrophobic / len(self.sequence)

    def charge_at_ph(self, ph: float = 7.0) -> float:
        """Estimate net charge at given pH (simplified)."""
        # Simplified pKa values
        charge = 0.0
        for aa in self.sequence:
            if aa == 'K':  # Lysine - positive at pH 7
                charge += 1 if ph < 10.5 else 0
            elif aa == 'R':  # Arginine - positive at pH 7
                charge += 1 if ph < 12.5 else 0
            elif aa == 'H':  # Histidine - positive at pH 7
                charge += 1 if ph < 6.0 else 0
            elif aa == 'D':  # Aspartic acid - negative at pH 7
                charge -= 1 if ph > 3.9 else 0
            elif aa == 'E':  # Glutamic acid - negative at pH 7
                charge -= 1 if ph > 4.1 else 0
        return charge

    def find_domains(self) -> List[Dict]:
        """
        Find potential protein domains (simplified motif detection).
        """
        domains = []

        # Common motifs
        motifs = {
            'ATP_binding': ['GGAGVGK', 'GXXXXGK'],  # Walker A motif
            'DNA_binding': ['C2H2', 'CXXC'],  # Zinc finger
            'kinase': ['HRDLKXXN'],  # Kinase catalytic
            'transmembrane': 'hydrophobic_stretch',
        }

        # Find hydrophobic stretches (potential transmembrane)
        if self.hydrophobicity() > 0.6:
            domains.append({
                'type': 'potential_transmembrane',
                'hydrophobicity': self.hydrophobicity()
            })

        return domains

    def __len__(self) -> int:
        return len(self.sequence)

    def __str__(self) -> str:
        preview = self.sequence[:30] + "..." if len(self.sequence) > 30 else self.sequence
        return f"Protein({self.name}, {len(self)}aa, {self.molecular_weight():.1f}Da)"


# ============================================================================
# GENE CLASS
# ============================================================================

@dataclass
class Gene:
    """
    Represents a gene with promoter, coding sequence, and terminator.
    """
    name: str
    coding_sequence: DNA
    promoter: Optional[str] = None
    terminator: Optional[str] = None
    function: str = "Unknown"
    essential: bool = False

    @property
    def length(self) -> int:
        return len(self.coding_sequence)

    def transcribe(self) -> RNA:
        """Transcribe the gene to mRNA."""
        return self.coding_sequence.transcribe()

    def translate(self) -> Protein:
        """Translate the gene to protein."""
        rna = self.transcribe()
        return rna.translate()

    @classmethod
    def random(cls, name: str = "RandomGene", length: int = 500) -> "Gene":
        """Create a random gene."""
        # Simple promoter consensus
        promoter = "TTGACA" + "N" * 17 + "TATAAT"  # -35 and -10 boxes

        # Start with ATG
        cds = "ATG" + ''.join(random.choice('ATGC') for _ in range(length - 6)) + "TAA"

        # Simple terminator
        terminator = "GGCC" + "N" * 10 + "GGCC"  # Hairpin-like

        return cls(
            name=name,
            coding_sequence=DNA(cds, name),
            promoter=promoter,
            terminator=terminator
        )


# ============================================================================
# ORGANISM CLASS
# ============================================================================

@dataclass
class Organism:
    """
    Represents a living organism with a genome.

    Can model bacteria, archaea, or simple eukaryotes.
    """
    name: str
    genome: DNA
    organism_type: OrganismType
    genes: List[Gene] = field(default_factory=list)

    @classmethod
    def minimal_bacterium(cls, name: str = "MinimalBacterium") -> "Organism":
        """
        Create a minimal bacterium based on Mycoplasma genitalium.

        M. genitalium has ~525 genes, one of the smallest known genomes.
        """
        # Generate genome with essential genes
        genes = []
        total_length = 0

        for gene_name in MINIMAL_GENOME_GENES[:100]:  # Use subset for demo
            gene_len = random.randint(200, 1500)
            gene = Gene(
                name=gene_name,
                coding_sequence=DNA.random(gene_len, gc_content=0.32),
                essential=True,
                function=f"Essential gene: {gene_name}"
            )
            genes.append(gene)
            total_length += gene_len

        # Add intergenic regions
        genome_parts = []
        for gene in genes:
            # Random intergenic spacer (50-200 bp)
            spacer = ''.join(random.choice('ATGC') for _ in range(random.randint(50, 200)))
            genome_parts.append(spacer)
            genome_parts.append(gene.coding_sequence.sequence)

        genome = DNA(''.join(genome_parts), name=name)

        return cls(
            name=name,
            genome=genome,
            organism_type=OrganismType.BACTERIA,
            genes=genes
        )

    @classmethod
    def simple_plant_cell(cls, name: str = "SimplePlantCell") -> "Organism":
        """
        Create a simplified plant cell model.

        Plants have 3 genomes: nuclear, chloroplast, and mitochondrial.
        This models just the chloroplast (smallest).
        """
        # Chloroplast genome is ~120-160 kb
        genome_size = random.randint(120000, 160000)

        # Higher GC content typical of plants
        genome = DNA.random(genome_size, gc_content=0.38, name=f"{name}_chloroplast")

        # Add some chloroplast-specific genes
        chloroplast_genes = [
            'rbcL',   # Rubisco large subunit
            'atpA', 'atpB', 'atpE', 'atpF', 'atpH', 'atpI',  # ATP synthase
            'psaA', 'psaB', 'psaC', 'psaI', 'psaJ',  # Photosystem I
            'psbA', 'psbB', 'psbC', 'psbD', 'psbE', 'psbF',  # Photosystem II
            'petA', 'petB', 'petD',  # Cytochrome
            'rpoA', 'rpoB', 'rpoC1', 'rpoC2',  # RNA polymerase
            'rRNA_16S', 'rRNA_23S', 'rRNA_4.5S', 'rRNA_5S',  # Ribosomal RNA
        ]

        genes = []
        for gene_name in chloroplast_genes:
            gene_len = random.randint(500, 2000)
            gene = Gene(
                name=gene_name,
                coding_sequence=DNA.random(gene_len, gc_content=0.38),
                function=f"Chloroplast gene: {gene_name}"
            )
            genes.append(gene)

        return cls(
            name=name,
            genome=genome,
            organism_type=OrganismType.PLANT,
            genes=genes
        )

    @classmethod
    def single_cell_eukaryote(cls, name: str = "Yeast") -> "Organism":
        """
        Create a single-cell eukaryote model (like yeast).

        S. cerevisiae has ~6000 genes in ~12 Mb genome.
        """
        # Simplified - smaller genome
        genome_size = 1000000  # 1 Mb
        genome = DNA.random(genome_size, gc_content=0.38, name=name)

        # Essential eukaryotic genes
        eukaryote_genes = [
            'ACT1',   # Actin
            'TUB1', 'TUB2',  # Tubulin
            'HIS3',   # Histidine biosynthesis
            'LEU2',   # Leucine biosynthesis
            'URA3',   # Uracil biosynthesis
            'TRP1',   # Tryptophan biosynthesis
            'ADE2',   # Adenine biosynthesis
            'CDC28',  # Cell cycle
            'SIR2',   # Silencing
            'HTA1', 'HTB1',  # Histones
            'RPO21',  # RNA polymerase II
            'TFIIA', 'TFIIB',  # Transcription factors
            'SRP54',  # Signal recognition particle
            'SEC61',  # Protein translocation
            'KAR2',   # Chaperone
            'HSP70',  # Heat shock
            'RIB1', 'RIB2',  # Ribosome biogenesis
        ]

        genes = []
        for gene_name in eukaryote_genes:
            gene_len = random.randint(500, 3000)
            gene = Gene(
                name=gene_name,
                coding_sequence=DNA.random(gene_len, gc_content=0.38),
                function=f"Essential eukaryote gene: {gene_name}"
            )
            genes.append(gene)

        return cls(
            name=name,
            genome=genome,
            organism_type=OrganismType.EUKARYOTE,
            genes=genes
        )

    @property
    def genome_size(self) -> int:
        """Genome size in base pairs."""
        return len(self.genome)

    @property
    def gene_count(self) -> int:
        """Number of genes."""
        return len(self.genes)

    def gc_content(self) -> float:
        """Genome GC content."""
        return self.genome.gc_content()

    def express_gene(self, gene_name: str) -> Optional[Protein]:
        """Express a gene (transcribe and translate)."""
        for gene in self.genes:
            if gene.name == gene_name:
                return gene.translate()
        return None

    def replicate(self, mutation_rate: float = 0.0001) -> "Organism":
        """
        Replicate the organism with mutations.

        Args:
            mutation_rate: Per-base mutation rate

        Returns:
            New organism with mutated genome
        """
        new_genome = self.genome.mutate(mutation_rate)

        # Mutate genes
        new_genes = []
        for gene in self.genes:
            new_gene = Gene(
                name=gene.name,
                coding_sequence=gene.coding_sequence.mutate(mutation_rate),
                promoter=gene.promoter,
                terminator=gene.terminator,
                function=gene.function,
                essential=gene.essential
            )
            new_genes.append(new_gene)

        return Organism(
            name=f"{self.name}_offspring",
            genome=new_genome,
            organism_type=self.organism_type,
            genes=new_genes
        )

    def stats(self) -> Dict:
        """Return organism statistics."""
        return {
            'name': self.name,
            'type': self.organism_type.value,
            'genome_size': self.genome_size,
            'gene_count': self.gene_count,
            'gc_content': self.gc_content(),
            'essential_genes': sum(1 for g in self.genes if g.essential),
        }

    def __str__(self) -> str:
        return f"Organism({self.name}, {self.organism_type.value}, {self.genome_size:,}bp, {self.gene_count} genes)"


# ============================================================================
# POPULATION SIMULATION
# ============================================================================

@dataclass
class Population:
    """
    Simulate a population of organisms evolving over time.
    """
    organisms: List[Organism] = field(default_factory=list)
    generation: int = 0

    @classmethod
    def from_organism(cls, organism: Organism, size: int = 100) -> "Population":
        """Create a population from a single organism."""
        organisms = [organism] * size  # Start with clones
        return cls(organisms)

    @property
    def size(self) -> int:
        return len(self.organisms)

    def evolve(self, generations: int = 10,
               mutation_rate: float = 0.0001,
               selection_pressure: float = 0.1) -> "Population":
        """
        Evolve the population for a number of generations.

        Args:
            generations: Number of generations to evolve
            mutation_rate: Per-base mutation rate
            selection_pressure: Fraction removed each generation

        Returns:
            Evolved population
        """
        for gen in range(generations):
            # Reproduction with mutation
            new_organisms = []
            for org in self.organisms:
                offspring = org.replicate(mutation_rate)
                new_organisms.append(offspring)

            # Selection (remove random individuals)
            if selection_pressure > 0:
                survivors = random.sample(
                    new_organisms,
                    int(len(new_organisms) * (1 - selection_pressure))
                )
                # Repopulate to original size
                while len(survivors) < len(self.organisms):
                    survivors.append(random.choice(survivors).replicate(mutation_rate))
                new_organisms = survivors

            self.organisms = new_organisms
            self.generation += 1

        return self

    def diversity(self) -> float:
        """Calculate genetic diversity (average pairwise distance)."""
        if len(self.organisms) < 2:
            return 0.0

        # Sample pairwise comparisons
        n_samples = min(100, len(self.organisms))
        distances = []

        for _ in range(n_samples):
            o1, o2 = random.sample(self.organisms, 2)
            # Compare first 1000 bases
            seq1 = o1.genome.sequence[:1000]
            seq2 = o2.genome.sequence[:1000]
            dist = sum(a != b for a, b in zip(seq1, seq2)) / len(seq1)
            distances.append(dist)

        return sum(distances) / len(distances)

    def stats(self) -> Dict:
        """Return population statistics."""
        return {
            'size': self.size,
            'generation': self.generation,
            'diversity': self.diversity(),
            'avg_gc': sum(o.gc_content() for o in self.organisms) / len(self.organisms),
        }


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def random_dna(length: int = 1000, gc_content: float = 0.5) -> DNA:
    """Generate random DNA sequence."""
    return DNA.random(length, gc_content)


def random_rna(length: int = 1000) -> RNA:
    """Generate random RNA sequence."""
    return RNA.random(length)


def random_protein(length: int = 100) -> Protein:
    """Generate random protein sequence."""
    amino_acids = 'ACDEFGHIKLMNPQRSTVWY'
    sequence = ''.join(random.choice(amino_acids) for _ in range(length))
    return Protein(sequence, "RandomProtein")


def calculate_tm(dna: DNA) -> float:
    """
    Calculate melting temperature (Tm) for DNA.

    Uses Wallace rule for short sequences (<14bp) and
    nearest-neighbor approximation for longer sequences.
    """
    seq = dna.sequence
    n = len(seq)

    if n < 14:
        # Wallace rule
        a = seq.count('A')
        t = seq.count('T')
        g = seq.count('G')
        c = seq.count('C')
        return 2 * (a + t) + 4 * (g + c)
    else:
        # Simple approximation
        gc = dna.gc_content()
        return 64.9 + 41 * (gc - 0.5)


def pcr_amplify(template: DNA, forward_primer: str, reverse_primer: str,
                cycles: int = 30) -> DNA:
    """
    Simulate PCR amplification.

    Returns amplified DNA product.
    """
    # Find primer binding sites
    fwd_pos = template.sequence.find(forward_primer)
    rev_pos = template.sequence.find(DNA(reverse_primer).reverse_complement)

    if fwd_pos == -1 or rev_pos == -1:
        raise ValueError("Primers don't bind to template")

    # Extract product
    if fwd_pos < rev_pos:
        product = template.sequence[fwd_pos:rev_pos + len(reverse_primer)]
    else:
        product = template.sequence[rev_pos:fwd_pos + len(forward_primer)]

    return DNA(product, f"PCR_product_{cycles}cycles")


# ============================================================================
# DEMO
# ============================================================================

def demo():
    """Demonstrate the DNA/RNA/Organism module."""
    print("=" * 70)
    print("DNA/RNA/ORGANISM SIMULATION DEMO")
    print("=" * 70)

    # 1. DNA operations
    print("\n[1] DNA OPERATIONS")
    print("-" * 40)
    dna = DNA.random(1000, gc_content=0.5, name="TestDNA")
    print(f"Generated: {dna}")
    print(f"GC content: {dna.gc_content():.2%}")
    print(f"Nucleotide counts: {dna.nucleotide_counts()}")
    print(f"Complement (first 20): {dna.complement[:20]}")

    # 2. Transcription and Translation
    print("\n[2] TRANSCRIPTION & TRANSLATION")
    print("-" * 40)
    rna = dna.transcribe()
    print(f"Transcribed: {rna}")

    protein = rna.translate()
    print(f"Translated: {protein}")
    print(f"Protein MW: {protein.molecular_weight():.1f} Da")
    print(f"Hydrophobicity: {protein.hydrophobicity():.2%}")

    # 3. Find ORFs
    print("\n[3] OPEN READING FRAMES")
    print("-" * 40)
    orfs = dna.find_orfs(min_length=100)
    print(f"Found {len(orfs)} ORFs:")
    for i, (start, end, frame) in enumerate(orfs[:5]):
        print(f"  ORF {i+1}: {start}-{end} ({end-start}bp) frame {frame}")

    # 4. Restriction sites
    print("\n[4] RESTRICTION ENZYME SITES")
    print("-" * 40)
    for enzyme in ['EcoRI', 'BamHI', 'HindIII']:
        sites = dna.find_restriction_sites(enzyme)
        print(f"  {enzyme}: {len(sites)} sites at {sites[:5]}{'...' if len(sites) > 5 else ''}")

    # 5. Minimal bacterium
    print("\n[5] MINIMAL BACTERIUM")
    print("-" * 40)
    bacterium = Organism.minimal_bacterium("SynthBacteria")
    print(f"Created: {bacterium}")
    print(f"Stats: {bacterium.stats()}")

    # Express a gene
    if bacterium.genes:
        first_gene = bacterium.genes[0]
        protein = first_gene.translate()
        print(f"Expressed {first_gene.name}: {protein}")

    # 6. Simple plant cell
    print("\n[6] SIMPLE PLANT CELL")
    print("-" * 40)
    plant = Organism.simple_plant_cell("SynthPlant")
    print(f"Created: {plant}")
    print(f"Stats: {plant.stats()}")

    # 7. Evolution simulation
    print("\n[7] EVOLUTION SIMULATION")
    print("-" * 40)
    pop = Population.from_organism(bacterium, size=50)
    print(f"Initial population: {pop.stats()}")

    pop.evolve(generations=10, mutation_rate=0.0001, selection_pressure=0.1)
    print(f"After 10 generations: {pop.stats()}")

    print("\n" + "=" * 70)
    print("DEMO COMPLETE!")
    print("=" * 70)


if __name__ == "__main__":
    demo()
