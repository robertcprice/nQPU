"""
Quantum Genome Assembly and CRISPR Simulation

Novel features:
1. Quantum genome assembly - use quantum optimization for contig joining
2. CRISPR simulation with quantum search for target sites
3. Quantum de novo assembly
4. Gene editing with quantum precision

Run with: python3 quantum_genome_tools.py
"""

import numpy as np
from typing import List, Tuple, Dict, Optional, Set
from dataclasses import dataclass, field
import random
from collections import defaultdict

from chem.dna_rna_organism import DNA, Gene
from bio.quantum_organism import QuantumDNA


# ============================================================================
# QUANTUM GENOME ASSEMBLY
# ============================================================================

@dataclass
class Read:
    """A sequencing read (short DNA fragment)."""
    sequence: str
    quality: List[int] = field(default_factory=list)
    position: int = -1  # Position in genome (unknown during assembly)

    @property
    def length(self) -> int:
        return len(self.sequence)

    def overlap_with(self, other: "Read", min_overlap: int = 10) -> int:
        """Find suffix-prefix overlap with another read."""
        max_overlap = min(len(self.sequence), len(other.sequence))

        for overlap in range(max_overlap, min_overlap - 1, -1):
            if self.sequence[-overlap:] == other.sequence[:overlap]:
                return overlap
        return 0


@dataclass
class Contig:
    """An assembled contig (continuous sequence)."""
    sequence: str
    reads: List[Read] = field(default_factory=list)
    coverage: float = 0.0

    @property
    def length(self) -> int:
        return len(self.sequence)


@dataclass
class QuantumGenomeAssembler:
    """
    Quantum-enhanced genome assembler.

    Uses quantum optimization for:
    1. Finding optimal read overlaps
    2. Resolving repetitive regions
    3. Contig ordering and orientation
    """
    reads: List[Read] = field(default_factory=list)
    min_overlap: int = 15
    coverage: int = 30

    @classmethod
    def simulate_reads(cls, genome: DNA, read_length: int = 100,
                      coverage: int = 30, error_rate: float = 0.01) -> "QuantumGenomeAssembler":
        """Simulate sequencing reads from a genome."""
        assembler = cls(coverage=coverage)

        n_reads = int(len(genome.sequence) * coverage / read_length)

        for i in range(n_reads):
            # Random position
            start = random.randint(0, len(genome.sequence) - read_length)
            end = start + read_length

            # Extract read with possible errors
            read_seq = list(genome.sequence[start:end])
            for j in range(len(read_seq)):
                if random.random() < error_rate:
                    read_seq[j] = random.choice(['A', 'T', 'G', 'C'])

            # Quality scores (higher = better)
            quality = [random.randint(30, 40) for _ in range(read_length)]

            read = Read(
                sequence=''.join(read_seq),
                quality=quality,
                position=start  # Hidden during real assembly
            )
            assembler.reads.append(read)

        return assembler

    def build_overlap_graph(self) -> Dict[int, List[Tuple[int, int]]]:
        """Build overlap graph between reads."""
        graph = defaultdict(list)

        for i, read1 in enumerate(self.reads):
            for j, read2 in enumerate(self.reads):
                if i != j:
                    overlap = read1.overlap_with(read2, self.min_overlap)
                    if overlap > 0:
                        graph[i].append((j, overlap))

        return graph

    def classical_assembly(self) -> List[Contig]:
        """Classical greedy assembly algorithm."""
        print("Running classical greedy assembly...")

        # Build overlap graph
        graph = self.build_overlap_graph()

        # Greedy assembly
        used = set()
        contigs = []

        for i, read in enumerate(self.reads):
            if i in used:
                continue

            # Start new contig
            contig_seq = read.sequence
            contig_reads = [read]
            used.add(i)

            # Extend right
            current = i
            while True:
                # Find best overlap
                best_j, best_overlap = -1, 0
                for j, overlap in graph.get(current, []):
                    if j not in used and overlap > best_overlap:
                        best_j, best_overlap = j, overlap

                if best_j < 0:
                    break

                # Extend contig
                contig_seq += self.reads[best_j].sequence[best_overlap:]
                contig_reads.append(self.reads[best_j])
                used.add(best_j)
                current = best_j

            if len(contig_seq) > 100:  # Minimum contig length
                contigs.append(Contig(
                    sequence=contig_seq,
                    reads=contig_reads,
                    coverage=len(contig_reads) * 100 / len(contig_seq)
                ))

        # Sort by length
        contigs.sort(key=lambda c: c.length, reverse=True)
        return contigs

    def quantum_assembly(self) -> List[Contig]:
        """
        Quantum-enhanced assembly.

        Uses quantum superposition to explore multiple assembly paths
        simultaneously, then "measures" the best assembly.
        """
        print("Running quantum-enhanced assembly...")

        # Build overlap graph
        graph = self.build_overlap_graph()

        # Quantum-inspired: explore multiple paths in superposition
        # Score each path by overlap quality and consistency

        used = set()
        contigs = []

        for i, read in enumerate(self.reads):
            if i in used:
                continue

            # Quantum path exploration: try multiple extensions
            best_path = self._quantum_path_search(i, graph, used)

            if best_path:
                # Build contig from best path
                contig_seq = self.reads[best_path[0]].sequence
                contig_reads = [self.reads[best_path[0]]]

                for k in range(1, len(best_path)):
                    j = best_path[k]
                    overlap = self.reads[best_path[k-1]].overlap_with(
                        self.reads[j], self.min_overlap
                    )
                    contig_seq += self.reads[j].sequence[overlap:]
                    contig_reads.append(self.reads[j])
                    used.add(j)

                used.add(best_path[0])

                if len(contig_seq) > 100:
                    contigs.append(Contig(
                        sequence=contig_seq,
                        reads=contig_reads,
                        coverage=len(contig_reads) * 100 / len(contig_seq)
                    ))

        contigs.sort(key=lambda c: c.length, reverse=True)
        return contigs

    def _quantum_path_search(self, start: int, graph: Dict, used: Set,
                             max_depth: int = 50) -> Optional[List[int]]:
        """
        Search for best assembly path using quantum-inspired exploration.

        Instead of greedy, explores multiple paths and returns best.
        """
        # Beam search with quantum superposition analogy
        beam_width = 5  # Number of paths to explore in "superposition"

        # Each beam entry: (path, score)
        beams = [([start], 0.0)]

        for depth in range(max_depth):
            new_beams = []

            for path, score in beams:
                current = path[-1]

                # Get candidates
                candidates = [(j, o) for j, o in graph.get(current, [])
                             if j not in used and j not in path]

                if not candidates:
                    new_beams.append((path, score))
                    continue

                # Sort by overlap and take top candidates
                candidates.sort(key=lambda x: x[1], reverse=True)
                candidates = candidates[:beam_width]

                for j, overlap in candidates:
                    new_path = path + [j]
                    # Score combines overlap and path consistency
                    new_score = score + overlap + self._path_consistency_score(new_path)
                    new_beams.append((new_path, new_score))

            # Keep top beams (quantum measurement analogy)
            new_beams.sort(key=lambda x: x[1], reverse=True)
            beams = new_beams[:beam_width]

            if not beams:
                break

        # Return best path (measurement outcome)
        if beams:
            return beams[0][0]
        return None

    def _path_consistency_score(self, path: List[int]) -> float:
        """Score path consistency (GC content variance, etc.)."""
        if len(path) < 2:
            return 0.0

        # Build partial sequence
        seq = self.reads[path[0]].sequence
        for k in range(1, len(path)):
            overlap = self.reads[path[k-1]].overlap_with(self.reads[path[k]], self.min_overlap)
            seq += self.reads[path[k]].sequence[overlap:]

        # Score based on realistic GC content (should be 0.3-0.7)
        gc = (seq.count('G') + seq.count('C')) / len(seq) if seq else 0.5
        gc_penalty = abs(gc - 0.5) * 10

        return -gc_penalty  # Lower penalty = higher score

    def scaffold_contigs(self, contigs: List[Contig]) -> List[Contig]:
        """
        Order and orient contigs into scaffolds.

        Uses read pairs to determine contig ordering.
        """
        if len(contigs) < 2:
            return contigs

        # Simple scaffolding by size (in reality would use paired-end reads)
        # This is a placeholder for quantum-optimized scaffolding

        scaffolds = []
        for contig in contigs:
            scaffolds.append(contig)

        return scaffolds


# ============================================================================
# CRISPR SIMULATION
# ============================================================================

@dataclass
class CRISPRGuide:
    """A CRISPR guide RNA."""
    sequence: str  # 20bp guide sequence
    pam: str = "NGG"  # PAM sequence (NGG for SpCas9)

    @property
    def length(self) -> int:
        return len(self.sequence)

    def find_targets(self, genome: DNA) -> List[Tuple[int, str, str]]:
        """
        Find all potential target sites in a genome.

        Returns list of (position, target_sequence, strand) tuples.
        """
        targets = []
        seq = genome.sequence

        # Search forward strand
        for i in range(len(seq) - 22):
            if seq[i+20:i+23] == "AGG" or seq[i+20:i+23] == "CGG" or \
               seq[i+20:i+23] == "GGG" or seq[i+20:i+23] == "TGG":
                target = seq[i:i+20]
                targets.append((i, target, '+'))

        # Search reverse strand
        complement = {'A': 'T', 'T': 'A', 'G': 'C', 'C': 'G'}
        rev_seq = ''.join(complement[b] for b in seq[::-1])

        for i in range(len(rev_seq) - 22):
            if rev_seq[i+20:i+23] == "AGG" or rev_seq[i+20:i+23] == "CGG" or \
               rev_seq[i+20:i+23] == "GGG" or rev_seq[i+20:i+23] == "TGG":
                target = rev_seq[i:i+20]
                # Convert position back to forward strand coordinates
                fwd_pos = len(seq) - i - 23
                targets.append((fwd_pos, target, '-'))

        return targets

    def specificity_score(self, genome: DNA) -> float:
        """
        Calculate guide specificity (off-target potential).

        Higher score = more specific = fewer off-targets.
        """
        targets = self.find_targets(genome)

        if len(targets) <= 1:
            return 1.0  # Perfect specificity

        # Check for similar sequences (off-targets)
        off_targets = 0
        for pos, target, strand in targets[1:]:  # Skip first (intended target)
            # Count mismatches with guide
            mismatches = sum(1 for a, b in zip(self.sequence, target) if a != b)
            if mismatches <= 3:  # Potential off-target
                off_targets += 1

        # Specificity score
        specificity = 1.0 / (1 + off_targets)
        return specificity


@dataclass
class CRISPRSimulation:
    """
    Simulate CRISPR gene editing.

    Features:
    1. Guide RNA design
    2. Target site finding
    3. Off-target prediction
    4. Quantum search for optimal guides
    """
    genome: DNA

    def design_guides(self, gene: Gene, n_guides: int = 5) -> List[CRISPRGuide]:
        """Design guide RNAs for a specific gene."""
        gene_seq = gene.coding_sequence.sequence

        guides = []
        for i in range(len(gene_seq) - 22):
            # Check for PAM
            if gene_seq[i+20:i+23] in ["AGG", "CGG", "GGG", "TGG"]:
                guide_seq = gene_seq[i:i+20]
                guide = CRISPRGuide(sequence=guide_seq)

                # Check specificity
                specificity = guide.specificity_score(self.genome)
                if specificity >= 0.5:  # Minimum specificity threshold (0.5 = 1 off-target)
                    guides.append(guide)

        # Sort by specificity and return top n
        guides.sort(key=lambda g: g.specificity_score(self.genome), reverse=True)
        return guides[:n_guides]

    def quantum_guide_search(self, gene: Gene, n_guides: int = 5) -> List[CRISPRGuide]:
        """
        Quantum-inspired search for optimal guide RNAs.

        Explores guide space in superposition, returns best candidates.
        """
        # Get all possible guides
        all_guides = self.design_guides(gene, n_guides=100)

        if len(all_guides) <= n_guides:
            return all_guides

        # Quantum scoring: combine multiple criteria
        scored_guides = []
        for guide in all_guides:
            # Specificity (most important)
            specificity = guide.specificity_score(self.genome)

            # GC content (40-60% is optimal)
            gc = (guide.sequence.count('G') + guide.sequence.count('C')) / 20
            gc_score = 1.0 - abs(gc - 0.5) * 2

            # Position in gene (early is better for knockout)
            gene_seq = gene.coding_sequence.sequence
            try:
                pos = gene_seq.index(guide.sequence)
                pos_score = 1.0 - (pos / len(gene_seq))
            except ValueError:
                pos_score = 0.5

            # Combined quantum score (superposition of criteria)
            quantum_score = (
                specificity * 0.5 +
                gc_score * 0.3 +
                pos_score * 0.2
            )

            scored_guides.append((guide, quantum_score))

        # "Measure" the best guides
        scored_guides.sort(key=lambda x: x[1], reverse=True)
        return [g for g, s in scored_guides[:n_guides]]

    def simulate_edit(self, guide: CRISPRGuide, position: int) -> Dict:
        """
        Simulate a CRISPR edit at the given position.

        Returns edit outcome with probabilities.
        """
        outcomes = {
            'deletion': 0.6,     # 60% chance of deletion
            'insertion': 0.2,    # 20% chance of insertion
            'substitution': 0.1, # 10% chance of substitution
            'no_edit': 0.1,      # 10% chance of no edit
        }

        # Simulate specific outcome
        rand = random.random()
        if rand < 0.6:
            # Deletion (1-20 bp)
            del_len = random.randint(1, 20)
            outcome_type = 'deletion'
            detail = f"-{del_len}bp"
        elif rand < 0.8:
            # Insertion (1-5 bp)
            ins_len = random.randint(1, 5)
            outcome_type = 'insertion'
            detail = f"+{ins_len}bp"
        elif rand < 0.9:
            # Substitution
            outcome_type = 'substitution'
            detail = "SNP"
        else:
            outcome_type = 'no_edit'
            detail = "no change"

        return {
            'type': outcome_type,
            'detail': detail,
            'position': position,
            'guide': guide.sequence,
        }


# ============================================================================
# QUANTUM DE NOVO ASSEMBLY
# ============================================================================

@dataclass
class DeNovoAssembler:
    """
    De novo genome assembly using quantum optimization.

    Assembles a genome from scratch without a reference.
    """
    reads: List[Read]
    kmer_size: int = 31

    def build_de_bruijn_graph(self) -> Dict[str, List[str]]:
        """Build de Bruijn graph from k-mers."""
        graph = defaultdict(list)

        for read in self.reads:
            for i in range(len(read.sequence) - self.kmer_size):
                kmer = read.sequence[i:i+self.kmer_size]
                next_kmer = read.sequence[i+1:i+1+self.kmer_size]

                if len(next_kmer) == self.kmer_size:
                    graph[kmer].append(next_kmer)

        return graph

    def quantum_eulerian_path(self, graph: Dict[str, List[str]]) -> List[str]:
        """
        Find Eulerian path in de Bruijn graph using quantum optimization.

        Classical: Hierholzer's algorithm
        Quantum: Explore all paths in superposition, measure best
        """
        # Find start node (more outgoing than incoming edges)
        in_degree = defaultdict(int)
        out_degree = defaultdict(int)

        for node, edges in graph.items():
            out_degree[node] = len(edges)
            for edge in edges:
                in_degree[edge] += 1

        start = None
        for node in set(in_degree.keys()) | set(out_degree.keys()):
            if out_degree[node] > in_degree[node]:
                start = node
                break

        if start is None:
            start = next(iter(graph.keys()))

        # Hierholzer's algorithm (quantum-enhanced path selection)
        stack = [start]
        path = []

        while stack:
            current = stack[-1]
            if graph[current]:
                # Quantum-inspired: choose edge based on k-mer quality
                next_node = self._quantum_edge_selection(current, graph[current])
                graph[current].remove(next_node)
                stack.append(next_node)
            else:
                path.append(stack.pop())

        return path[::-1]

    def _quantum_edge_selection(self, current: str, candidates: List[str]) -> str:
        """Select next edge using quantum-inspired scoring."""
        if len(candidates) == 1:
            return candidates[0]

        # Score by GC content (prefer balanced)
        scores = []
        for candidate in candidates:
            gc = (candidate.count('G') + candidate.count('C')) / len(candidate)
            balance = 1.0 - abs(gc - 0.5) * 2
            scores.append((candidate, balance))

        # Add quantum randomness
        for i, (cand, score) in enumerate(scores):
            scores[i] = (cand, score + random.random() * 0.1)

        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[0][0]

    def assemble(self) -> Contig:
        """Perform de novo assembly."""
        print(f"Assembling {len(self.reads)} reads with k={self.kmer_size}...")

        # Build graph
        graph = self.build_de_bruijn_graph()
        print(f"  Built de Bruijn graph with {len(graph)} nodes")

        # Find path
        path = self.quantum_eulerian_path(graph)

        # Reconstruct sequence
        if not path:
            return Contig(sequence="", coverage=0.0)

        sequence = path[0]
        for kmer in path[1:]:
            sequence += kmer[-1]

        print(f"  Assembled contig: {len(sequence)} bp")
        return Contig(sequence=sequence, coverage=len(self.reads) * 100 / max(len(sequence), 1))


# ============================================================================
# DEMO
# ============================================================================

def demo():
    """Demonstrate quantum genome tools."""
    print("=" * 70)
    print("QUANTUM GENOME ASSEMBLY & CRISPR SIMULATION")
    print("=" * 70)

    # 1. Generate test genome with PAM sites for CRISPR
    print("\n[1] GENERATING TEST GENOME")
    print("-" * 40)

    # Create genome with embedded PAM sites (NGG) for CRISPR testing
    random.seed(42)  # For reproducibility
    base_seq = []
    for i in range(100):  # 100 chunks of ~100bp
        chunk = ''.join(random.choices(['A', 'T', 'G', 'C'], k=97))
        # Insert PAM site (NGG) every 100 bp
        pam = random.choice(['A', 'T', 'G', 'C']) + 'GG'
        chunk = chunk[:20] + pam + chunk[20:]  # PAM at position 20-22
        base_seq.append(chunk)

    genome_seq = ''.join(base_seq)
    genome = DNA(genome_seq, "TestGenome")
    print(f"  Generated genome: {len(genome)} bp")

    # Gene with PAM sites embedded
    gene_seq = genome_seq[1000:2000]
    gene = Gene(
        name="TestGene",
        coding_sequence=DNA(gene_seq, "TestGene_CDS"),
        function="Test gene for CRISPR"
    )
    print(f"  Created gene: {gene.name} ({len(gene.coding_sequence)} bp)")

    # 2. Simulate reads and assemble
    print("\n[2] GENOME ASSEMBLY")
    print("-" * 40)
    assembler = QuantumGenomeAssembler.simulate_reads(
        genome, read_length=100, coverage=30, error_rate=0.01
    )
    print(f"  Simulated {len(assembler.reads)} reads")

    # Classical assembly
    classical_contigs = assembler.classical_assembly()
    print(f"  Classical assembly: {len(classical_contigs)} contigs")
    if classical_contigs:
        print(f"    Largest: {classical_contigs[0].length} bp")

    # Quantum assembly
    quantum_contigs = assembler.quantum_assembly()
    print(f"  Quantum assembly: {len(quantum_contigs)} contigs")
    if quantum_contigs:
        print(f"    Largest: {quantum_contigs[0].length} bp")

    # 3. CRISPR guide design
    print("\n[3] CRISPR GUIDE DESIGN")
    print("-" * 40)

    sim = CRISPRSimulation(genome)

    # Classical guide design
    classical_guides = sim.design_guides(gene, n_guides=3)
    print(f"  Classical design: {len(classical_guides)} guides")
    for i, guide in enumerate(classical_guides):
        spec = guide.specificity_score(genome)
        print(f"    Guide {i+1}: {guide.sequence[:20]}... (specificity: {spec:.2f})")

    # Quantum guide design
    quantum_guides = sim.quantum_guide_search(gene, n_guides=3)
    print(f"  Quantum design: {len(quantum_guides)} guides")
    for i, guide in enumerate(quantum_guides):
        spec = guide.specificity_score(genome)
        print(f"    Guide {i+1}: {guide.sequence[:20]}... (specificity: {spec:.2f})")

    # 4. Simulate CRISPR edit
    print("\n[4] CRISPR EDIT SIMULATION")
    print("-" * 40)

    if quantum_guides:
        guide = quantum_guides[0]
        edit = sim.simulate_edit(guide, position=1500)
        print(f"  Guide: {guide.sequence}")
        print(f"  Outcome: {edit['type']} ({edit['detail']})")

    # 5. De novo assembly
    print("\n[5] DE NOVO ASSEMBLY")
    print("-" * 40)

    de_novo = DeNovoAssembler(reads=assembler.reads[:500], kmer_size=31)
    contig = de_novo.assemble()
    print(f"  De novo contig: {contig.length} bp")
    print(f"  Coverage: {contig.coverage:.1f}x")

    print("\n" + "=" * 70)
    print("QUANTUM GENOME TOOLS DEMO COMPLETE!")
    print("=" * 70)


if __name__ == "__main__":
    demo()
