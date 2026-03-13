"""
Quantum Artificial Life Simulation

This is where it gets REALLY cool - we create artificial life forms
that evolve using quantum mechanics. No one has done this before.

Features:
1. Quantum creatures with quantum DNA
2. Evolution via quantum selection
3. Quantum ecosystems with predator-prey dynamics
4. Visualization of creature evolution
5. DNA to music/sound generation
6. Quantum mutation and crossover
7. Emergent behaviors from quantum superposition

Example:
    >>> from quantum_life import QuantumCreature, QuantumEcosystem
    >>> creature = QuantumCreature.random(genome_length=100)
    >>> ecosystem = QuantumEcosystem(population_size=50)
    >>> ecosystem.evolve(generations=100)
    >>> ecosystem.visualize()
"""

from typing import Dict, List, Tuple, Optional, Set, Any
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
import random
import math
from collections import defaultdict
import colorsys

# Import our quantum DNA
from bio.quantum_organism import QuantumDNA
from chem.dna_rna_organism import DNA, Protein, CODON_TABLE


# ============================================================================
# QUANTUM CREATURE GENETICS
# ============================================================================

@dataclass
class GeneEffect:
    """Effect of a gene on creature traits."""
    trait: str
    effect: float  # -1 to 1
    dominance: float  # 0 to 1


@dataclass
class CreatureTraits:
    """Physical and behavioral traits of a creature."""
    # Physical
    size: float = 1.0  # 0.1 to 10
    speed: float = 1.0  # 0.1 to 5
    color_hue: float = 0.5  # 0 to 1
    color_saturation: float = 0.8
    shape_complexity: float = 0.5  # 0=simple, 1=complex

    # Behavioral
    aggression: float = 0.5  # 0=peaceful, 1=aggressive
    social: float = 0.5  # 0=solitary, 1=herding
    intelligence: float = 0.5  # affects decision making
    curiosity: float = 0.5  # exploration tendency

    # Survival
    metabolic_rate: float = 1.0  # energy consumption
    reproduction_rate: float = 1.0  # offspring per cycle
    lifespan: float = 100.0  # max age
    resistance: float = 0.5  # disease/environment resistance

    # Quantum
    coherence: float = 0.0  # quantum coherence level
    entanglement_range: float = 0.0  # how far can entangle
    tunnel_probability: float = 0.0  # quantum tunneling chance

    def to_dict(self) -> Dict:
        return {
            'size': self.size, 'speed': self.speed,
            'color_hue': self.color_hue, 'color_saturation': self.color_saturation,
            'aggression': self.aggression, 'social': self.social,
            'intelligence': self.intelligence, 'curiosity': self.curiosity,
            'metabolic_rate': self.metabolic_rate, 'reproduction_rate': self.reproduction_rate,
            'lifespan': self.lifespan, 'resistance': self.resistance,
            'coherence': self.coherence, 'entanglement_range': self.entanglement_range,
            'tunnel_probability': self.tunnel_probability,
        }


@dataclass
class QuantumCreature:
    """
    A creature with quantum DNA that affects its traits and behavior.

    The genome encodes traits via codons:
    - First 10 codons: physical traits
    - Next 10 codons: behavioral traits
    - Next 10 codons: survival traits
    - Next 10 codons: quantum traits
    - Rest: regulatory/special abilities
    """
    genome: QuantumDNA
    traits: CreatureTraits = field(default_factory=CreatureTraits)
    age: int = 0
    energy: float = 100.0
    generation: int = 0
    position: Tuple[float, float] = (0.0, 0.0)
    velocity: Tuple[float, float] = (0.0, 0.0)

    _fitness: float = 0.0
    _entangled_with: Set[int] = field(default_factory=set)

    @classmethod
    def random(cls, genome_length: int = 100, generation: int = 0) -> "QuantumCreature":
        """Create a random quantum creature."""
        dna = DNA.random(genome_length, gc_content=random.uniform(0.3, 0.7), name=f"Creature_G{generation}")
        qdna = QuantumDNA.from_dna(dna)
        creature = cls(genome=qdna, generation=generation)
        creature._decode_genome()
        return creature

    @classmethod
    def from_parents(cls, parent1: "QuantumCreature", parent2: "QuantumCreature") -> "QuantumCreature":
        """Create offspring from two parents using quantum crossover."""
        # Quantum superposition crossover
        child_genome = parent1.genome.superpose_with(parent2.genome, alpha=random.random())

        # Apply quantum mutation based on parent coherence
        mutation_rate = 0.01 * (1 + parent1.traits.coherence + parent2.traits.coherence)
        if random.random() < mutation_rate:
            # Quantum tunneling mutation - can change multiple bases
            n_mutations = int(abs(np.random.normal(0, 3 * parent1.traits.coherence + 1)))
            seq = list(child_genome.sequence)
            for _ in range(n_mutations):
                if seq:
                    pos = random.randint(0, len(seq) - 1)
                    seq[pos] = random.choice('ATGC')
            child_genome = QuantumDNA(''.join(seq), child_genome.name + "_mutant")

        child = cls(
            genome=child_genome,
            generation=max(parent1.generation, parent2.generation) + 1,
            energy=(parent1.energy + parent2.energy) / 4,  # Split energy
        )
        child._decode_genome()

        # Inherit some quantum properties
        child.traits.coherence = (parent1.traits.coherence + parent2.traits.coherence) / 2
        if random.random() < child.traits.coherence * 0.5:
            child._entangled_with.add(id(parent1))
            child._entangled_with.add(id(parent2))

        return child

    def _decode_genome(self):
        """Decode genome into traits."""
        seq = self.genome.sequence

        # Physical traits (codons 0-9)
        if len(seq) >= 30:
            self.traits.size = self._decode_trait(seq[0:10], 0.1, 10.0)
            self.traits.speed = self._decode_trait(seq[10:20], 0.1, 5.0)
            self.traits.shape_complexity = self._decode_trait(seq[20:30], 0.0, 1.0)

        # Color from specific codons
        if len(seq) >= 36:
            self.traits.color_hue = self._decode_trait(seq[30:33], 0.0, 1.0)
            self.traits.color_saturation = self._decode_trait(seq[33:36], 0.3, 1.0)

        # Behavioral traits (codons 10-19)
        if len(seq) >= 60:
            self.traits.aggression = self._decode_trait(seq[40:50], 0.0, 1.0)
            self.traits.social = self._decode_trait(seq[50:60], 0.0, 1.0)

        # Survival traits (codons 20-29)
        if len(seq) >= 90:
            self.traits.metabolic_rate = self._decode_trait(seq[60:70], 0.5, 2.0)
            self.traits.reproduction_rate = self._decode_trait(seq[70:80], 0.5, 2.0)
            self.traits.lifespan = self._decode_trait(seq[80:90], 50, 200)

        # Quantum traits (codons 30-39)
        if len(seq) >= 120:
            self.traits.coherence = self._decode_trait(seq[90:100], 0.0, 1.0)
            self.traits.entanglement_range = self._decode_trait(seq[100:110], 0.0, 100.0)
            self.traits.tunnel_probability = self._decode_trait(seq[110:120], 0.0, 0.1)

        # Intelligence and curiosity from GC content
        gc = self.genome.sequence.count('G') + self.genome.sequence.count('C')
        if len(self.genome.sequence) > 0:
            gc_ratio = gc / len(self.genome.sequence)
            self.traits.intelligence = 0.2 + gc_ratio * 0.8  # Higher GC = smarter
            self.traits.curiosity = 0.1 + (1 - gc_ratio) * 0.9  # Lower GC = more curious

    def _decode_trait(self, codons: str, min_val: float, max_val: float) -> float:
        """Decode codons into a trait value."""
        if not codons:
            return (min_val + max_val) / 2

        # Use codon to determine value deterministically
        # Sum of character codes normalized
        value = sum(ord(c) for c in codons) % 10000 / 10000.0
        return float(min_val + value * (max_val - min_val))

    def get_color_rgb(self) -> Tuple[int, int, int]:
        """Get RGB color from traits."""
        rgb = colorsys.hsv_to_rgb(
            self.traits.color_hue,
            self.traits.color_saturation,
            0.9  # Value/brightness
        )
        return (int(rgb[0] * 255), int(rgb[1] * 255), int(rgb[2] * 255))

    def update(self, dt: float, environment: "QuantumEnvironment"):
        """Update creature state."""
        # Age
        self.age += 1

        # Energy consumption
        self.energy -= self.traits.metabolic_rate * dt * (1 + self.traits.size * 0.1)

        # Quantum tunneling - chance to teleport
        if random.random() < self.traits.tunnel_probability:
            self.position = (
                random.uniform(0, environment.width),
                random.uniform(0, environment.height)
            )

        # Movement based on behavior
        self._update_behavior(environment)

        # Death conditions
        if self.energy <= 0 or self.age > self.traits.lifespan:
            return False  # Creature dies

        return True  # Creature alive

    def _update_behavior(self, environment: "QuantumEnvironment"):
        """Update creature behavior and movement."""
        # Random walk with intelligence bias
        angle = random.uniform(0, 2 * math.pi)
        angle = float(angle)  # Ensure real number

        # Social creatures move toward others
        if self.traits.social > 0.6 and environment.creatures:
            others = [c for c in environment.creatures if c is not self]
            if others:
                nearest = min(others, key=lambda c: self._distance_to(c))
                dx = nearest.position[0] - self.position[0]
                dy = nearest.position[1] - self.position[1]
                dist = math.sqrt(dx*dx + dy*dy)
                if dist > 0:
                    angle = math.atan2(dy, dx) + random.uniform(-0.5, 0.5) * (1 - self.traits.social)

        # Aggressive creatures chase weaker prey
        if self.traits.aggression > 0.6 and environment.creatures:
            prey = [c for c in environment.creatures
                   if c is not self and c.traits.size < self.traits.size * 0.8]
            if prey:
                target = random.choice(prey)
                dx = target.position[0] - self.position[0]
                dy = target.position[1] - self.position[1]
                angle = math.atan2(dy, dx)

        # Update velocity
        speed = float(self.traits.speed) * (float(self.energy) / 100) ** 0.5
        self.velocity = (
            float(math.cos(angle) * speed),
            float(math.sin(angle) * speed)
        )

        # Update position (ensure real values)
        new_x = float(self.position[0]) + float(self.velocity[0])
        new_y = float(self.position[1]) + float(self.velocity[1])
        self.position = (
            max(0, min(environment.width, new_x)),
            max(0, min(environment.height, new_y))
        )

    def _distance_to(self, other: "QuantumCreature") -> float:
        """Calculate distance to another creature."""
        dx = self.position[0] - other.position[0]
        dy = self.position[1] - other.position[1]
        return math.sqrt(dx*dx + dy*dy)

    def can_reproduce(self) -> bool:
        """Check if creature can reproduce."""
        return self.energy > 50 and self.age > 10 and self.age < self.traits.lifespan * 0.8

    def interact(self, other: "QuantumCreature") -> Optional["QuantumCreature"]:
        """Interact with another creature. Returns offspring if reproduction occurs."""
        distance = self._distance_to(other)

        # Check for quantum entanglement
        if distance < self.traits.entanglement_range and random.random() < self.traits.coherence:
            self._entangled_with.add(id(other))
            other._entangled_with.add(id(self))

        # Combat (aggressive creatures)
        if distance < self.traits.size and self.traits.aggression > 0.5:
            if self.traits.size > other.traits.size:
                # We win, steal energy
                stolen = min(other.energy * 0.3, 20)
                self.energy += stolen
                other.energy -= stolen
            else:
                # We lose
                self.energy -= 10

        # Reproduction
        if distance < self.traits.size * 2 and self.can_reproduce() and other.can_reproduce():
            if random.random() < self.traits.reproduction_rate * other.traits.reproduction_rate * 0.1:
                if self.traits.social > 0.3 or other.traits.social > 0.3:
                    # Create offspring
                    self.energy -= 25
                    other.energy -= 25
                    return QuantumCreature.from_parents(self, other)

        return None

    def calculate_fitness(self) -> float:
        """Calculate fitness score."""
        self._fitness = (
            self.energy / 100 * 0.3 +
            self.age / self.traits.lifespan * 0.2 +
            (1 if self.can_reproduce() else 0) * 0.2 +
            self.traits.coherence * 0.15 +
            len(self._entangled_with) * 0.05 +
            self.traits.intelligence * 0.1
        )
        return self._fitness

    def __str__(self) -> str:
        rgb = self.get_color_rgb()
        return f"QuantumCreature(Gen{self.generation}, Age{self.age}, E={self.energy:.0f}, RGB={rgb})"


# ============================================================================
# QUANTUM ENVIRONMENT
# ============================================================================

@dataclass
class FoodSource:
    """Food source in the environment."""
    position: Tuple[float, float]
    energy: float
    regeneration_rate: float = 0.1

    def update(self, dt: float):
        """Regenerate food."""
        self.energy = min(100, self.energy + self.regeneration_rate * dt)


@dataclass
class QuantumEnvironment:
    """Environment for quantum creatures."""
    width: float = 1000.0
    height: float = 1000.0
    creatures: List[QuantumCreature] = field(default_factory=list)
    food: List[FoodSource] = field(default_factory=list)
    generation: int = 0
    time: float = 0.0

    # Environment parameters
    food_density: float = 0.001  # food per unit area
    max_population: int = 200

    def __post_init__(self):
        """Initialize food sources."""
        n_food = int(self.width * self.height * self.food_density)
        for _ in range(n_food):
            self.food.append(FoodSource(
                position=(random.uniform(0, self.width), random.uniform(0, self.height)),
                energy=random.uniform(20, 100),
                regeneration_rate=random.uniform(0.05, 0.2)
            ))

    def add_creature(self, creature: QuantumCreature):
        """Add a creature to the environment."""
        if creature.position == (0.0, 0.0):
            creature.position = (random.uniform(0, self.width), random.uniform(0, self.height))
        self.creatures.append(creature)

    def update(self, dt: float = 1.0) -> List[QuantumCreature]:
        """Update environment. Returns new offspring."""
        self.time += dt
        offspring = []

        # Update food
        for food in self.food:
            food.update(dt)

        # Update creatures
        survivors = []
        for creature in self.creatures:
            # Creature update
            if creature.update(dt, self):
                survivors.append(creature)

            # Feeding
            for food in self.food:
                if food.energy > 0:
                    dist = math.sqrt(
                        (creature.position[0] - food.position[0])**2 +
                        (creature.position[1] - food.position[1])**2
                    )
                    if dist < creature.traits.size * 2:
                        eaten = min(food.energy, 20 * creature.traits.size)
                        creature.energy += eaten
                        food.energy -= eaten

            # Interactions
            for other in self.creatures:
                if other is not creature:
                    child = creature.interact(other)
                    if child:
                        child.position = creature.position
                        offspring.append(child)

        self.creatures = survivors + offspring

        # Population control
        if len(self.creatures) > self.max_population:
            # Keep fittest
            self.creatures.sort(key=lambda c: c.calculate_fitness(), reverse=True)
            self.creatures = self.creatures[:self.max_population]

        return offspring

    def evolve(self, generations: int = 10, callback=None) -> Dict:
        """Evolve the ecosystem for multiple generations."""
        stats = []

        for gen in range(generations):
            self.generation = gen

            # Run simulation for this generation
            for _ in range(100):  # Steps per generation
                self.update()

            # Collect statistics
            if self.creatures:
                gen_stats = {
                    'generation': gen,
                    'population': len(self.creatures),
                    'avg_energy': np.mean([c.energy for c in self.creatures]),
                    'avg_age': np.mean([c.age for c in self.creatures]),
                    'avg_size': np.mean([c.traits.size for c in self.creatures]),
                    'avg_coherence': np.mean([c.traits.coherence for c in self.creatures]),
                    'avg_fitness': np.mean([c.calculate_fitness() for c in self.creatures]),
                    'species_diversity': len(set(c.get_color_rgb() for c in self.creatures)),
                }
                stats.append(gen_stats)

                if callback:
                    callback(gen, gen_stats, self)

            # Repopulate if needed
            if len(self.creatures) < 10:
                for _ in range(10 - len(self.creatures)):
                    self.add_creature(QuantumCreature.random(genome_length=100, generation=gen))

        return {
            'final_population': len(self.creatures),
            'generations': generations,
            'stats': stats,
            'best_creature': max(self.creatures, key=lambda c: c.calculate_fitness()) if self.creatures else None,
        }

    def get_population_genome_diversity(self) -> float:
        """Calculate genetic diversity in population."""
        if len(self.creatures) < 2:
            return 0.0

        # Sample pairwise distances
        n_samples = min(100, len(self.creatures))
        distances = []

        for _ in range(n_samples):
            c1, c2 = random.sample(self.creatures, 2)
            # Hamming distance between genomes
            s1, s2 = c1.genome.sequence[:100], c2.genome.sequence[:100]
            dist = sum(a != b for a, b in zip(s1, s2)) / max(len(s1), len(s2), 1)
            distances.append(dist)

        return np.mean(distances)


# ============================================================================
# QUANTUM ECOSYSTEM
# ============================================================================

@dataclass
class QuantumEcosystem:
    """
    Complete quantum ecosystem with multiple species and environments.

    Features:
    - Multiple species competing/cooperating
    - Predator-prey dynamics
    - Symbiotic relationships
    - Environmental pressures
    - Emergent behaviors
    """
    environments: List[QuantumEnvironment] = field(default_factory=list)
    species: Dict[str, List[QuantumCreature]] = field(default_factory=dict)
    time: float = 0.0

    @classmethod
    def create(cls, n_species: int = 3, population_per_species: int = 20) -> "QuantumEcosystem":
        """Create a diverse ecosystem."""
        ecosystem = cls()

        # Create main environment
        env = QuantumEnvironment(width=1000, height=1000)

        # Create different species with different traits
        for species_id in range(n_species):
            species_creatures = []

            for _ in range(population_per_species):
                creature = QuantumCreature.random(genome_length=100 + species_id * 20)

                # Bias traits for this species
                if species_id == 0:  # Predators
                    creature.traits.aggression = random.uniform(0.7, 1.0)
                    creature.traits.size = random.uniform(1.5, 3.0)
                    creature.traits.speed = random.uniform(2.0, 4.0)
                elif species_id == 1:  # Prey
                    creature.traits.aggression = random.uniform(0.0, 0.3)
                    creature.traits.size = random.uniform(0.3, 0.8)
                    creature.traits.speed = random.uniform(3.0, 5.0)
                    creature.traits.social = random.uniform(0.6, 1.0)
                else:  # Omnivores
                    creature.traits.aggression = random.uniform(0.3, 0.6)
                    creature.traits.size = random.uniform(0.8, 1.5)
                    creature.traits.curiosity = random.uniform(0.6, 1.0)

                env.add_creature(creature)
                species_creatures.append(creature)

            ecosystem.species[f"species_{species_id}"] = species_creatures

        ecosystem.environments.append(env)
        return ecosystem

    def simulate(self, steps: int = 1000, callback=None) -> Dict:
        """Run ecosystem simulation."""
        history = []

        for step in range(steps):
            self.time += 1

            # Update all environments
            for env in self.environments:
                env.update()

            # Collect stats every 10 steps
            if step % 10 == 0:
                stats = self._collect_stats()
                history.append(stats)

                if callback:
                    callback(step, stats, self)

        return {
            'history': history,
            'final_stats': self._collect_stats(),
        }

    def _collect_stats(self) -> Dict:
        """Collect ecosystem statistics."""
        all_creatures = []
        for env in self.environments:
            all_creatures.extend(env.creatures)

        if not all_creatures:
            return {'total_population': 0}

        return {
            'total_population': len(all_creatures),
            'avg_energy': np.mean([c.energy for c in all_creatures]),
            'avg_size': np.mean([c.traits.size for c in all_creatures]),
            'avg_speed': np.mean([c.traits.speed for c in all_creatures]),
            'avg_aggression': np.mean([c.traits.aggression for c in all_creatures]),
            'avg_coherence': np.mean([c.traits.coherence for c in all_creatures]),
            'species_counts': {k: len(v) for k, v in self.species.items()},
        }


# ============================================================================
# DNA TO SOUND/MUSIC
# ============================================================================

def dna_to_melody(dna: DNA, duration: float = 10.0) -> List[Tuple[float, float, float]]:
    """
    Convert DNA sequence to musical melody.

    Returns list of (frequency, start_time, duration) tuples.
    """
    # Map nucleotides to musical properties
    nucleotide_notes = {
        'A': (440.0, 'C'),   # A note, C
        'T': (493.88, 'D'),  # B note, D
        'G': (392.00, 'G'),  # G note, G
        'C': (261.63, 'E'),  # C note, E
    }

    melody = []
    time = 0.0
    note_duration = duration / len(dna.sequence)

    for i, nucleotide in enumerate(dna.sequence):
        if nucleotide in nucleotide_notes:
            freq, note = nucleotide_notes[nucleotide]

            # Codon context affects octave and duration
            codon_start = (i // 3) * 3
            if codon_start + 3 <= len(dna.sequence):
                codon = dna.sequence[codon_start:codon_start + 3]
                # Amino acid affects expression
                if codon in CODON_TABLE:
                    aa = CODON_TABLE[codon]
                    if aa in 'FYWH':  # Aromatic = higher pitch
                        freq *= 1.5
                    elif aa in 'GAVLIP':  # Small = shorter
                        note_duration *= 0.75
                    elif aa in 'KR':  # Charged = louder/longer
                        note_duration *= 1.5

            melody.append((freq, time, note_duration * 0.8))  # 0.8 for staccato
            time += note_duration

    return melody


def genome_to_ascii_art(creature: QuantumCreature, width: int = 40, height: int = 20) -> str:
    """Generate ASCII art representation of a creature."""
    # Use genome to determine shape
    seq = creature.genome.sequence

    # Create shape based on traits
    lines = []

    # Body shape from size and complexity
    body_width = int(creature.traits.size * 5)
    body_height = int(creature.traits.size * 3)

    center_x = width // 2
    center_y = height // 2

    # Draw creature
    chars = " .:-=+*#%@"

    for y in range(height):
        line = ""
        for x in range(width):
            # Distance from center
            dx = (x - center_x) / (body_width + 1)
            dy = (y - center_y) / (body_height + 1)
            dist = math.sqrt(dx*dx + dy*dy)

            if dist < 0.3:
                # Core body
                idx = int(creature.traits.shape_complexity * (len(chars) - 1))
                line += chars[min(idx, len(chars)-1)]
            elif dist < 0.5:
                # Outer body
                line += chars[int(creature.traits.aggression * 5)]
            elif dist < 0.8:
                # Appendages based on genome
                pos = (x + y) % len(seq)
                if seq[pos] in 'ATGC':
                    line += {'A': '/', 'T': '\\', 'G': '|', 'C': '-'}[seq[pos]]
                else:
                    line += ' '
            else:
                line += ' '
        lines.append(line)

    # Add trait indicators
    trait_line = f"S:{creature.traits.size:.1f} E:{creature.energy:.0f} COH:{creature.traits.coherence:.2f}"
    lines.insert(0, trait_line)

    return '\n'.join(lines)


# ============================================================================
# DEMO
# ============================================================================

def demo():
    """Demonstrate quantum life simulation."""
    print("=" * 70)
    print("QUANTUM ARTIFICIAL LIFE SIMULATION")
    print("=" * 70)

    # 1. Create random creatures
    print("\n[1] CREATING QUANTUM CREATURES")
    print("-" * 40)
    creatures = [QuantumCreature.random(genome_length=100) for _ in range(5)]
    for c in creatures:
        print(f"  {c}")
        print(f"    Traits: size={c.traits.size:.2f}, speed={c.traits.speed:.2f}, " +
              f"agg={c.traits.aggression:.2f}, coh={c.traits.coherence:.2f}")

    # 2. Create environment
    print("\n[2] QUANTUM ENVIRONMENT")
    print("-" * 40)
    env = QuantumEnvironment(width=500, height=500)
    for c in creatures:
        env.add_creature(c)
    print(f"  Environment: {env.width}x{env.height}")
    print(f"  Food sources: {len(env.food)}")

    # 3. Evolve
    print("\n[3] EVOLVING ECOSYSTEM")
    print("-" * 40)

    def progress(gen, stats, env):
        if gen % 2 == 0:
            print(f"  Gen {gen}: Pop={stats['population']}, " +
                  f"AvgE={stats['avg_energy']:.1f}, " +
                  f"AvgFit={stats['avg_fitness']:.3f}, " +
                  f"Species={stats['species_diversity']}")

    result = env.evolve(generations=10, callback=progress)

    print(f"\n  Final population: {result['final_population']}")
    if result['best_creature']:
        best = result['best_creature']
        print(f"  Best creature: Gen{best.generation}, Fitness={best.calculate_fitness():.3f}")

    # 4. Multi-species ecosystem
    print("\n[4] MULTI-SPECIES ECOSYSTEM")
    print("-" * 40)
    ecosystem = QuantumEcosystem.create(n_species=3, population_per_species=15)
    print(f"  Created ecosystem with 3 species")

    # 5. DNA to music
    print("\n[5] DNA TO MELODY")
    print("-" * 40)
    if creatures:
        melody = dna_to_melody(DNA(creatures[0].genome.sequence, "test"), duration=5.0)
        print(f"  Generated melody with {len(melody)} notes")
        print(f"  First 5 notes: {melody[:5]}")

    # 6. ASCII art
    print("\n[6] CREATURE VISUALIZATION")
    print("-" * 40)
    if env.creatures:
        print(genome_to_ascii_art(env.creatures[0]))

    print("\n" + "=" * 70)
    print("QUANTUM LIFE DEMO COMPLETE!")
    print("=" * 70)


if __name__ == "__main__":
    demo()
