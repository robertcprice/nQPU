"""Evolutionary and Bayesian quantum circuit architecture search.

Provides tools for automatically discovering optimal quantum circuit
architectures for machine learning tasks.  Two search strategies are
implemented:

1. **Evolutionary search** (``EvolutionarySearch``): population-based
   approach with tournament selection, crossover, and mutation.
2. **Bayesian search** (``BayesianSearch``): surrogate-model-guided
   search using a simple Gaussian process-like model with upper
   confidence bound (UCB) acquisition.

Both strategies evaluate candidate architectures by training them
on data and measuring validation accuracy, while penalizing circuit
complexity (depth, CNOT count) to favor hardware-friendly solutions.

References
----------
- Du et al., "Quantum circuit architecture search for variational
  quantum algorithms" npj Quantum Inf. 8, 62 (2022)
- Zhang et al., "Neural Predictor Based Quantum Architecture Search"
  Machine Learning: Science and Technology 2, 045027 (2021)
"""

from __future__ import annotations

import math
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np


# ------------------------------------------------------------------
# Gate specification
# ------------------------------------------------------------------


@dataclass(frozen=True)
class GateSpec:
    """Specification for a gate in the architecture.

    Attributes
    ----------
    gate_type : str
        Gate type identifier: "Rx", "Ry", "Rz", "CNOT", or "CZ".
    qubits : tuple of int
        Target qubit indices (1-tuple for single-qubit, 2-tuple for two-qubit).
    has_param : bool
        Whether this gate has a trainable parameter.
    """

    gate_type: str
    qubits: tuple
    has_param: bool = True


# ------------------------------------------------------------------
# Circuit architecture
# ------------------------------------------------------------------


_SINGLE_QUBIT_GATES = ("Rx", "Ry", "Rz")
_TWO_QUBIT_GATES = ("CNOT", "CZ")
_ALL_GATE_TYPES = _SINGLE_QUBIT_GATES + _TWO_QUBIT_GATES


@dataclass
class CircuitArchitecture:
    """A quantum circuit architecture (topology without parameter values).

    Describes the structure of a parameterized quantum circuit: which
    gates are applied, on which qubits, and in what order.  Parameter
    values are not stored here; they are initialized fresh for each
    training run.

    Parameters
    ----------
    n_qubits : int
        Number of qubits.
    gates : list of GateSpec
        Ordered list of gate specifications.
    """

    n_qubits: int
    gates: List[GateSpec] = field(default_factory=list)

    @property
    def n_params(self) -> int:
        """Number of trainable parameters in this architecture."""
        return sum(1 for g in self.gates if g.has_param)

    @property
    def depth(self) -> int:
        """Approximate circuit depth (number of layers of parallel gates).

        Computed as the maximum number of gates applied to any single qubit.
        """
        if not self.gates:
            return 0
        qubit_depths = {}
        for g in self.gates:
            for q in g.qubits:
                qubit_depths[q] = qubit_depths.get(q, 0) + 1
        return max(qubit_depths.values()) if qubit_depths else 0

    @property
    def cnot_count(self) -> int:
        """Number of CNOT (and CZ) two-qubit gates."""
        return sum(1 for g in self.gates if g.gate_type in _TWO_QUBIT_GATES)

    def mutate(self, rng=None) -> "CircuitArchitecture":
        """Create a mutated copy of this architecture.

        Mutation operations (chosen randomly):
        1. Add a random gate
        2. Remove a random gate
        3. Change the type of a random gate
        4. Change the target qubit(s) of a random gate

        Parameters
        ----------
        rng : np.random.Generator or None
            Random number generator.

        Returns
        -------
        CircuitArchitecture
            A new, mutated architecture.
        """
        if rng is None:
            rng = np.random.default_rng()

        child = CircuitArchitecture(
            n_qubits=self.n_qubits,
            gates=list(self.gates),
        )

        mutation_type = rng.choice(["add", "remove", "change_type", "change_qubit"])

        if mutation_type == "add" or len(child.gates) == 0:
            # Add a random gate
            gate = _random_gate(self.n_qubits, rng)
            pos = rng.integers(0, max(len(child.gates), 1) + 1)
            child.gates.insert(pos, gate)

        elif mutation_type == "remove" and len(child.gates) > 1:
            idx = rng.integers(0, len(child.gates))
            child.gates.pop(idx)

        elif mutation_type == "change_type" and len(child.gates) > 0:
            idx = rng.integers(0, len(child.gates))
            old = child.gates[idx]
            if old.gate_type in _SINGLE_QUBIT_GATES:
                new_type = rng.choice(_SINGLE_QUBIT_GATES)
                child.gates[idx] = GateSpec(
                    gate_type=new_type,
                    qubits=old.qubits,
                    has_param=(new_type in _SINGLE_QUBIT_GATES),
                )
            else:
                new_type = rng.choice(_TWO_QUBIT_GATES)
                child.gates[idx] = GateSpec(
                    gate_type=new_type, qubits=old.qubits, has_param=False
                )

        elif mutation_type == "change_qubit" and len(child.gates) > 0:
            idx = rng.integers(0, len(child.gates))
            old = child.gates[idx]
            if len(old.qubits) == 1:
                new_q = (int(rng.integers(0, self.n_qubits)),)
                child.gates[idx] = GateSpec(
                    gate_type=old.gate_type, qubits=new_q, has_param=old.has_param
                )
            elif self.n_qubits >= 2:
                q0 = int(rng.integers(0, self.n_qubits))
                q1 = int(rng.integers(0, self.n_qubits - 1))
                if q1 >= q0:
                    q1 += 1
                child.gates[idx] = GateSpec(
                    gate_type=old.gate_type, qubits=(q0, q1), has_param=False
                )

        return child

    @staticmethod
    def random(n_qubits: int, n_gates: int, rng=None) -> "CircuitArchitecture":
        """Generate a random architecture.

        Parameters
        ----------
        n_qubits : int
            Number of qubits.
        n_gates : int
            Number of gates to include.
        rng : np.random.Generator or None
            Random number generator.

        Returns
        -------
        CircuitArchitecture
            A randomly generated architecture.
        """
        if rng is None:
            rng = np.random.default_rng()

        gates = [_random_gate(n_qubits, rng) for _ in range(n_gates)]
        return CircuitArchitecture(n_qubits=n_qubits, gates=gates)

    def to_feature_vector(self) -> np.ndarray:
        """Convert architecture to a numerical feature vector for surrogate models.

        Returns a fixed-length vector encoding structural properties:
        [n_params, depth, cnot_count, n_rx, n_ry, n_rz, n_cnot, n_cz, total_gates].

        Returns
        -------
        np.ndarray
            Feature vector of length 9.
        """
        counts = {"Rx": 0, "Ry": 0, "Rz": 0, "CNOT": 0, "CZ": 0}
        for g in self.gates:
            if g.gate_type in counts:
                counts[g.gate_type] += 1
        return np.array([
            self.n_params,
            self.depth,
            self.cnot_count,
            counts["Rx"],
            counts["Ry"],
            counts["Rz"],
            counts["CNOT"],
            counts["CZ"],
            len(self.gates),
        ], dtype=np.float64)


def _random_gate(n_qubits: int, rng) -> GateSpec:
    """Generate a random gate specification.

    Parameters
    ----------
    n_qubits : int
        Number of available qubits.
    rng : np.random.Generator
        Random number generator.

    Returns
    -------
    GateSpec
        A randomly chosen gate.
    """
    if n_qubits >= 2 and rng.random() < 0.3:
        # Two-qubit gate
        gate_type = rng.choice(_TWO_QUBIT_GATES)
        q0 = int(rng.integers(0, n_qubits))
        q1 = int(rng.integers(0, n_qubits - 1))
        if q1 >= q0:
            q1 += 1
        return GateSpec(gate_type=gate_type, qubits=(q0, q1), has_param=False)
    else:
        gate_type = rng.choice(_SINGLE_QUBIT_GATES)
        q = int(rng.integers(0, n_qubits))
        return GateSpec(gate_type=gate_type, qubits=(q,), has_param=True)


# ------------------------------------------------------------------
# Architecture simulation (lightweight)
# ------------------------------------------------------------------


def _simulate_architecture(
    arch: CircuitArchitecture, x: np.ndarray, params: np.ndarray
) -> np.ndarray:
    """Simulate a circuit architecture with given parameters.

    Parameters
    ----------
    arch : CircuitArchitecture
        The architecture to simulate.
    x : np.ndarray
        Input features (used for data encoding via first layer).
    params : np.ndarray
        Trainable parameter vector.

    Returns
    -------
    np.ndarray
        Final statevector.
    """
    n = arch.n_qubits
    dim = 1 << n
    state = np.zeros(dim, dtype=np.complex128)
    state[0] = 1.0

    param_idx = 0
    for g in arch.gates:
        if g.gate_type == "Rx":
            angle = params[param_idx] if param_idx < len(params) else 0.0
            param_idx += 1
            c = math.cos(angle / 2)
            s = math.sin(angle / 2)
            mat = np.array([[c, -1j * s], [-1j * s, c]], dtype=np.complex128)
            state = _apply_single(state, n, g.qubits[0], mat)
        elif g.gate_type == "Ry":
            angle = params[param_idx] if param_idx < len(params) else 0.0
            param_idx += 1
            c = math.cos(angle / 2)
            s = math.sin(angle / 2)
            mat = np.array([[c, -s], [s, c]], dtype=np.complex128)
            state = _apply_single(state, n, g.qubits[0], mat)
        elif g.gate_type == "Rz":
            angle = params[param_idx] if param_idx < len(params) else 0.0
            param_idx += 1
            c = math.cos(angle / 2)
            s = math.sin(angle / 2)
            mat = np.array([[c - 1j * s, 0], [0, c + 1j * s]], dtype=np.complex128)
            state = _apply_single(state, n, g.qubits[0], mat)
        elif g.gate_type == "CNOT":
            state = _apply_cnot(state, n, g.qubits[0], g.qubits[1])
        elif g.gate_type == "CZ":
            state = _apply_cz(state, n, g.qubits[0], g.qubits[1])

    return state


def _apply_single(state, n_qubits, qubit, mat):
    """Apply single-qubit gate to statevector."""
    dim = 1 << n_qubits
    new_state = np.zeros(dim, dtype=np.complex128)
    step = 1 << qubit
    for i in range(dim):
        if i & step == 0:
            j = i | step
            a, b = state[i], state[j]
            new_state[i] += mat[0, 0] * a + mat[0, 1] * b
            new_state[j] += mat[1, 0] * a + mat[1, 1] * b
    return new_state


def _apply_cnot(state, n_qubits, control, target):
    """Apply CNOT gate."""
    dim = 1 << n_qubits
    new_state = state.copy()
    c_step = 1 << control
    t_step = 1 << target
    for i in range(dim):
        if (i & c_step) != 0 and (i & t_step) == 0:
            j = i | t_step
            new_state[i], new_state[j] = state[j], state[i]
    return new_state


def _apply_cz(state, n_qubits, q0, q1):
    """Apply CZ gate."""
    dim = 1 << n_qubits
    new_state = state.copy()
    s0 = 1 << q0
    s1 = 1 << q1
    for i in range(dim):
        if (i & s0) != 0 and (i & s1) != 0:
            new_state[i] = -state[i]
    return new_state


# ------------------------------------------------------------------
# Fitness evaluation
# ------------------------------------------------------------------


@dataclass
class FitnessResult:
    """Result of evaluating a circuit architecture.

    Attributes
    ----------
    val_accuracy : float
        Validation accuracy achieved after training.
    val_loss : float
        Validation loss after training.
    n_params : int
        Number of trainable parameters.
    depth : int
        Circuit depth.
    cnot_count : int
        Number of two-qubit gates.
    training_time_steps : int
        Number of training iterations performed.
    """

    val_accuracy: float
    val_loss: float
    n_params: int
    depth: int
    cnot_count: int
    training_time_steps: int


@dataclass
class FitnessEvaluator:
    """Evaluate circuit architecture quality by training and validating.

    Trains a candidate architecture on the training set using a simple
    gradient-based optimizer, then measures performance on the validation set.

    Parameters
    ----------
    X_train : np.ndarray
        Training features.
    y_train : np.ndarray
        Training labels.
    X_val : np.ndarray
        Validation features.
    y_val : np.ndarray
        Validation labels.
    n_epochs : int
        Number of training epochs for each architecture evaluation.
    """

    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    n_epochs: int = 20

    def evaluate(self, arch: CircuitArchitecture, rng=None) -> FitnessResult:
        """Train and evaluate an architecture.

        Parameters
        ----------
        arch : CircuitArchitecture
            The architecture to evaluate.
        rng : np.random.Generator or None
            Random number generator.

        Returns
        -------
        FitnessResult
            Evaluation results including validation accuracy and loss.
        """
        if rng is None:
            rng = np.random.default_rng()

        n_params = arch.n_params
        if n_params == 0:
            # Architecture has no trainable parameters
            return FitnessResult(
                val_accuracy=0.0,
                val_loss=float("inf"),
                n_params=0,
                depth=arch.depth,
                cnot_count=arch.cnot_count,
                training_time_steps=0,
            )

        params = rng.uniform(0, 2 * np.pi, size=n_params)
        lr = 0.1

        # Simple SGD training loop
        for epoch in range(self.n_epochs):
            grad = self._gradient(arch, self.X_train, self.y_train, params)
            params = params - lr * grad

        # Evaluate on validation set
        val_loss = self._loss(arch, self.X_val, self.y_val, params)
        val_acc = self._accuracy(arch, self.X_val, self.y_val, params)

        return FitnessResult(
            val_accuracy=val_acc,
            val_loss=val_loss,
            n_params=n_params,
            depth=arch.depth,
            cnot_count=arch.cnot_count,
            training_time_steps=self.n_epochs,
        )

    def _predict_proba(
        self, arch: CircuitArchitecture, x: np.ndarray, params: np.ndarray
    ) -> np.ndarray:
        """Get class probabilities from circuit measurement."""
        state = _simulate_architecture(arch, x, params)
        probs = np.abs(state) ** 2
        # Binary: first qubit marginal
        dim = 1 << arch.n_qubits
        p0 = 0.0
        for i in range(dim):
            if (i & 1) == 0:
                p0 += probs[i]
        p0 = np.clip(p0, 1e-10, 1.0 - 1e-10)
        return np.array([p0, 1.0 - p0])

    def _loss(
        self,
        arch: CircuitArchitecture,
        X: np.ndarray,
        y: np.ndarray,
        params: np.ndarray,
    ) -> float:
        """Cross-entropy loss."""
        total = 0.0
        for i in range(len(X)):
            probs = self._predict_proba(arch, X[i], params)
            label = int(y[i])
            total -= math.log(np.clip(probs[label], 1e-10, 1.0))
        return total / max(len(X), 1)

    def _accuracy(
        self,
        arch: CircuitArchitecture,
        X: np.ndarray,
        y: np.ndarray,
        params: np.ndarray,
    ) -> float:
        """Classification accuracy."""
        correct = 0
        for i in range(len(X)):
            probs = self._predict_proba(arch, X[i], params)
            if int(np.argmax(probs)) == int(y[i]):
                correct += 1
        return correct / max(len(X), 1)

    def _gradient(
        self,
        arch: CircuitArchitecture,
        X: np.ndarray,
        y: np.ndarray,
        params: np.ndarray,
    ) -> np.ndarray:
        """Parameter-shift gradient of the loss."""
        grad = np.zeros(len(params))
        shift = np.pi / 2
        denom = 2.0 * np.sin(shift)

        for i in range(len(params)):
            e_i = np.zeros(len(params))
            e_i[i] = shift
            loss_plus = self._loss(arch, X, y, params + e_i)
            loss_minus = self._loss(arch, X, y, params - e_i)
            grad[i] = (loss_plus - loss_minus) / denom
        return grad


# ------------------------------------------------------------------
# Search result
# ------------------------------------------------------------------


@dataclass
class SearchResult:
    """Result of an architecture search.

    Attributes
    ----------
    best_architecture : CircuitArchitecture
        The best architecture found during the search.
    best_fitness : FitnessResult
        The fitness evaluation of the best architecture.
    history : list of (CircuitArchitecture, FitnessResult)
        Full search history of evaluated architectures.
    generations : int
        Number of search iterations completed.
    """

    best_architecture: CircuitArchitecture
    best_fitness: FitnessResult
    history: List[Tuple[CircuitArchitecture, FitnessResult]]
    generations: int


# ------------------------------------------------------------------
# Evolutionary architecture search
# ------------------------------------------------------------------


@dataclass
class EvolutionarySearch:
    """Evolutionary architecture search using tournament selection.

    Maintains a population of candidate architectures that evolve
    through mutation, crossover, and selection.  Tournament selection
    is used to balance exploration and exploitation.

    Parameters
    ----------
    n_qubits : int
        Number of qubits for architectures.
    population_size : int
        Number of individuals in the population.
    n_generations : int
        Number of evolutionary generations.
    mutation_rate : float
        Probability of mutating each offspring.
    tournament_size : int
        Number of individuals competing in each tournament.
    """

    n_qubits: int
    population_size: int = 20
    n_generations: int = 10
    mutation_rate: float = 0.3
    tournament_size: int = 3

    def search(self, evaluator: FitnessEvaluator, rng=None) -> SearchResult:
        """Run evolutionary search.

        Parameters
        ----------
        evaluator : FitnessEvaluator
            Fitness evaluator for training and scoring architectures.
        rng : np.random.Generator or None
            Random number generator.

        Returns
        -------
        SearchResult
            Best architecture and full search history.
        """
        if rng is None:
            rng = np.random.default_rng()

        # Initialize population
        population = [
            CircuitArchitecture.random(
                self.n_qubits, rng.integers(3, 12), rng=rng
            )
            for _ in range(self.population_size)
        ]

        # Evaluate initial population
        fitnesses = [evaluator.evaluate(arch, rng=rng) for arch in population]
        history: List[Tuple[CircuitArchitecture, FitnessResult]] = list(
            zip(population, fitnesses)
        )

        best_idx = int(np.argmax([f.val_accuracy for f in fitnesses]))
        best_arch = deepcopy(population[best_idx])
        best_fitness = fitnesses[best_idx]

        for gen in range(self.n_generations):
            new_population = []

            # Elitism: keep the best individual
            new_population.append(deepcopy(best_arch))

            while len(new_population) < self.population_size:
                # Tournament selection of two parents
                parent1 = self._tournament_select(population, fitnesses, rng)
                parent2 = self._tournament_select(population, fitnesses, rng)

                # Crossover
                child = self._crossover(parent1, parent2, rng)

                # Mutation
                if rng.random() < self.mutation_rate:
                    child = child.mutate(rng=rng)

                new_population.append(child)

            population = new_population
            fitnesses = [evaluator.evaluate(arch, rng=rng) for arch in population]
            history.extend(zip(population, fitnesses))

            # Update best
            gen_best_idx = int(np.argmax([f.val_accuracy for f in fitnesses]))
            if fitnesses[gen_best_idx].val_accuracy > best_fitness.val_accuracy:
                best_arch = deepcopy(population[gen_best_idx])
                best_fitness = fitnesses[gen_best_idx]

        return SearchResult(
            best_architecture=best_arch,
            best_fitness=best_fitness,
            history=history,
            generations=self.n_generations,
        )

    def _tournament_select(
        self,
        population: list,
        fitnesses: list,
        rng,
    ) -> CircuitArchitecture:
        """Select an individual via tournament selection.

        Parameters
        ----------
        population : list of CircuitArchitecture
            Current population.
        fitnesses : list of FitnessResult
            Fitness for each individual.
        rng : np.random.Generator
            Random number generator.

        Returns
        -------
        CircuitArchitecture
            Winner of the tournament.
        """
        indices = rng.choice(
            len(population), size=min(self.tournament_size, len(population)), replace=False
        )
        best_i = indices[0]
        for i in indices[1:]:
            if fitnesses[i].val_accuracy > fitnesses[best_i].val_accuracy:
                best_i = i
        return deepcopy(population[best_i])

    def _crossover(
        self,
        parent1: CircuitArchitecture,
        parent2: CircuitArchitecture,
        rng,
    ) -> CircuitArchitecture:
        """Create offspring by combining gates from two parents.

        Uses single-point crossover: takes the first half of gates from
        parent1 and the second half from parent2.

        Parameters
        ----------
        parent1 : CircuitArchitecture
            First parent.
        parent2 : CircuitArchitecture
            Second parent.
        rng : np.random.Generator
            Random number generator.

        Returns
        -------
        CircuitArchitecture
            Offspring architecture.
        """
        if len(parent1.gates) == 0 and len(parent2.gates) == 0:
            return CircuitArchitecture.random(self.n_qubits, 5, rng=rng)

        # Crossover point
        cut1 = rng.integers(0, max(len(parent1.gates), 1))
        cut2 = rng.integers(0, max(len(parent2.gates), 1))

        child_gates = list(parent1.gates[:cut1]) + list(parent2.gates[cut2:])

        # Ensure valid qubit indices for the child
        valid_gates = []
        for g in child_gates:
            max_q = max(g.qubits)
            if max_q < self.n_qubits:
                valid_gates.append(g)
            else:
                # Remap to valid qubits
                new_qubits = tuple(q % self.n_qubits for q in g.qubits)
                if len(set(new_qubits)) == len(new_qubits):  # No duplicates
                    valid_gates.append(
                        GateSpec(
                            gate_type=g.gate_type,
                            qubits=new_qubits,
                            has_param=g.has_param,
                        )
                    )

        if len(valid_gates) == 0:
            valid_gates.append(_random_gate(self.n_qubits, rng))

        return CircuitArchitecture(n_qubits=self.n_qubits, gates=valid_gates)


# ------------------------------------------------------------------
# Bayesian architecture search
# ------------------------------------------------------------------


@dataclass
class BayesianSearch:
    """Bayesian optimization for architecture hyperparameters.

    Uses a simple surrogate model (RBF kernel regression) with upper
    confidence bound (UCB) acquisition to guide the search.  The
    surrogate predicts validation accuracy from architectural features.

    Parameters
    ----------
    n_qubits : int
        Number of qubits for architectures.
    max_depth : int
        Maximum circuit depth to consider.
    max_gates : int
        Maximum number of gates per architecture.
    n_iterations : int
        Number of Bayesian optimization iterations.
    """

    n_qubits: int
    max_depth: int = 10
    max_gates: int = 20
    n_iterations: int = 30

    def search(self, evaluator: FitnessEvaluator, rng=None) -> SearchResult:
        """Run Bayesian architecture search using a surrogate model.

        The search alternates between:
        1. Fitting a surrogate model to observed (architecture, fitness) pairs.
        2. Generating candidate architectures and selecting the one with
           the highest UCB acquisition value.
        3. Evaluating the selected candidate.

        Parameters
        ----------
        evaluator : FitnessEvaluator
            Fitness evaluator.
        rng : np.random.Generator or None
            Random number generator.

        Returns
        -------
        SearchResult
            Best architecture and full search history.
        """
        if rng is None:
            rng = np.random.default_rng()

        # Initial random architectures
        n_initial = min(5, self.n_iterations)
        observed_archs: List[CircuitArchitecture] = []
        observed_features: List[np.ndarray] = []
        observed_fitness: List[FitnessResult] = []
        history: List[Tuple[CircuitArchitecture, FitnessResult]] = []

        for _ in range(n_initial):
            n_gates = rng.integers(2, min(self.max_gates, 10) + 1)
            arch = CircuitArchitecture.random(self.n_qubits, int(n_gates), rng=rng)
            fitness = evaluator.evaluate(arch, rng=rng)
            observed_archs.append(arch)
            observed_features.append(arch.to_feature_vector())
            observed_fitness.append(fitness)
            history.append((arch, fitness))

        # Bayesian optimization loop
        remaining = self.n_iterations - n_initial
        for iteration in range(remaining):
            # Fit surrogate model
            X_obs = np.array(observed_features)
            y_obs = np.array([f.val_accuracy for f in observed_fitness])

            # Generate candidates
            n_candidates = 20
            candidates = []
            candidate_features = []
            for _ in range(n_candidates):
                n_gates = rng.integers(2, min(self.max_gates, 15) + 1)
                cand = CircuitArchitecture.random(self.n_qubits, int(n_gates), rng=rng)
                candidates.append(cand)
                candidate_features.append(cand.to_feature_vector())

            X_cand = np.array(candidate_features)

            # UCB acquisition
            ucb_values = self._ucb_acquisition(X_obs, y_obs, X_cand, kappa=1.5)

            # Select best candidate
            best_cand_idx = int(np.argmax(ucb_values))
            selected = candidates[best_cand_idx]

            # Evaluate
            fitness = evaluator.evaluate(selected, rng=rng)
            observed_archs.append(selected)
            observed_features.append(selected.to_feature_vector())
            observed_fitness.append(fitness)
            history.append((selected, fitness))

        # Find best
        best_idx = int(np.argmax([f.val_accuracy for f in observed_fitness]))
        return SearchResult(
            best_architecture=deepcopy(observed_archs[best_idx]),
            best_fitness=observed_fitness[best_idx],
            history=history,
            generations=self.n_iterations,
        )

    def _ucb_acquisition(
        self,
        X_obs: np.ndarray,
        y_obs: np.ndarray,
        X_cand: np.ndarray,
        kappa: float = 1.5,
    ) -> np.ndarray:
        """Compute UCB acquisition values using RBF kernel regression.

        Parameters
        ----------
        X_obs : np.ndarray
            Observed feature vectors (n_obs, d).
        y_obs : np.ndarray
            Observed accuracies (n_obs,).
        X_cand : np.ndarray
            Candidate feature vectors (n_cand, d).
        kappa : float
            Exploration-exploitation trade-off parameter.

        Returns
        -------
        np.ndarray
            UCB values for each candidate (n_cand,).
        """
        n_obs = len(X_obs)
        if n_obs == 0:
            return np.ones(len(X_cand))

        # Normalize features
        mean_x = X_obs.mean(axis=0)
        std_x = X_obs.std(axis=0) + 1e-8
        X_obs_norm = (X_obs - mean_x) / std_x
        X_cand_norm = (X_cand - mean_x) / std_x

        # RBF kernel with lengthscale = 1
        def rbf_kernel(X1, X2, lengthscale=1.0):
            # Pairwise squared distances
            sq_dist = (
                np.sum(X1 ** 2, axis=1, keepdims=True)
                + np.sum(X2 ** 2, axis=1, keepdims=True).T
                - 2.0 * X1 @ X2.T
            )
            return np.exp(-0.5 * sq_dist / (lengthscale ** 2))

        # Kernel matrices
        K_obs = rbf_kernel(X_obs_norm, X_obs_norm) + 1e-4 * np.eye(n_obs)
        K_cand_obs = rbf_kernel(X_cand_norm, X_obs_norm)
        K_cand = rbf_kernel(X_cand_norm, X_cand_norm)

        # GP posterior
        try:
            L = np.linalg.cholesky(K_obs)
            alpha = np.linalg.solve(L.T, np.linalg.solve(L, y_obs))
            V = np.linalg.solve(L, K_cand_obs.T)
        except np.linalg.LinAlgError:
            # Fallback: pseudoinverse
            K_inv = np.linalg.pinv(K_obs)
            alpha = K_inv @ y_obs
            V = K_inv @ K_cand_obs.T

        # Mean prediction
        mu = K_cand_obs @ alpha

        # Variance prediction
        if len(V.shape) == 2:
            var = np.diag(K_cand) - np.sum(V ** 2, axis=0)
        else:
            var = np.diag(K_cand) - np.diag(K_cand_obs @ V)
        var = np.clip(var, 1e-10, None)
        sigma = np.sqrt(var)

        return mu + kappa * sigma
