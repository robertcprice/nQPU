//! Quantum physics: walks, thermodynamics, topology, consciousness models, and biology.

// Foundational and interpretational physics modules.
#[path = "foundations/contextuality.rs"]
pub mod contextuality;
#[path = "foundations/ctc_simulation.rs"]
pub mod ctc_simulation;
#[path = "foundations/many_worlds.rs"]
pub mod many_worlds;
#[path = "foundations/quantum_cloning.rs"]
pub mod quantum_cloning;

// Information-theoretic and scrambling-focused physics.
#[path = "information/hayden_preskill.rs"]
pub mod hayden_preskill;
#[path = "information/quantum_darwinism.rs"]
pub mod quantum_darwinism;
#[path = "information/quantum_echoes.rs"]
pub mod quantum_echoes;

// Condensed-matter, topology, and thermodynamic models.
#[path = "matter/majorana_model.rs"]
pub mod majorana_model;
#[path = "matter/quantum_battery.rs"]
pub mod quantum_battery;
#[path = "matter/quantum_cellular_automata.rs"]
pub mod quantum_cellular_automata;
#[path = "matter/quantum_chaos.rs"]
pub mod quantum_chaos;
#[path = "matter/quantum_thermodynamics.rs"]
pub mod quantum_thermodynamics;
#[path = "matter/topological_expanded.rs"]
pub mod topological_expanded;
#[path = "matter/topological_quantum.rs"]
pub mod topological_quantum;

// Transport, walk, and propagation models.
#[path = "transport/quantum_random_walk.rs"]
pub mod quantum_random_walk;
#[path = "transport/quantum_walk.rs"]
pub mod quantum_walk;

// Biology-facing and consciousness-themed experiments.
#[path = "biology/quantum_biology.rs"]
pub mod quantum_biology;
#[path = "consciousness/anharmonic.rs"]
pub mod anharmonic;
#[path = "consciousness/quantum_iit.rs"]
pub mod quantum_iit;

#[cfg(feature = "experimental")]
#[path = "consciousness/orch_or.rs"]
pub mod orch_or;

#[cfg(feature = "experimental")]
#[path = "consciousness/microtubule_augmentor.rs"]
pub mod microtubule_augmentor;
