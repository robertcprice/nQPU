//! Quantum chemistry: molecular simulation, factorization, DMRG, drug design, and materials.

// Core electronic-structure solvers and factorization helpers.
#[path = "electronic_structure/camps_dmrg.rs"]
pub mod camps_dmrg;
#[path = "electronic_structure/double_factorized.rs"]
pub mod double_factorized;
#[path = "electronic_structure/molecular_integrals.rs"]
pub mod molecular_integrals;
#[path = "electronic_structure/quantum_chemistry.rs"]
pub mod quantum_chemistry;

// Domain-facing chemistry and materials applications.
#[path = "applications/quantum_drug_design.rs"]
pub mod quantum_drug_design;
#[path = "applications/quantum_materials.rs"]
pub mod quantum_materials;
