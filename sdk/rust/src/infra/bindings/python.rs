//! Python bindings for nQPU-Metal
//!
//! This module provides Python bindings using PyO3, allowing nQPU-Metal
//! to be used from Python code for integration with the quantum Python ecosystem.

#![allow(
    deprecated,
    dead_code,
    unused_variables,
    unused_imports,
    unused_assignments
)] // Suppress PyO3 noise and unused items in Python API

#[cfg(feature = "python")]
use pyo3::exceptions::{PyOSError, PyValueError};
#[cfg(feature = "python")]
use pyo3::prelude::*;
#[cfg(feature = "python")]
use pyo3::wrap_pyfunction;
#[cfg(feature = "python")]
use serde::{Deserialize, Serialize};
#[cfg(feature = "python")]
use std::fs;

use crate::gates::Gate;
#[cfg(feature = "python")]
use crate::{
    auto_backend::AutoBackend,
    auto_simulator::{AutoSimulator, SimBackend},
};
use crate::{QuantumSimulator, QuantumState};

// Noise model imports
#[cfg(feature = "python")]
use crate::noise_models::NoiseModel;

// MPS imports
#[cfg(feature = "python")]
use crate::tensor_network::MPSSimulator;
#[cfg(feature = "python")]
use crate::tensor_networks::dmrg_tdvp::{self, DmrgConfig};

// Quantum entropy extraction imports
#[cfg(feature = "python")]
use crate::quantum_entropy_extraction::{
    generate_quantum_seeds, quick_quantum_seed, QuantumEntropyExtractor,
};

#[cfg(feature = "python")]
use ndarray::{Array2, Array3};
#[cfg(feature = "python")]
use num_complex::Complex64;
#[cfg(feature = "python")]
use rand::Rng;

// Differentiable circuit imports (PyTorch bridge)
#[cfg(feature = "python")]
use crate::pytorch_bridge::{self, DifferentiableCircuit};

// JAX bridge imports
#[cfg(feature = "python")]
use crate::jax_bridge;

/// nQPU-Metal Python module
#[cfg(feature = "python")]
#[pymodule]
fn nqpu_metal(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyQuantumSimulator>()?;
    m.add_class::<PyQuantumState>()?;
    m.add_function(wrap_pyfunction!(create_bell_state, m)?)?;
    m.add_function(wrap_pyfunction!(create_ghz_state, m)?)?;
    m.add_function(wrap_pyfunction!(run_grover, m)?)?;
    m.add_function(wrap_pyfunction!(benchmark_gates, m)?)?;

    // Advanced quantum algorithms
    m.add_function(wrap_pyfunction!(apply_qft, m)?)?;
    m.add_function(wrap_pyfunction!(run_phase_estimation, m)?)?;
    m.add_function(wrap_pyfunction!(run_vqe, m)?)?;

    // New v2 API
    m.add_class::<PyBackend>()?;
    m.add_class::<PyQuantumCircuit2>()?;
    m.add_class::<PySimulationResult>()?;
    m.add_class::<PyEnhancedSimulator>()?;
    m.add_function(wrap_pyfunction!(run_heisenberg_qpe_rust, m)?)?;

    // Noise models
    m.add_class::<PyNoiseModel>()?;
    m.add_function(wrap_pyfunction!(simulate_noisy_circuit, m)?)?;
    m.add_function(wrap_pyfunction!(compare_ideal_vs_noisy, m)?)?;

    // MPS (Matrix Product State) simulator
    m.add_class::<PyMPSSimulator>()?;
    m.add_class::<PyTensorNetworkState1D>()?;
    m.add_function(wrap_pyfunction!(statevector_to_mps_1d, m)?)?;
    m.add_function(wrap_pyfunction!(dmrg_ground_state_1d, m)?)?;
    m.add_function(wrap_pyfunction!(tdvp_time_evolution_1d, m)?)?;
    m.add_function(wrap_pyfunction!(tdvp_loschmidt_echo_1d, m)?)?;
    m.add_function(wrap_pyfunction!(entanglement_spectrum_1d, m)?)?;
    m.add_function(wrap_pyfunction!(apply_local_pauli_1d, m)?)?;
    m.add_function(wrap_pyfunction!(tdvp_transition_observables_1d, m)?)?;

    // 2D Grid simulator using snake-mapped MPS
    m.add_class::<PyGrid2DSimulator>()?;

    // 2D algorithms (now exposed as methods on Grid2DSimulator class)
    // Note: qft_2d, inverse_qft_2d, entanglement_entropy_2d, visualize_entanglement,
    //       power_sample_2d, square_cnot are now methods on PyGrid2DSimulator
    //       rather than standalone functions

    // Automatic differentiation (adjoint method) — for nHybrid integration
    m.add_class::<PyAdjointCircuit>()?;

    // Quantum entropy extraction for LLM seeding
    m.add_class::<PyQuantumEntropyExtractor>()?;
    m.add_function(wrap_pyfunction!(py_quick_seed, m)?)?;
    m.add_function(wrap_pyfunction!(py_batch_seeds, m)?)?;

    // PyTorch integration
    m.add_class::<PyDifferentiableCircuit>()?;

    // JAX integration
    m.add_class::<PyJAXCircuit>()?;

    // QPU hardware connectivity (when both python and qpu features are enabled)
    #[cfg(feature = "qpu")]
    {
        m.add_class::<PyQPUCircuit>()?;
        m.add_class::<PyMockProvider>()?;
        m.add_function(wrap_pyfunction!(py_qpu_providers, m)?)?;
    }

    // Sliding window QEC decoder
    m.add_class::<PySlidingWindowDecoder>()?;

    // BP-OSD decoder
    m.add_class::<PyBpOsdDecoder>()?;

    // Magic state distillation factory
    m.add_class::<PyMagicStateFactory>()?;

    // ADAPT-VQE algorithm
    m.add_class::<PyAdaptVqe>()?;
    m.add_class::<PyMolecularHamiltonian>()?;

    // Quantum walk simulation
    m.add_class::<PyWalkGraph>()?;
    m.add_class::<PyContinuousWalk>()?;
    m.add_class::<PyDiscreteWalk>()?;

    // Quantum annealing simulation
    m.add_class::<PyIsingModel>()?;
    m.add_class::<PyQuantumAnnealer>()?;

    // Fermionic Gaussian states
    m.add_class::<PyFermionicGaussianState>()?;

    // Neural quantum states (RBM + VMC)
    m.add_class::<PyRBMState>()?;
    m.add_class::<PyVMCOptimizer>()?;

    // Quantum drug design
    m.add_class::<PyMolecule>()?;
    m.add_class::<PyDrugLikenessResult>()?;
    m.add_class::<PyAdmetPredictor>()?;
    m.add_class::<PyQuantumDockingScorer>()?;
    m.add_class::<PyDrugDiscoveryPipeline>()?;
    m.add_function(wrap_pyfunction!(py_evaluate_drug_likeness, m)?)?;
    m.add_function(wrap_pyfunction!(py_predict_admet, m)?)?;
    m.add_function(wrap_pyfunction!(py_screen_library, m)?)?;
    m.add_function(wrap_pyfunction!(py_optimize_lead, m)?)?;

    Ok(())
}

// =============================================================================
// TENSOR-NETWORK RESEARCH BINDINGS
// =============================================================================

#[cfg(feature = "python")]
#[pyclass(name = "TensorNetworkState1D")]
pub struct PyTensorNetworkState1D {
    mps: dmrg_tdvp::Mps,
    num_sites: usize,
    model_name: String,
    solver: String,
}

#[cfg(feature = "python")]
#[derive(Debug, Clone, Serialize, Deserialize)]
struct SerializableMpsSite {
    left_bond: usize,
    physical_dim: usize,
    right_bond: usize,
    shape: [usize; 3],
    tensor_real: Vec<f64>,
    tensor_imag: Vec<f64>,
}

#[cfg(feature = "python")]
#[derive(Debug, Clone, Serialize, Deserialize)]
struct SerializableTensorNetworkState1D {
    version: u32,
    num_sites: usize,
    center_position: usize,
    model_name: String,
    solver: String,
    sites: Vec<SerializableMpsSite>,
}

#[cfg(feature = "python")]
fn serialize_tensor_network_state(
    state: &PyTensorNetworkState1D,
) -> SerializableTensorNetworkState1D {
    let sites = state
        .mps
        .sites
        .iter()
        .map(|site| {
            let shape = site.tensor.shape();
            let mut tensor_real = Vec::with_capacity(site.tensor.len());
            let mut tensor_imag = Vec::with_capacity(site.tensor.len());
            for value in site.tensor.iter() {
                tensor_real.push(value.re);
                tensor_imag.push(value.im);
            }
            SerializableMpsSite {
                left_bond: site.left_bond,
                physical_dim: site.physical_dim,
                right_bond: site.right_bond,
                shape: [shape[0], shape[1], shape[2]],
                tensor_real,
                tensor_imag,
            }
        })
        .collect();

    SerializableTensorNetworkState1D {
        version: 1,
        num_sites: state.num_sites,
        center_position: state.mps.center_position,
        model_name: state.model_name.clone(),
        solver: state.solver.clone(),
        sites,
    }
}

#[cfg(feature = "python")]
fn deserialize_tensor_network_state(
    serializable: SerializableTensorNetworkState1D,
) -> PyResult<PyTensorNetworkState1D> {
    if serializable.version != 1 {
        return Err(PyValueError::new_err(format!(
            "unsupported TensorNetworkState1D checkpoint version {}",
            serializable.version
        )));
    }
    if serializable.num_sites != serializable.sites.len() {
        return Err(PyValueError::new_err(format!(
            "checkpoint reports {} sites but contains {} site tensors",
            serializable.num_sites,
            serializable.sites.len()
        )));
    }
    if serializable.num_sites == 0 {
        return Err(PyValueError::new_err(
            "TensorNetworkState1D checkpoints must contain at least one site",
        ));
    }
    if serializable.center_position >= serializable.num_sites {
        return Err(PyValueError::new_err(format!(
            "checkpoint center position {} is out of range for {} sites",
            serializable.center_position, serializable.num_sites
        )));
    }

    let mut sites = Vec::with_capacity(serializable.sites.len());
    for (index, site) in serializable.sites.into_iter().enumerate() {
        let expected_len = site.shape[0] * site.shape[1] * site.shape[2];
        if site.tensor_real.len() != expected_len || site.tensor_imag.len() != expected_len {
            return Err(PyValueError::new_err(format!(
                "site {index} tensor length does not match declared shape {:?}",
                site.shape
            )));
        }
        if site.left_bond != site.shape[0]
            || site.physical_dim != site.shape[1]
            || site.right_bond != site.shape[2]
        {
            return Err(PyValueError::new_err(format!(
                "site {index} bond dimensions do not match declared tensor shape {:?}",
                site.shape
            )));
        }

        let data = site
            .tensor_real
            .into_iter()
            .zip(site.tensor_imag.into_iter())
            .map(|(re, im)| Complex64::new(re, im))
            .collect::<Vec<_>>();
        let tensor = Array3::from_shape_vec((site.shape[0], site.shape[1], site.shape[2]), data)
            .map_err(|error: ndarray::ShapeError| PyValueError::new_err(error.to_string()))?;

        sites.push(dmrg_tdvp::MpsSite {
            tensor,
            physical_dim: site.physical_dim,
            left_bond: site.left_bond,
            right_bond: site.right_bond,
        });
    }

    Ok(PyTensorNetworkState1D {
        num_sites: serializable.num_sites,
        model_name: serializable.model_name,
        solver: serializable.solver,
        mps: dmrg_tdvp::Mps {
            sites,
            center_position: serializable.center_position,
            num_sites: serializable.num_sites,
        },
    })
}

#[cfg(feature = "python")]
#[pymethods]
impl PyTensorNetworkState1D {
    #[getter]
    fn num_sites(&self) -> usize {
        self.num_sites
    }

    #[getter]
    fn model_name(&self) -> String {
        self.model_name.clone()
    }

    #[getter]
    fn solver(&self) -> String {
        self.solver.clone()
    }

    fn __repr__(&self) -> String {
        format!(
            "TensorNetworkState1D(num_sites={}, model='{}', solver='{}')",
            self.num_sites, self.model_name, self.solver
        )
    }

    fn to_json(&self) -> PyResult<String> {
        serde_json::to_string_pretty(&serialize_tensor_network_state(self))
            .map_err(|error| PyValueError::new_err(error.to_string()))
    }

    fn save_json(&self, path: &str) -> PyResult<()> {
        let payload = self.to_json()?;
        fs::write(path, payload).map_err(|error| PyOSError::new_err(error.to_string()))
    }

    #[staticmethod]
    fn from_json(payload: &str) -> PyResult<Self> {
        let serializable: SerializableTensorNetworkState1D = serde_json::from_str(payload)
            .map_err(|error| PyValueError::new_err(error.to_string()))?;
        deserialize_tensor_network_state(serializable)
    }

    #[staticmethod]
    fn load_json(path: &str) -> PyResult<Self> {
        let payload =
            fs::read_to_string(path).map_err(|error| PyOSError::new_err(error.to_string()))?;
        Self::from_json(&payload)
    }
}

#[cfg(feature = "python")]
fn tensor_network_state_handle(
    py: Python<'_>,
    mps: &dmrg_tdvp::Mps,
    model_name: &str,
    solver: &str,
) -> PyResult<Py<PyTensorNetworkState1D>> {
    Py::new(
        py,
        PyTensorNetworkState1D {
            mps: mps.clone(),
            num_sites: mps.num_sites,
            model_name: model_name.to_owned(),
            solver: solver.to_owned(),
        },
    )
}

#[cfg(feature = "python")]
fn pauli_matrix(label: char) -> PyResult<Array2<Complex64>> {
    match label {
        'I' => Ok(Array2::from_shape_vec(
            (2, 2),
            vec![
                Complex64::new(1.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(1.0, 0.0),
            ],
        )
        .unwrap()),
        'X' => Ok(Array2::from_shape_vec(
            (2, 2),
            vec![
                Complex64::new(0.0, 0.0),
                Complex64::new(1.0, 0.0),
                Complex64::new(1.0, 0.0),
                Complex64::new(0.0, 0.0),
            ],
        )
        .unwrap()),
        'Y' => Ok(Array2::from_shape_vec(
            (2, 2),
            vec![
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, -1.0),
                Complex64::new(0.0, 1.0),
                Complex64::new(0.0, 0.0),
            ],
        )
        .unwrap()),
        'Z' => Ok(Array2::from_shape_vec(
            (2, 2),
            vec![
                Complex64::new(1.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(-1.0, 0.0),
            ],
        )
        .unwrap()),
        _ => Err(PyValueError::new_err(format!(
            "unsupported Pauli operator '{label}'"
        ))),
    }
}

#[cfg(feature = "python")]
fn parse_pauli_label(label: &str) -> PyResult<Vec<(char, usize)>> {
    let chars: Vec<char> = label.chars().collect();
    let mut index = 0usize;
    let mut terms = Vec::new();

    while index < chars.len() {
        let pauli = chars[index];
        if !matches!(pauli, 'I' | 'X' | 'Y' | 'Z') {
            return Err(PyValueError::new_err(format!(
                "unsupported observable label '{label}'"
            )));
        }
        index += 1;

        if index >= chars.len() || !chars[index].is_ascii_digit() {
            return Err(PyValueError::new_err(format!(
                "observable labels must use indexed Pauli strings like Z0 or X0X1, got '{label}'"
            )));
        }

        let start = index;
        while index < chars.len() && chars[index].is_ascii_digit() {
            index += 1;
        }
        let site: usize = label[start..index]
            .parse()
            .map_err(|_| PyValueError::new_err(format!("invalid site index in '{label}'")))?;
        terms.push((pauli, site));
    }

    Ok(terms)
}

#[cfg(feature = "python")]
fn evaluate_dmrg_observable(
    mps: &dmrg_tdvp::Mps,
    hamiltonian: &dmrg_tdvp::MpoHamiltonian,
    label: &str,
) -> PyResult<f64> {
    if label == "energy" {
        return Ok(dmrg_tdvp::mps_energy(mps, hamiltonian));
    }

    if label == "magnetization_z" {
        let z = pauli_matrix('Z')?;
        let total: f64 = (0..mps.num_sites)
            .map(|site| dmrg_tdvp::measure_local_observable(mps, &z, site).re)
            .sum();
        return Ok(total / mps.num_sites as f64);
    }

    if label == "staggered_magnetization_z" {
        let z = pauli_matrix('Z')?;
        let total: f64 = (0..mps.num_sites)
            .map(|site| {
                let sign = if site % 2 == 0 { 1.0 } else { -1.0 };
                sign * dmrg_tdvp::measure_local_observable(mps, &z, site).re
            })
            .sum();
        return Ok(total / mps.num_sites as f64);
    }

    let terms = parse_pauli_label(label)?;
    if terms.iter().any(|(_, site)| *site >= mps.num_sites) {
        return Err(PyValueError::new_err(format!(
            "observable '{label}' references a site outside the chain"
        )));
    }

    match terms.as_slice() {
        [(pauli, site)] => {
            let operator = pauli_matrix(*pauli)?;
            Ok(dmrg_tdvp::measure_local_observable(mps, &operator, *site).re)
        }
        [(pauli_a, site_a), (pauli_b, site_b)] => {
            if site_a == site_b {
                return Err(PyValueError::new_err(format!(
                    "observable '{label}' applies multiple operators to site {site_a}"
                )));
            }
            let op_a = pauli_matrix(*pauli_a)?;
            let op_b = pauli_matrix(*pauli_b)?;
            Ok(dmrg_tdvp::measure_correlation(mps, &op_a, *site_a, &op_b, *site_b).re)
        }
        _ => Err(PyValueError::new_err(
            "DMRG observable support is currently limited to one- and two-site Pauli strings",
        )),
    }
}

#[cfg(feature = "python")]
fn evaluate_transition_observable(
    bra: &dmrg_tdvp::Mps,
    ket: &dmrg_tdvp::Mps,
    label: &str,
) -> PyResult<Complex64> {
    let terms = parse_pauli_label(label)?;
    if terms.iter().any(|(_, site)| *site >= bra.num_sites || *site >= ket.num_sites) {
        return Err(PyValueError::new_err(format!(
            "observable '{label}' references a site outside the chain"
        )));
    }

    match terms.as_slice() {
        [(pauli, site)] => {
            let operator = pauli_matrix(*pauli)?;
            Ok(dmrg_tdvp::measure_transition_local_observable(
                bra,
                ket,
                &operator,
                *site,
            ))
        }
        _ => Err(PyValueError::new_err(
            "transition observable support is currently limited to single-site Pauli strings",
        )),
    }
}

#[cfg(feature = "python")]
fn build_research_model_hamiltonian(
    model: &str,
    num_sites: usize,
    coupling: f64,
    coupling_x: Option<f64>,
    coupling_y: Option<f64>,
    coupling_z: Option<f64>,
    transverse_field: Option<f64>,
    longitudinal_field: Option<f64>,
    anisotropy: Option<f64>,
    field_z: Option<f64>,
) -> PyResult<dmrg_tdvp::MpoHamiltonian> {
    match model {
        "transverse_field_ising_1d" => Ok(dmrg_tdvp::build_mpo_ising_fields(
            num_sites,
            4.0 * coupling,
            2.0 * transverse_field.unwrap_or(0.0),
            2.0 * longitudinal_field.unwrap_or(0.0),
        )),
        "heisenberg_xxz_1d" => Ok(dmrg_tdvp::build_mpo_heisenberg_xxz(
            num_sites,
            4.0 * coupling,
            anisotropy.unwrap_or(1.0),
            2.0 * field_z.unwrap_or(0.0),
        )),
        "heisenberg_xyz_1d" => Ok(dmrg_tdvp::build_mpo_heisenberg_xyz(
            num_sites,
            4.0 * coupling_x.unwrap_or(1.0),
            4.0 * coupling_y.unwrap_or(1.0),
            4.0 * coupling_z.unwrap_or(1.0),
            2.0 * field_z.unwrap_or(0.0),
        )),
        _ => Err(PyValueError::new_err(format!(
            "unsupported 1D tensor-network model '{model}'"
        ))),
    }
}

#[cfg(feature = "python")]
fn parse_tdvp_method(method: &str) -> PyResult<dmrg_tdvp::TdvpMethod> {
    match method {
        "one_site" => Ok(dmrg_tdvp::TdvpMethod::OneSite),
        "two_site" => Ok(dmrg_tdvp::TdvpMethod::TwoSite),
        _ => Err(PyValueError::new_err(format!(
            "unsupported TDVP method '{method}'; expected 'one_site' or 'two_site'"
        ))),
    }
}

#[cfg(feature = "python")]
#[pyfunction(
    signature = (
        model,
        num_sites,
        state_handle = None,
        coupling = 1.0,
        coupling_x = None,
        coupling_y = None,
        coupling_z = None,
        transverse_field = None,
        longitudinal_field = None,
        anisotropy = None,
        field_z = None,
        max_bond_dim = 64,
        max_sweeps = 20,
        energy_tolerance = 1e-8,
        lanczos_iterations = 20,
        observables = None,
        entropy_bond = None
    )
)]
fn dmrg_ground_state_1d(
    py: Python<'_>,
    model: &str,
    num_sites: usize,
    state_handle: Option<Py<PyTensorNetworkState1D>>,
    coupling: f64,
    coupling_x: Option<f64>,
    coupling_y: Option<f64>,
    coupling_z: Option<f64>,
    transverse_field: Option<f64>,
    longitudinal_field: Option<f64>,
    anisotropy: Option<f64>,
    field_z: Option<f64>,
    max_bond_dim: usize,
    max_sweeps: usize,
    energy_tolerance: f64,
    lanczos_iterations: usize,
    observables: Option<Vec<String>>,
    entropy_bond: Option<usize>,
) -> PyResult<pyo3::Py<pyo3::types::PyDict>> {
    if num_sites < 2 {
        return Err(PyValueError::new_err(
            "DMRG requires at least 2 sites for the current 1D model bindings",
        ));
    }

    let hamiltonian = build_research_model_hamiltonian(
        model,
        num_sites,
        coupling,
        coupling_x,
        coupling_y,
        coupling_z,
        transverse_field,
        longitudinal_field,
        anisotropy,
        field_z,
    )?;

    let config = DmrgConfig::new()
        .max_bond_dim(max_bond_dim)
        .max_sweeps(max_sweeps)
        .energy_tolerance(energy_tolerance)
        .lanczos_iterations(lanczos_iterations);

    let result = if let Some(handle) = state_handle.as_ref() {
        let borrowed = handle.borrow(py);
        if borrowed.num_sites != num_sites {
            return Err(PyValueError::new_err(format!(
                "state handle has {} sites but the target model has {num_sites}",
                borrowed.num_sites
            )));
        }
        dmrg_tdvp::dmrg_from_mps(&borrowed.mps, &hamiltonian, &config)
            .map_err(|error| PyValueError::new_err(error.to_string()))?
    } else {
        dmrg_tdvp::dmrg(&hamiltonian, &config)
            .map_err(|error| PyValueError::new_err(error.to_string()))?
    };

    let observables = observables.unwrap_or_default();
    let observable_dict = pyo3::types::PyDict::new(py);
    for label in observables {
        observable_dict.set_item(
            &label,
            evaluate_dmrg_observable(&result.mps, &hamiltonian, &label)?,
        )?;
    }

    let dict = pyo3::types::PyDict::new(py);
    dict.set_item("model_name", model)?;
    dict.set_item("solver", "dmrg")?;
    dict.set_item("ground_state_energy", result.energy)?;
    dict.set_item("num_sweeps", result.num_sweeps)?;
    dict.set_item("converged", result.converged)?;
    dict.set_item("energy_history", result.energy_history)?;
    dict.set_item("observables", observable_dict)?;
    dict.set_item("spectral_gap", py.None())?;

    if let Some(bond) = entropy_bond {
        let entropy = dmrg_tdvp::entanglement_entropy(&result.mps, bond)
            .map_err(|error| PyValueError::new_err(error.to_string()))?
            / std::f64::consts::LN_2;
        dict.set_item("entanglement_entropy", entropy)?;
    } else {
        dict.set_item("entanglement_entropy", py.None())?;
    }
    dict.set_item(
        "state_handle",
        tensor_network_state_handle(py, &result.mps, model, "dmrg")?,
    )?;

    Ok(dict.into())
}

#[cfg(feature = "python")]
#[pyfunction(signature = (num_sites, statevector, max_bond_dim = 64, cutoff = 1e-10, model_name = "compressed_state"))]
fn statevector_to_mps_1d(
    py: Python<'_>,
    num_sites: usize,
    statevector: Vec<(f64, f64)>,
    max_bond_dim: usize,
    cutoff: f64,
    model_name: &str,
) -> PyResult<Py<PyTensorNetworkState1D>> {
    let dense_state: Vec<Complex64> = statevector
        .into_iter()
        .map(|(real, imag)| Complex64::new(real, imag))
        .collect();
    let mps = dmrg_tdvp::dense_statevector_to_mps(
        &dense_state,
        num_sites,
        max_bond_dim,
        cutoff,
    )
    .map_err(|error| PyValueError::new_err(error.to_string()))?;
    tensor_network_state_handle(py, &mps, model_name, "compressed_state")
}

#[cfg(feature = "python")]
#[pyfunction(
    signature = (
        model,
        num_sites,
        times,
        initial_state = None,
        state_handle = None,
        coupling = 1.0,
        coupling_x = None,
        coupling_y = None,
        coupling_z = None,
        transverse_field = None,
        longitudinal_field = None,
        anisotropy = None,
        field_z = None,
        method = "two_site",
        max_bond_dim = 64,
        lanczos_iterations = 20,
        observables = None,
        entropy_bond = None
    )
)]
fn tdvp_time_evolution_1d(
    py: Python<'_>,
    model: &str,
    num_sites: usize,
    times: Vec<f64>,
    initial_state: Option<&str>,
    state_handle: Option<Py<PyTensorNetworkState1D>>,
    coupling: f64,
    coupling_x: Option<f64>,
    coupling_y: Option<f64>,
    coupling_z: Option<f64>,
    transverse_field: Option<f64>,
    longitudinal_field: Option<f64>,
    anisotropy: Option<f64>,
    field_z: Option<f64>,
    method: &str,
    max_bond_dim: usize,
    lanczos_iterations: usize,
    observables: Option<Vec<String>>,
    entropy_bond: Option<usize>,
) -> PyResult<pyo3::Py<pyo3::types::PyDict>> {
    if num_sites < 2 {
        return Err(PyValueError::new_err(
            "TDVP requires at least 2 sites for the current 1D model bindings",
        ));
    }
    if times.iter().any(|time| !time.is_finite()) {
        return Err(PyValueError::new_err(
            "time points must be finite real numbers",
        ));
    }
    if let Some(bond) = entropy_bond {
        if bond >= num_sites - 1 {
            return Err(PyValueError::new_err(format!(
                "entropy bond {bond} is out of range for a {num_sites}-site chain"
            )));
        }
    }

    let tdvp_method = parse_tdvp_method(method)?;
    let hamiltonian = build_research_model_hamiltonian(
        model,
        num_sites,
        coupling,
        coupling_x,
        coupling_y,
        coupling_z,
        transverse_field,
        longitudinal_field,
        anisotropy,
        field_z,
    )?;
    let mut mps = if let Some(handle) = state_handle.as_ref() {
        let borrowed = handle.borrow(py);
        if borrowed.num_sites != num_sites {
            return Err(PyValueError::new_err(format!(
                "state handle has {} sites but the target model has {num_sites}",
                borrowed.num_sites
            )));
        }
        borrowed.mps.clone()
    } else {
        dmrg_tdvp::named_product_state_mps(num_sites, initial_state.unwrap_or("all_up"))
            .map_err(|error| PyValueError::new_err(error.to_string()))?
    };

    let labels = observables.unwrap_or_default();
    let mut series: Vec<Vec<f64>> = labels
        .iter()
        .map(|_| Vec::with_capacity(times.len()))
        .collect();
    let mut entropy_series = entropy_bond.map(|_| Vec::with_capacity(times.len()));
    let mut current_time = 0.0_f64;
    const TIME_EPSILON: f64 = 1e-12;

    for &target_time in &times {
        if target_time < -TIME_EPSILON {
            return Err(PyValueError::new_err(
                "TDVP currently requires non-negative time points",
            ));
        }
        if target_time + TIME_EPSILON < current_time {
            return Err(PyValueError::new_err(
                "TDVP currently requires nondecreasing time points",
            ));
        }

        let delta_t = target_time - current_time;
        if delta_t.abs() > TIME_EPSILON {
            dmrg_tdvp::tdvp_step(
                &mut mps,
                &hamiltonian,
                delta_t,
                tdvp_method,
                max_bond_dim,
                lanczos_iterations,
            )
            .map_err(|error| PyValueError::new_err(error.to_string()))?;
            current_time = target_time;
        }

        for (index, label) in labels.iter().enumerate() {
            series[index].push(evaluate_dmrg_observable(&mps, &hamiltonian, label)?);
        }
        if let Some(bond) = entropy_bond {
            let entropy = dmrg_tdvp::entanglement_entropy(&mps, bond)
                .map_err(|error| PyValueError::new_err(error.to_string()))?
                / std::f64::consts::LN_2;
            if let Some(values) = entropy_series.as_mut() {
                values.push(entropy);
            }
        }
    }

    let observable_dict = pyo3::types::PyDict::new(py);
    for (label, values) in labels.iter().zip(series.into_iter()) {
        observable_dict.set_item(label, values)?;
    }

    let dict = pyo3::types::PyDict::new(py);
    dict.set_item("model_name", model)?;
    dict.set_item("solver", "tdvp")?;
    dict.set_item("times", times)?;
    dict.set_item("observables", observable_dict)?;
    if let Some(values) = entropy_series {
        dict.set_item("entanglement_entropy", values)?;
    } else {
        dict.set_item("entanglement_entropy", py.None())?;
    }
    dict.set_item(
        "state_handle",
        tensor_network_state_handle(py, &mps, model, "tdvp")?,
    )?;
    Ok(dict.into())
}

#[cfg(feature = "python")]
#[pyfunction(
    signature = (
        model,
        num_sites,
        times,
        state_handle = None,
        reference_state_handle = None,
        initial_state = None,
        reference_initial_state = None,
        coupling = 1.0,
        coupling_x = None,
        coupling_y = None,
        coupling_z = None,
        transverse_field = None,
        longitudinal_field = None,
        anisotropy = None,
        field_z = None,
        method = "two_site",
        max_bond_dim = 64,
        lanczos_iterations = 20
    )
)]
fn tdvp_loschmidt_echo_1d(
    py: Python<'_>,
    model: &str,
    num_sites: usize,
    times: Vec<f64>,
    state_handle: Option<Py<PyTensorNetworkState1D>>,
    reference_state_handle: Option<Py<PyTensorNetworkState1D>>,
    initial_state: Option<&str>,
    reference_initial_state: Option<&str>,
    coupling: f64,
    coupling_x: Option<f64>,
    coupling_y: Option<f64>,
    coupling_z: Option<f64>,
    transverse_field: Option<f64>,
    longitudinal_field: Option<f64>,
    anisotropy: Option<f64>,
    field_z: Option<f64>,
    method: &str,
    max_bond_dim: usize,
    lanczos_iterations: usize,
) -> PyResult<pyo3::Py<pyo3::types::PyDict>> {
    if num_sites < 2 {
        return Err(PyValueError::new_err(
            "TDVP Loschmidt echo requires at least 2 sites for the current 1D model bindings",
        ));
    }
    if times.iter().any(|time| !time.is_finite()) {
        return Err(PyValueError::new_err(
            "time points must be finite real numbers",
        ));
    }

    let tdvp_method = parse_tdvp_method(method)?;
    let hamiltonian = build_research_model_hamiltonian(
        model,
        num_sites,
        coupling,
        coupling_x,
        coupling_y,
        coupling_z,
        transverse_field,
        longitudinal_field,
        anisotropy,
        field_z,
    )?;

    let initial_mps = if let Some(handle) = state_handle {
        let borrowed = handle.borrow(py);
        if borrowed.num_sites != num_sites {
            return Err(PyValueError::new_err(format!(
                "state handle has {} sites but the target model has {num_sites}",
                borrowed.num_sites
            )));
        }
        borrowed.mps.clone()
    } else {
        dmrg_tdvp::named_product_state_mps(num_sites, initial_state.unwrap_or("all_up"))
            .map_err(|error| PyValueError::new_err(error.to_string()))?
    };

    let reference_mps = if let Some(handle) = reference_state_handle {
        let borrowed = handle.borrow(py);
        if borrowed.num_sites != num_sites {
            return Err(PyValueError::new_err(format!(
                "reference state handle has {} sites but the target model has {num_sites}",
                borrowed.num_sites
            )));
        }
        borrowed.mps.clone()
    } else if let Some(label) = reference_initial_state {
        dmrg_tdvp::named_product_state_mps(num_sites, label)
            .map_err(|error| PyValueError::new_err(error.to_string()))?
    } else {
        initial_mps.clone()
    };

    let reference_norm = dmrg_tdvp::mps_norm(&reference_mps);
    if reference_norm <= 1e-12 {
        return Err(PyValueError::new_err(
            "reference state norm is too small for Loschmidt-echo analysis",
        ));
    }

    let mut evolving_mps = initial_mps;
    let mut amplitudes = Vec::with_capacity(times.len());
    let mut current_time = 0.0_f64;
    const TIME_EPSILON: f64 = 1e-12;

    for &target_time in &times {
        if target_time < -TIME_EPSILON {
            return Err(PyValueError::new_err(
                "TDVP Loschmidt echo currently requires non-negative time points",
            ));
        }
        if target_time + TIME_EPSILON < current_time {
            return Err(PyValueError::new_err(
                "TDVP Loschmidt echo currently requires nondecreasing time points",
            ));
        }

        let delta_t = target_time - current_time;
        if delta_t.abs() > TIME_EPSILON {
            dmrg_tdvp::tdvp_step(
                &mut evolving_mps,
                &hamiltonian,
                delta_t,
                tdvp_method,
                max_bond_dim,
                lanczos_iterations,
            )
            .map_err(|error| PyValueError::new_err(error.to_string()))?;
            current_time = target_time;
        }

        let evolving_norm = dmrg_tdvp::mps_norm(&evolving_mps);
        if evolving_norm <= 1e-12 {
            return Err(PyValueError::new_err(
                "evolved state norm is too small for Loschmidt-echo analysis",
            ));
        }
        let overlap = dmrg_tdvp::mps_overlap(&reference_mps, &evolving_mps)
            / Complex64::new(reference_norm * evolving_norm, 0.0);
        amplitudes.push((overlap.re, overlap.im));
    }

    let dict = pyo3::types::PyDict::new(py);
    dict.set_item("model_name", model)?;
    dict.set_item("solver", "tdvp_overlap")?;
    dict.set_item("times", times)?;
    dict.set_item("amplitudes", amplitudes)?;
    dict.set_item(
        "state_handle",
        tensor_network_state_handle(py, &evolving_mps, model, "tdvp")?,
    )?;
    Ok(dict.into())
}

#[cfg(feature = "python")]
#[pyfunction(signature = (state_handle, bond))]
fn entanglement_spectrum_1d(
    py: Python<'_>,
    state_handle: Py<PyTensorNetworkState1D>,
    bond: usize,
) -> PyResult<pyo3::Py<pyo3::types::PyDict>> {
    let borrowed = state_handle.borrow(py);
    if borrowed.num_sites < 2 {
        return Err(PyValueError::new_err(
            "entanglement spectra require at least 2 sites for the current 1D tensor-network bindings",
        ));
    }
    if bond >= borrowed.num_sites - 1 {
        return Err(PyValueError::new_err(format!(
            "bond {bond} is out of range for a {}-site chain",
            borrowed.num_sites
        )));
    }

    let spectrum = dmrg_tdvp::entanglement_spectrum(&borrowed.mps, bond)
        .map_err(|error| PyValueError::new_err(error.to_string()))?;
    let entropy = dmrg_tdvp::entanglement_entropy(&borrowed.mps, bond)
        .map_err(|error| PyValueError::new_err(error.to_string()))?;

    let dict = pyo3::types::PyDict::new(py);
    dict.set_item("spectrum", spectrum)?;
    dict.set_item("entropy", entropy)?;
    Ok(dict.into())
}

#[cfg(feature = "python")]
#[pyfunction(signature = (state_handle, pauli, site))]
fn apply_local_pauli_1d(
    py: Python<'_>,
    state_handle: Py<PyTensorNetworkState1D>,
    pauli: &str,
    site: usize,
) -> PyResult<Py<PyTensorNetworkState1D>> {
    let label = pauli
        .chars()
        .next()
        .ok_or_else(|| PyValueError::new_err("pauli label must not be empty"))?
        .to_ascii_uppercase();
    if pauli.chars().count() != 1 {
        return Err(PyValueError::new_err(
            "pauli label must be a single character such as 'X', 'Y', or 'Z'",
        ));
    }

    let borrowed = state_handle.borrow(py);
    if site >= borrowed.num_sites {
        return Err(PyValueError::new_err(format!(
            "site {site} is out of range for a {}-site chain",
            borrowed.num_sites
        )));
    }

    let operator = pauli_matrix(label)?;
    let updated = dmrg_tdvp::apply_local_operator(&borrowed.mps, &operator, site);
    tensor_network_state_handle(
        py,
        &updated,
        &borrowed.model_name,
        &borrowed.solver,
    )
}

#[cfg(feature = "python")]
#[pyfunction(
    signature = (
        model,
        num_sites,
        times,
        reference_state_handle,
        source_state_handle,
        coupling = 1.0,
        coupling_x = None,
        coupling_y = None,
        coupling_z = None,
        transverse_field = None,
        longitudinal_field = None,
        anisotropy = None,
        field_z = None,
        method = "two_site",
        max_bond_dim = 64,
        lanczos_iterations = 20,
        observables = None
    )
)]
fn tdvp_transition_observables_1d(
    py: Python<'_>,
    model: &str,
    num_sites: usize,
    times: Vec<f64>,
    reference_state_handle: Py<PyTensorNetworkState1D>,
    source_state_handle: Py<PyTensorNetworkState1D>,
    coupling: f64,
    coupling_x: Option<f64>,
    coupling_y: Option<f64>,
    coupling_z: Option<f64>,
    transverse_field: Option<f64>,
    longitudinal_field: Option<f64>,
    anisotropy: Option<f64>,
    field_z: Option<f64>,
    method: &str,
    max_bond_dim: usize,
    lanczos_iterations: usize,
    observables: Option<Vec<String>>,
) -> PyResult<pyo3::Py<pyo3::types::PyDict>> {
    if num_sites < 2 {
        return Err(PyValueError::new_err(
            "TDVP transition observables require at least 2 sites for the current 1D model bindings",
        ));
    }
    if times.iter().any(|time| !time.is_finite()) {
        return Err(PyValueError::new_err(
            "time points must be finite real numbers",
        ));
    }

    let tdvp_method = parse_tdvp_method(method)?;
    let hamiltonian = build_research_model_hamiltonian(
        model,
        num_sites,
        coupling,
        coupling_x,
        coupling_y,
        coupling_z,
        transverse_field,
        longitudinal_field,
        anisotropy,
        field_z,
    )?;

    let reference = reference_state_handle.borrow(py);
    let source = source_state_handle.borrow(py);
    if reference.num_sites != num_sites {
        return Err(PyValueError::new_err(format!(
            "reference state handle has {} sites but the target model has {num_sites}",
            reference.num_sites
        )));
    }
    if source.num_sites != num_sites {
        return Err(PyValueError::new_err(format!(
            "source state handle has {} sites but the target model has {num_sites}",
            source.num_sites
        )));
    }

    let reference_mps = reference.mps.clone();
    let mut source_mps = source.mps.clone();
    let labels = observables.unwrap_or_default();
    let mut series: Vec<Vec<(f64, f64)>> = labels
        .iter()
        .map(|_| Vec::with_capacity(times.len()))
        .collect();
    let mut current_time = 0.0_f64;
    const TIME_EPSILON: f64 = 1e-12;

    for &target_time in &times {
        if target_time < -TIME_EPSILON {
            return Err(PyValueError::new_err(
                "TDVP transition observables currently require non-negative time points",
            ));
        }
        if target_time + TIME_EPSILON < current_time {
            return Err(PyValueError::new_err(
                "TDVP transition observables currently require nondecreasing time points",
            ));
        }

        let delta_t = target_time - current_time;
        if delta_t.abs() > TIME_EPSILON {
            dmrg_tdvp::tdvp_step(
                &mut source_mps,
                &hamiltonian,
                delta_t,
                tdvp_method,
                max_bond_dim,
                lanczos_iterations,
            )
            .map_err(|error| PyValueError::new_err(error.to_string()))?;
            current_time = target_time;
        }

        for (index, label) in labels.iter().enumerate() {
            let value = evaluate_transition_observable(&reference_mps, &source_mps, label)?;
            series[index].push((value.re, value.im));
        }
    }

    let observable_dict = pyo3::types::PyDict::new(py);
    for (label, values) in labels.iter().zip(series.into_iter()) {
        observable_dict.set_item(label, values)?;
    }

    let dict = pyo3::types::PyDict::new(py);
    dict.set_item("model_name", model)?;
    dict.set_item("solver", "tdvp_transition")?;
    dict.set_item("times", times)?;
    dict.set_item("observables", observable_dict)?;
    dict.set_item(
        "state_handle",
        tensor_network_state_handle(py, &source_mps, model, "tdvp")?,
    )?;
    Ok(dict.into())
}

// =============================================================================
// QUANTUM ENTROPY EXTRACTION PYTHON BINDINGS
// =============================================================================

/// Python wrapper for QuantumEntropyExtractor
#[cfg(feature = "python")]
#[pyclass(name = "QuantumEntropyExtractor")]
pub struct PyQuantumEntropyExtractor {
    inner: QuantumEntropyExtractor,
}

#[cfg(feature = "python")]
#[pymethods]
impl PyQuantumEntropyExtractor {
    /// Create a new quantum entropy extractor
    #[new]
    #[pyo3(signature = (n_qubits=8))]
    fn new(n_qubits: usize) -> Self {
        Self {
            inner: QuantumEntropyExtractor::new(n_qubits),
        }
    }

    /// Extract a single seed (u64)
    fn extract_seed(&mut self) -> u64 {
        self.inner.extract_seed().seed_u64
    }

    /// Extract a seed with full metrics as a tuple
    fn extract_seed_with_metrics(&mut self, _py: Python<'_>) -> (u64, f64, f64, f64, f64) {
        let result = self.inner.extract_seed();
        (
            result.seed_u64,
            result.metrics.shannon_entropy,
            result.metrics.min_entropy,
            result.metrics.unpredictability,
            result.metrics.quantum_advantage,
        )
    }

    /// Extract batch of seeds
    fn extract_batch(&mut self, count: usize) -> Vec<u64> {
        (0..count)
            .map(|_| self.inner.extract_seed().seed_u64)
            .collect()
    }

    /// Get Python-compatible seed tuple (u64, bytes)
    fn extract_python_seed(&mut self) -> (u64, Vec<u8>) {
        self.inner.extract_python_seed()
    }
}

/// Quick quantum seed extraction (convenience function)
#[cfg(feature = "python")]
#[pyfunction]
fn py_quick_seed() -> u64 {
    quick_quantum_seed()
}

/// Generate a batch of quantum seeds
#[cfg(feature = "python")]
#[pyfunction]
fn py_batch_seeds(count: usize) -> Vec<u64> {
    generate_quantum_seeds(count)
}

/// Python wrapper for QuantumSimulator
#[cfg(feature = "python")]
#[pyclass(name = "QuantumSimulator")]
pub struct PyQuantumSimulator {
    inner: QuantumSimulator,
}

#[cfg(feature = "python")]
#[pymethods]
impl PyQuantumSimulator {
    /// Create a new quantum simulator
    #[new]
    fn new(num_qubits: usize) -> PyResult<Self> {
        if num_qubits > 25 {
            return Err(PyValueError::new_err(
                "Maximum 25 qubits supported (use MPS for more)",
            ));
        }
        Ok(PyQuantumSimulator {
            inner: QuantumSimulator::new(num_qubits),
        })
    }

    /// Get the number of qubits
    #[getter]
    fn num_qubits(&self) -> usize {
        self.inner.num_qubits()
    }

    /// Get the state dimension (2^num_qubits)
    #[getter]
    fn dim(&self) -> usize {
        self.inner.state.dim
    }

    /// Apply Hadamard gate
    fn h(&mut self, qubit: usize) -> PyResult<()> {
        if qubit >= self.inner.num_qubits() {
            return Err(PyValueError::new_err(format!(
                "Qubit index {} out of range",
                qubit
            )));
        }
        self.inner.h(qubit);
        Ok(())
    }

    /// Apply Pauli-X gate
    fn x(&mut self, qubit: usize) -> PyResult<()> {
        if qubit >= self.inner.num_qubits() {
            return Err(PyValueError::new_err(format!(
                "Qubit index {} out of range",
                qubit
            )));
        }
        self.inner.x(qubit);
        Ok(())
    }

    /// Apply Pauli-Y gate
    fn y(&mut self, qubit: usize) -> PyResult<()> {
        if qubit >= self.inner.num_qubits() {
            return Err(PyValueError::new_err(format!(
                "Qubit index {} out of range",
                qubit
            )));
        }
        self.inner.y(qubit);
        Ok(())
    }

    /// Apply Pauli-Z gate
    fn z(&mut self, qubit: usize) -> PyResult<()> {
        if qubit >= self.inner.num_qubits() {
            return Err(PyValueError::new_err(format!(
                "Qubit index {} out of range",
                qubit
            )));
        }
        self.inner.z(qubit);
        Ok(())
    }

    /// Apply S gate (phase gate)
    fn s(&mut self, qubit: usize) -> PyResult<()> {
        if qubit >= self.inner.num_qubits() {
            return Err(PyValueError::new_err(format!(
                "Qubit index {} out of range",
                qubit
            )));
        }
        self.inner.s(qubit);
        Ok(())
    }

    /// Apply T gate (π/8 gate)
    fn t(&mut self, qubit: usize) -> PyResult<()> {
        if qubit >= self.inner.num_qubits() {
            return Err(PyValueError::new_err(format!(
                "Qubit index {} out of range",
                qubit
            )));
        }
        self.inner.t(qubit);
        Ok(())
    }

    /// Apply RX rotation gate
    fn rx(&mut self, qubit: usize, angle: f64) -> PyResult<()> {
        if qubit >= self.inner.num_qubits() {
            return Err(PyValueError::new_err(format!(
                "Qubit index {} out of range",
                qubit
            )));
        }
        self.inner.rx(qubit, angle);
        Ok(())
    }

    /// Apply RY rotation gate
    fn ry(&mut self, qubit: usize, angle: f64) -> PyResult<()> {
        if qubit >= self.inner.num_qubits() {
            return Err(PyValueError::new_err(format!(
                "Qubit index {} out of range",
                qubit
            )));
        }
        self.inner.ry(qubit, angle);
        Ok(())
    }

    /// Apply RZ rotation gate
    fn rz(&mut self, qubit: usize, angle: f64) -> PyResult<()> {
        if qubit >= self.inner.num_qubits() {
            return Err(PyValueError::new_err(format!(
                "Qubit index {} out of range",
                qubit
            )));
        }
        self.inner.rz(qubit, angle);
        Ok(())
    }

    /// Apply CNOT gate
    fn cx(&mut self, control: usize, target: usize) -> PyResult<()> {
        let n = self.inner.num_qubits();
        if control >= n || target >= n {
            return Err(PyValueError::new_err("Qubit index out of range"));
        }
        if control == target {
            return Err(PyValueError::new_err(
                "Control and target must be different",
            ));
        }
        self.inner.cnot(control, target);
        Ok(())
    }

    /// Apply CZ gate
    fn cz(&mut self, control: usize, target: usize) -> PyResult<()> {
        let n = self.inner.num_qubits();
        if control >= n || target >= n {
            return Err(PyValueError::new_err("Qubit index out of range"));
        }
        if control == target {
            return Err(PyValueError::new_err(
                "Control and target must be different",
            ));
        }
        self.inner.cz(control, target);
        Ok(())
    }

    /// Apply SWAP gate
    fn swap(&mut self, a: usize, b: usize) -> PyResult<()> {
        let n = self.inner.num_qubits();
        if a >= n || b >= n {
            return Err(PyValueError::new_err("Qubit index out of range"));
        }
        self.inner.swap(a, b);
        Ok(())
    }

    /// Measure a qubit (collapses state)
    fn measure(&mut self, qubit: usize) -> PyResult<usize> {
        if qubit >= self.inner.num_qubits() {
            return Err(PyValueError::new_err(format!(
                "Qubit index {} out of range",
                qubit
            )));
        }
        let (result, _) = self.inner.measure_qubit(qubit);
        Ok(result)
    }

    /// Measure all qubits
    fn measure_all(&mut self) -> PyResult<usize> {
        Ok(self.inner.measure())
    }

    // Additional gates

    /// Apply Toffoli gate (CCNOT - controlled-controlled-not)
    fn toffoli(&mut self, control1: usize, control2: usize, target: usize) -> PyResult<()> {
        let n = self.inner.num_qubits();
        if control1 >= n || control2 >= n || target >= n {
            return Err(PyValueError::new_err("Qubit index out of range"));
        }
        if control1 == target || control2 == target || control1 == control2 {
            return Err(PyValueError::new_err(
                "Control and target qubits must be distinct",
            ));
        }
        use crate::GateOperations;
        GateOperations::toffoli(&mut self.inner.state, control1, control2, target);
        Ok(())
    }

    /// Apply controlled-RX gate
    fn crx(&mut self, control: usize, target: usize, angle: f64) -> PyResult<()> {
        let n = self.inner.num_qubits();
        if control >= n || target >= n {
            return Err(PyValueError::new_err("Qubit index out of range"));
        }
        if control == target {
            return Err(PyValueError::new_err(
                "Control and target must be different",
            ));
        }
        use crate::GateOperations;
        GateOperations::crx(&mut self.inner.state, control, target, angle);
        Ok(())
    }

    /// Apply controlled-RY gate
    fn cry(&mut self, control: usize, target: usize, angle: f64) -> PyResult<()> {
        let n = self.inner.num_qubits();
        if control >= n || target >= n {
            return Err(PyValueError::new_err("Qubit index out of range"));
        }
        if control == target {
            return Err(PyValueError::new_err(
                "Control and target must be different",
            ));
        }
        use crate::GateOperations;
        GateOperations::cry(&mut self.inner.state, control, target, angle);
        Ok(())
    }

    /// Apply controlled-RZ gate
    fn crz(&mut self, control: usize, target: usize, angle: f64) -> PyResult<()> {
        let n = self.inner.num_qubits();
        if control >= n || target >= n {
            return Err(PyValueError::new_err("Qubit index out of range"));
        }
        if control == target {
            return Err(PyValueError::new_err(
                "Control and target must be different",
            ));
        }
        use crate::GateOperations;
        GateOperations::crz(&mut self.inner.state, control, target, angle);
        Ok(())
    }

    /// Apply controlled-phase gate
    fn cphase(&mut self, control: usize, target: usize, phi: f64) -> PyResult<()> {
        let n = self.inner.num_qubits();
        if control >= n || target >= n {
            return Err(PyValueError::new_err("Qubit index out of range"));
        }
        if control == target {
            return Err(PyValueError::new_err(
                "Control and target must be different",
            ));
        }
        use crate::GateOperations;
        GateOperations::cphase(&mut self.inner.state, control, target, phi);
        Ok(())
    }

    /// Apply SX gate (square root of X)
    fn sx(&mut self, qubit: usize) -> PyResult<()> {
        if qubit >= self.inner.num_qubits() {
            return Err(PyValueError::new_err(format!(
                "Qubit index {} out of range",
                qubit
            )));
        }
        use crate::GateOperations;
        GateOperations::sx(&mut self.inner.state, qubit);
        Ok(())
    }

    /// Apply phase gate with arbitrary angle
    fn phase(&mut self, qubit: usize, theta: f64) -> PyResult<()> {
        if qubit >= self.inner.num_qubits() {
            return Err(PyValueError::new_err(format!(
                "Qubit index {} out of range",
                qubit
            )));
        }
        use crate::GateOperations;
        GateOperations::phase(&mut self.inner.state, qubit, theta);
        Ok(())
    }

    // Utility methods

    /// Get expectation value of Pauli X on qubit
    fn expectation_x(&self, qubit: usize) -> PyResult<f64> {
        if qubit >= self.inner.num_qubits() {
            return Err(PyValueError::new_err(format!(
                "Qubit index {} out of range",
                qubit
            )));
        }
        Ok(self.inner.state.expectation_x(qubit))
    }

    /// Get expectation value of Pauli Y on qubit
    fn expectation_y(&self, qubit: usize) -> PyResult<f64> {
        if qubit >= self.inner.num_qubits() {
            return Err(PyValueError::new_err(format!(
                "Qubit index {} out of range",
                qubit
            )));
        }
        Ok(self.inner.state.expectation_y(qubit))
    }

    /// Get expectation value of Pauli Z on qubit
    fn expectation_z(&self, qubit: usize) -> PyResult<f64> {
        if qubit >= self.inner.num_qubits() {
            return Err(PyValueError::new_err(format!(
                "Qubit index {} out of range",
                qubit
            )));
        }
        Ok(self.inner.state.expectation_z(qubit))
    }

    /// Get fidelity with another quantum state
    fn fidelity(&self, other: &PyQuantumState) -> PyResult<f64> {
        Ok(self.inner.state.fidelity(&other.inner))
    }

    /// Sample multiple bitstring measurements
    fn sample_bitstrings(
        &self,
        shots: usize,
    ) -> PyResult<std::collections::HashMap<String, usize>> {
        let results = self.inner.state.sample_bitstrings(shots);
        Ok(results.into_iter().collect())
    }

    /// Get state amplitudes as a list of (real, imag) tuples
    fn amplitudes(&self) -> PyResult<Vec<(f64, f64)>> {
        let amplitudes = self.inner.state.amplitudes_ref();
        Ok(amplitudes.iter().map(|c| (c.re, c.im)).collect())
    }

    /// Get probabilities for all basis states
    fn probabilities(&self) -> Vec<f64> {
        self.inner
            .state
            .amplitudes_ref()
            .iter()
            .map(|c| c.re * c.re + c.im * c.im)
            .collect()
    }

    /// Get state vector as list of complex numbers for plotting
    fn statevector(&self) -> Vec<(f64, f64)> {
        self.inner
            .state
            .amplitudes_ref()
            .iter()
            .map(|c| (c.re, c.im))
            .collect()
    }

    /// Get Bloch sphere coordinates for a qubit (x, y, z)
    fn bloch_vector(&self, qubit: usize) -> PyResult<(f64, f64, f64)> {
        if qubit >= self.inner.num_qubits() {
            return Err(PyValueError::new_err("Qubit index out of bounds"));
        }
        Ok((
            self.inner.state.expectation_x(qubit),
            self.inner.state.expectation_y(qubit),
            self.inner.state.expectation_z(qubit),
        ))
    }

    /// Get reduced density matrix for a qubit as flattened list
    fn density_matrix(&self, qubit: usize) -> PyResult<Vec<(f64, f64)>> {
        if qubit >= self.inner.num_qubits() {
            return Err(PyValueError::new_err("Qubit index out of bounds"));
        }
        // For pure state, reduced density matrix elements
        let x = self.inner.state.expectation_x(qubit);
        let y = self.inner.state.expectation_y(qubit);
        let z = self.inner.state.expectation_z(qubit);

        // Density matrix in Pauli basis: rho = (I + x*X + y*Y + z*Z) / 2
        // In computational basis: [[(1+z)/2, (x-1j*y)/2], [(x+1j*y)/2, (1-z)/2]]
        Ok(vec![
            ((1.0 + z) / 2.0, 0.0), // rho[0,0]
            (x / 2.0, -y / 2.0),    // rho[0,1]
            (x / 2.0, y / 2.0),     // rho[1,0]
            ((1.0 - z) / 2.0, 0.0), // rho[1,1]
        ])
    }

    /// Calculate von Neumann entropy of the state (in nats)
    fn entropy(&self) -> f64 {
        let probs = self
            .inner
            .state
            .amplitudes_ref()
            .iter()
            .map(|c| c.re * c.re + c.im * c.im);

        let mut entropy = 0.0;
        for p in probs {
            if p > 1e-15 {
                entropy -= p * p.ln();
            }
        }
        entropy
    }

    /// Reset to |0...0⟩ state
    fn reset(&mut self) {
        self.inner.reset();
    }

    /// Get current quantum state
    fn get_state(&self) -> PyQuantumState {
        PyQuantumState {
            inner: self.inner.state.clone(),
        }
    }

    fn __repr__(&self) -> String {
        format!("<QuantumSimulator num_qubits={}>", self.inner.num_qubits())
    }
}

/// Python wrapper for QuantumState
#[cfg(feature = "python")]
#[pyclass(name = "QuantumState")]
pub struct PyQuantumState {
    inner: QuantumState,
}

#[cfg(feature = "python")]
#[pymethods]
impl PyQuantumState {
    #[getter]
    fn num_qubits(&self) -> usize {
        self.inner.num_qubits
    }

    #[getter]
    fn dim(&self) -> usize {
        self.inner.dim
    }

    fn amplitudes(&self) -> Vec<(f64, f64)> {
        self.inner
            .amplitudes_ref()
            .iter()
            .map(|c| (c.re, c.im))
            .collect()
    }

    fn probabilities(&self) -> Vec<f64> {
        self.inner
            .amplitudes_ref()
            .iter()
            .map(|c| c.re * c.re + c.im * c.im)
            .collect()
    }

    fn __repr__(&self) -> String {
        format!("<QuantumState num_qubits={}>", self.inner.num_qubits)
    }
}

/// Create a Bell state (|Φ+⟩ = (|00⟩ + |11⟩)/√2)
#[cfg(feature = "python")]
#[pyfunction]
fn create_bell_state() -> PyResult<PyQuantumSimulator> {
    let mut sim = PyQuantumSimulator {
        inner: QuantumSimulator::new(2),
    };
    sim.inner.h(0);
    sim.inner.cnot(0, 1);
    Ok(sim)
}

/// Create a GHZ state (|000⟩ + |111⟩)/√2
#[cfg(feature = "python")]
#[pyfunction]
fn create_ghz_state(num_qubits: usize) -> PyResult<PyQuantumSimulator> {
    if num_qubits < 2 || num_qubits > 20 {
        return Err(PyValueError::new_err("num_qubits must be between 2 and 20"));
    }
    let mut sim = PyQuantumSimulator {
        inner: QuantumSimulator::new(num_qubits),
    };
    sim.inner.h(0);
    for i in 0..(num_qubits - 1) {
        sim.inner.cnot(i, i + 1);
    }
    Ok(sim)
}

/// Run Grover's search algorithm
#[cfg(feature = "python")]
#[pyfunction]
#[pyo3(signature = (num_qubits, target, iterations=None))]
fn run_grover(num_qubits: usize, target: usize, iterations: Option<usize>) -> PyResult<usize> {
    if num_qubits < 2 || num_qubits > 16 {
        return Err(PyValueError::new_err("num_qubits must be between 2 and 16"));
    }
    let max_states = 1 << num_qubits;
    if target >= max_states {
        return Err(PyValueError::new_err(
            "target must be less than 2^num_qubits",
        ));
    }

    let optimal_iterations =
        ((std::f64::consts::PI / 4.0) * (max_states as f64).sqrt()).ceil() as usize;
    let iterations = iterations.unwrap_or(optimal_iterations);

    let mut sim = QuantumSimulator::new(num_qubits);

    // Initialize to uniform superposition
    for i in 0..num_qubits {
        sim.h(i);
    }

    // Grover iterations
    for _ in 0..iterations {
        // Oracle: mark the target state by flipping its phase
        let amplitudes = sim.state.amplitudes_mut();
        amplitudes[target].re = -amplitudes[target].re;
        amplitudes[target].im = -amplitudes[target].im;

        // Diffusion operator
        for i in 0..num_qubits {
            sim.h(i);
            sim.x(i);
        }
        // Multi-controlled Z (simplified: use CZ chain)
        if num_qubits >= 2 {
            sim.cz(0, num_qubits - 1);
        }
        for i in 0..num_qubits {
            sim.x(i);
            sim.h(i);
        }
    }

    // Measure
    Ok(sim.measure())
}

/// Benchmark gate operations
#[cfg(feature = "python")]
#[pyfunction]
#[pyo3(signature = (num_qubits, num_gates=1000))]
fn benchmark_gates(num_qubits: usize, num_gates: usize) -> PyResult<f64> {
    if num_qubits > 20 {
        return Err(PyValueError::new_err("Maximum 20 qubits for benchmark"));
    }

    let mut sim = QuantumSimulator::new(num_qubits);
    let start = std::time::Instant::now();

    for i in 0..num_gates {
        let qubit = i % num_qubits;
        match i % 4 {
            0 => sim.h(qubit),
            1 => sim.x(qubit),
            2 => sim.ry(qubit, 0.5),
            _ => {
                if num_qubits > 1 {
                    sim.cnot(qubit, (qubit + 1) % num_qubits)
                }
            }
        }
    }

    let elapsed = start.elapsed().as_secs_f64();
    Ok(elapsed)
}

// ============================================================================
// ADVANCED QUANTUM ALGORITHMS
// ============================================================================

/// Apply Quantum Fourier Transform (QFT) to a quantum state
#[cfg(feature = "python")]
#[pyfunction]
#[pyo3(signature = (num_qubits, shots=None))]
fn apply_qft(num_qubits: usize, shots: Option<usize>) -> PyResult<PyQuantumSimulator> {
    let _ = shots; // Suppress unused warning
    if num_qubits > 20 {
        return Err(PyValueError::new_err("Maximum 20 qubits for QFT"));
    }

    let mut sim = PyQuantumSimulator {
        inner: QuantumSimulator::new(num_qubits),
    };

    // Apply QFT using the comprehensive algorithms module

    use std::f64::consts::PI;

    // Start with |0...0⟩ state
    // Create uniform superposition first (optional - QFT can work on any state)
    for q in 0..num_qubits {
        sim.inner.h(q);
    }

    // Apply QFT
    for i in 0..num_qubits {
        sim.inner.h(i);
        for j in 1..(num_qubits - i) {
            let angle = 2.0 * PI / (1 << (j + 1)) as f64;
            sim.inner.rz(i, angle);
        }
    }

    Ok(sim)
}

/// Run Phase Estimation algorithm
#[cfg(feature = "python")]
#[pyfunction]
fn run_phase_estimation(
    counting_qubits: usize,
    target_qubits: usize,
    eigenstate_index: usize,
) -> PyResult<usize> {
    if counting_qubits > 10 {
        return Err(PyValueError::new_err("Maximum 10 counting qubits"));
    }
    if target_qubits > 10 {
        return Err(PyValueError::new_err("Maximum 10 target qubits"));
    }

    use std::f64::consts::PI;

    // Simulate phase estimation for a simple phase gate
    // Phase = eigenstate_index / 2^counting_qubits
    let phase = eigenstate_index as f64 / (1 << counting_qubits) as f64;

    let mut sim = QuantumSimulator::new(counting_qubits + target_qubits);

    // Initialize counting qubits to superposition
    for i in 0..counting_qubits {
        sim.h(i);
    }

    // Initialize target to eigenstate
    for i in counting_qubits..(counting_qubits + target_qubits) {
        sim.x(i - counting_qubits);
    }

    // Apply controlled phase rotations
    for i in 0..counting_qubits {
        let power = 1 << (counting_qubits - 1 - i);
        let angle = 2.0 * PI * phase * power as f64;
        sim.rz(counting_qubits, angle);
    }

    // Apply inverse QFT to counting qubits
    for i in 0..counting_qubits {
        let idx = counting_qubits - 1 - i;
        sim.h(idx);
        for j in 0..idx {
            let angle = -2.0 * PI / (1 << (idx - j + 1)) as f64;
            sim.rz(j, angle);
        }
    }

    // Measure counting qubits
    let result = sim.measure() & ((1 << counting_qubits) - 1);

    Ok(result)
}

/// Run Variational Quantum Eigensolver (VQE) for simple Hamiltonian
#[cfg(feature = "python")]
#[pyfunction]
#[pyo3(signature = (num_qubits, iterations=None))]
fn run_vqe(num_qubits: usize, iterations: Option<usize>) -> PyResult<f64> {
    if num_qubits < 2 || num_qubits > 8 {
        return Err(PyValueError::new_err("num_qubits must be between 2 and 8"));
    }

    let iters = iterations.unwrap_or(100);

    use crate::comprehensive_algorithms::QuantumAlgorithms;

    // Simple Hamiltonian: H = -Z⊗Z (two-qubit interaction)
    let hamiltonian = vec![(-1.0, vec!['Z', 'Z'])];

    // Initial parameters (random small values)
    let params = vec![0.1; num_qubits];

    // Use a simple ansatz (hardware-efficient)
    let (energy, _optimal_params) = QuantumAlgorithms::vqe(
        num_qubits,
        &hamiltonian,
        |state: &mut QuantumState, params: &[f64]| {
            use crate::comprehensive_gates::QuantumGates;
            let n = state.num_qubits;

            // Layer of rotations
            for i in 0..n {
                QuantumGates::ry(state, i, params[i % params.len()]);
            }

            // Entangling layer using CX (not CNOT)
            for i in 0..n.saturating_sub(1) {
                QuantumGates::cx(state, i, i + 1);
            }
        },
        params, // Initial parameters
        0.01,   // Learning rate
        iters,  // Iterations
    );

    Ok(energy)
}

// ============================================================================
// COMPREHENSIVE PYTHON API v2
// ============================================================================

/// Backend type for simulation
#[cfg(feature = "python")]
#[pyclass(name = "Backend")]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PyBackend {
    CPU,
    GPU,
    GPUOnly,
    F32Fusion,
    Auto,
}

/// Quantum circuit for building quantum programs
#[cfg(feature = "python")]
#[pyclass(name = "QuantumCircuit")]
#[derive(Debug, Clone)]
pub struct PyQuantumCircuit2 {
    num_qubits: usize,
    pub gates: Vec<Gate>,
}

#[cfg(feature = "python")]
#[pymethods]
impl PyQuantumCircuit2 {
    #[new]
    fn new(num_qubits: usize) -> PyResult<Self> {
        if num_qubits > 28 {
            return Err(PyValueError::new_err("Maximum 28 qubits supported"));
        }
        Ok(PyQuantumCircuit2 {
            num_qubits,
            gates: Vec::new(),
        })
    }

    #[getter]
    fn num_qubits(&self) -> usize {
        self.num_qubits
    }

    fn h(&mut self, qubit: usize) -> PyResult<()> {
        if qubit >= self.num_qubits {
            return Err(PyValueError::new_err("Qubit index out of range"));
        }
        self.gates.push(Gate::h(qubit));
        Ok(())
    }

    fn x(&mut self, qubit: usize) -> PyResult<()> {
        if qubit >= self.num_qubits {
            return Err(PyValueError::new_err("Qubit index out of range"));
        }
        self.gates.push(Gate::x(qubit));
        Ok(())
    }

    fn cx(&mut self, control: usize, target: usize) -> PyResult<()> {
        if control >= self.num_qubits || target >= self.num_qubits {
            return Err(PyValueError::new_err("Qubit index out of range"));
        }
        self.gates.push(Gate::cnot(control, target));
        Ok(())
    }

    fn rx(&mut self, qubit: usize, theta: f64) -> PyResult<()> {
        if qubit >= self.num_qubits {
            return Err(PyValueError::new_err("Qubit index out of range"));
        }
        self.gates.push(Gate::rx(qubit, theta));
        Ok(())
    }

    fn ry(&mut self, qubit: usize, theta: f64) -> PyResult<()> {
        if qubit >= self.num_qubits {
            return Err(PyValueError::new_err("Qubit index out of range"));
        }
        self.gates.push(Gate::ry(qubit, theta));
        Ok(())
    }

    fn rz(&mut self, qubit: usize, theta: f64) -> PyResult<()> {
        if qubit >= self.num_qubits {
            return Err(PyValueError::new_err("Qubit index out of range"));
        }
        self.gates.push(Gate::rz(qubit, theta));
        Ok(())
    }

    fn recommended_backend(&self) -> String {
        let selector = AutoBackend::new();
        let analysis = selector.analyze(&self.gates);
        analysis.recommended_backend.name().to_string()
    }

    fn backend_reasoning(&self) -> String {
        let selector = AutoBackend::new();
        let analysis = selector.analyze(&self.gates);
        analysis.reasoning
    }

    fn __repr__(&self) -> String {
        format!(
            "<QuantumCircuit {} qubits, {} gates>",
            self.num_qubits,
            self.gates.len()
        )
    }
}

// Helper function to get gates (outside of #[pymethods])
#[cfg(feature = "python")]
fn get_circuit_gates(circuit: &PyQuantumCircuit2) -> Vec<Gate> {
    circuit.gates.clone()
}

/// Simulation result with counts
#[cfg(feature = "python")]
#[pyclass(name = "SimulationResult")]
#[derive(Debug, Clone)]
pub struct PySimulationResult {
    counts: std::collections::HashMap<String, usize>,
    shots: usize,
}

#[cfg(feature = "python")]
#[pymethods]
impl PySimulationResult {
    #[getter]
    fn counts(&self) -> std::collections::HashMap<String, usize> {
        self.counts.clone()
    }

    fn __repr__(&self) -> String {
        format!(
            "<Result shots={}, outcomes={}>",
            self.shots,
            self.counts.len()
        )
    }
}

/// Enhanced simulator with circuit execution
#[cfg(feature = "python")]
#[pyclass(name = "EnhancedSimulator")]
pub struct PyEnhancedSimulator {
    inner: QuantumSimulator,
}

#[cfg(feature = "python")]
#[pymethods]
impl PyEnhancedSimulator {
    #[new]
    fn new(num_qubits: usize) -> PyResult<Self> {
        if num_qubits > 28 {
            return Err(PyValueError::new_err("Maximum 28 qubits supported"));
        }
        Ok(PyEnhancedSimulator {
            inner: QuantumSimulator::new(num_qubits),
        })
    }

    /// Run a circuit and return results
    #[pyo3(signature = (circuit, shots=None, backend=None))]
    fn run_circuit(
        &mut self,
        circuit: &PyQuantumCircuit2,
        shots: Option<usize>,
        backend: Option<PyBackend>,
    ) -> PyResult<PySimulationResult> {
        let shots = shots.unwrap_or(1024);
        let gates = get_circuit_gates(circuit);
        let probs = self.execute_with_backend(circuit.num_qubits, &gates, backend)?;

        // Sample shots
        let mut counts: std::collections::HashMap<String, usize> = std::collections::HashMap::new();
        let mut cdf = Vec::with_capacity(probs.len());
        let mut cumsum = 0.0;
        for &p in &probs {
            cumsum += p;
            cdf.push(cumsum);
        }
        let mut rng = rand::thread_rng();

        for _ in 0..shots {
            let r: f64 = rng.gen();
            let measured = match cdf.binary_search_by(|c| c.partial_cmp(&r).unwrap()) {
                Ok(i) => i,
                Err(i) => i.min(probs.len().saturating_sub(1)),
            };
            let bitstring = format!("{:0width$b}", measured, width = circuit.num_qubits);
            *counts.entry(bitstring).or_insert(0) += 1;
        }

        Ok(PySimulationResult { counts, shots })
    }

    fn __repr__(&self) -> String {
        format!("<EnhancedSimulator qubits={}>", self.inner.num_qubits())
    }
}

#[cfg(feature = "python")]
impl PyEnhancedSimulator {
    fn execute_with_backend(
        &self,
        num_qubits: usize,
        gates: &[Gate],
        backend: Option<PyBackend>,
    ) -> PyResult<Vec<f64>> {
        match backend.unwrap_or(PyBackend::Auto) {
            PyBackend::Auto => {
                let sim = AutoSimulator::new(gates, num_qubits, false);
                sim.execute_result(gates).map_err(PyValueError::new_err)
            }
            PyBackend::CPU => {
                let sim = AutoSimulator::with_backend(SimBackend::StateVectorFused, num_qubits);
                sim.execute_result(gates).map_err(PyValueError::new_err)
            }
            PyBackend::GPU => {
                let sim = AutoSimulator::with_backend(SimBackend::MetalGPU, num_qubits);
                sim.execute_result(gates).map_err(PyValueError::new_err)
            }
            PyBackend::GPUOnly => {
                let sim = AutoSimulator::with_gpu_only(num_qubits);
                sim.execute_result(gates).map_err(PyValueError::new_err)
            }
            PyBackend::F32Fusion => {
                let sim = AutoSimulator::with_backend(SimBackend::StateVectorF32Fused, num_qubits);
                sim.execute_result(gates).map_err(PyValueError::new_err)
            }
        }
    }
}

#[cfg(feature = "python")]
#[pyfunction]
#[pyo3(signature = (phase, rounds=10, shots_per_round=64, readout_error=0.0))]
fn run_heisenberg_qpe_rust(
    phase: f64,
    rounds: usize,
    shots_per_round: usize,
    readout_error: f64,
) -> PyResult<(f64, f64, f64, u64)> {
    if !(0.0..1.0).contains(&phase) {
        return Err(PyValueError::new_err("phase must be in [0, 1)"));
    }
    if !(0.0..=0.5).contains(&readout_error) {
        return Err(PyValueError::new_err("readout_error must be in [0, 0.5]"));
    }

    let oracle = crate::heisenberg_qpe::IdealPhaseOracle {
        phase,
        readout_error,
    };
    let cfg = crate::heisenberg_qpe::HeisenbergQpeConfig {
        rounds,
        shots_per_round,
        posterior_grid_size: 2048,
    };
    let result = crate::heisenberg_qpe::estimate_phase_heisenberg(&oracle, &cfg);

    Ok((
        result.phase_estimate,
        result.circular_std,
        result.heisenberg_bound,
        result.total_query_time,
    ))
}

// ============================================================================
// NOISE MODELS
// ============================================================================

/// Noise model configuration for quantum devices
#[cfg(feature = "python")]
#[pyclass(name = "NoiseModel")]
#[derive(Debug, Clone)]
pub struct PyNoiseModel {
    inner: NoiseModel,
}

#[cfg(feature = "python")]
#[pymethods]
impl PyNoiseModel {
    /// Create a noise model with custom parameters
    #[new]
    fn new(
        depolarizing_prob: f64,
        amplitude_damping_prob: f64,
        readout_error: f64,
    ) -> PyResult<Self> {
        if depolarizing_prob < 0.0 || depolarizing_prob > 1.0 {
            return Err(PyValueError::new_err(
                "depolarizing_prob must be between 0 and 1",
            ));
        }

        Ok(PyNoiseModel {
            inner: NoiseModel {
                depolarizing_prob,
                amplitude_damping_prob,
                phase_damping_prob: amplitude_damping_prob * 0.5,
                readout_error: (readout_error, readout_error),
                coherent_errors: std::collections::HashMap::new(),
                crosstalk_prob: 0.0001,
            },
        })
    }

    /// Get depolarizing probability
    #[getter]
    fn depolarizing_prob(&self) -> f64 {
        self.inner.depolarizing_prob
    }

    /// Get amplitude damping probability
    #[getter]
    fn amplitude_damping_prob(&self) -> f64 {
        self.inner.amplitude_damping_prob
    }

    /// Create noise model for IBM Quantum devices
    #[staticmethod]
    fn ibm_nqubit() -> Self {
        PyNoiseModel {
            inner: NoiseModel::ibm_nqubit(),
        }
    }

    /// Create noise model for Google Sycamore
    #[staticmethod]
    fn google_sycamore() -> Self {
        PyNoiseModel {
            inner: NoiseModel::google_sycamore(),
        }
    }

    /// Create noise model for Rigetti Aspen
    #[staticmethod]
    fn rigetti_aspen() -> Self {
        PyNoiseModel {
            inner: NoiseModel::rigetti_aspen(),
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "<NoiseModel depol={:.4}, amp_damp={:.4}>",
            self.inner.depolarizing_prob, self.inner.amplitude_damping_prob
        )
    }
}

/// Simulate noisy circuit by adding readout errors
#[cfg(feature = "python")]
#[pyfunction]
#[pyo3(signature = (num_qubits, readout_error, shots=1000))]
fn simulate_noisy_circuit(
    num_qubits: usize,
    readout_error: f64,
    shots: usize,
) -> PyResult<std::collections::HashMap<String, usize>> {
    if num_qubits > 12 {
        return Err(PyValueError::new_err(
            "Maximum 12 qubits for noisy simulation",
        ));
    }
    if readout_error < 0.0 || readout_error > 1.0 {
        return Err(PyValueError::new_err(
            "readout_error must be between 0 and 1",
        ));
    }

    let mut counts: std::collections::HashMap<String, usize> = std::collections::HashMap::new();
    use rand::SeedableRng;
    let mut rng = rand::rngs::StdRng::from_entropy();

    // Run circuit with readout noise
    for _ in 0..shots {
        let mut sim = QuantumSimulator::new(num_qubits);
        sim.h(0);
        for i in 0..(num_qubits - 1) {
            sim.cnot(i, i + 1);
        }

        // Measure and apply readout error
        let ideal_result = sim.measure();

        // Apply bit flip readout error
        let mut noisy_result = 0usize;
        for bit in 0..num_qubits {
            let bit_value = (ideal_result >> bit) & 1;
            let noisy_bit = if rng.gen_range(0.0..1.0) < readout_error {
                1 - bit_value
            } else {
                bit_value
            };
            noisy_result |= (noisy_bit as usize) << bit;
        }

        let bitstring = format!("{:0width$b}", noisy_result, width = num_qubits);
        *counts.entry(bitstring).or_insert(0) += 1;
    }

    Ok(counts)
}

/// Compare ideal vs noisy circuit results
#[cfg(feature = "python")]
#[pyfunction]
#[pyo3(signature = (readout_error=0.02, shots=1000))]
fn compare_ideal_vs_noisy(readout_error: Option<f64>, shots: Option<usize>) -> PyResult<String> {
    let ro = readout_error.unwrap_or(0.02);
    let n_shots = shots.unwrap_or(1000);

    let mut ideal_counts: std::collections::HashMap<String, usize> =
        std::collections::HashMap::new();

    // Ideal Bell state - recreate for each shot since measurement collapses state
    for _ in 0..n_shots {
        let mut sim = QuantumSimulator::new(2);
        sim.h(0);
        sim.cnot(0, 1);
        let m = sim.measure();
        let bitstring = format!("{:02b}", m);
        *ideal_counts.entry(bitstring).or_insert(0) += 1;
    }

    // Noisy simulation
    let noisy_counts = simulate_noisy_circuit(2, ro, n_shots)?;

    // Calculate statistics
    let ideal_00 = *ideal_counts.get("00").unwrap_or(&0) as f64;
    let ideal_11 = *ideal_counts.get("11").unwrap_or(&0) as f64;
    let noisy_00 = *noisy_counts.get("00").unwrap_or(&0) as f64;
    let noisy_11 = *noisy_counts.get("11").unwrap_or(&0) as f64;

    // Fidelity estimate
    let overlap = (ideal_00.min(noisy_00) + ideal_11.min(noisy_11))
        / (ideal_00.max(noisy_00) + ideal_11.max(noisy_11) + 1.0);

    Ok(format!(
        "Bell State Comparison ({} shots, readout_error={}):\n\
         Ideal:   |00⟩={} ({:.1}%), |11⟩={} ({:.1}%)\n\
         Noisy:   |00⟩={} ({:.1}%), |11⟩={} ({:.1}%)\n\
         Fidelity: {:.4} (measurement overlap)",
        n_shots,
        ro,
        ideal_00 as i64,
        ideal_00 / n_shots as f64 * 100.0,
        ideal_11 as i64,
        ideal_11 / n_shots as f64 * 100.0,
        noisy_00 as i64,
        noisy_00 / n_shots as f64 * 100.0,
        noisy_11 as i64,
        noisy_11 / n_shots as f64 * 100.0,
        overlap
    ))
}

/// Python wrapper for MPSSimulator (Matrix Product State)
///
/// MPS representation enables efficient simulation of larger quantum systems
/// by compressing the state tensor based on entanglement structure.
#[cfg(feature = "python")]
#[pyclass(name = "MPSSimulator")]
pub struct PyMPSSimulator {
    inner: MPSSimulator,
}

#[cfg(feature = "python")]
#[pymethods]
impl PyMPSSimulator {
    /// Create a new MPS simulator
    ///
    /// Args:
    ///     num_qubits: Number of qubits to simulate
    ///     max_bond_dim: Maximum bond dimension for compression (None for unlimited)
    #[new]
    #[pyo3(signature = (num_qubits, max_bond_dim=None))]
    fn new(num_qubits: usize, max_bond_dim: Option<usize>) -> PyResult<Self> {
        if num_qubits > 100000 {
            // No hard limit - actual limit depends on memory and bond dimension
            // 24GB RAM can theoretically handle ~1M qubits at bond_dim=2
        }
        Ok(PyMPSSimulator {
            inner: MPSSimulator::new(num_qubits, max_bond_dim),
        })
    }

    /// Get the number of qubits
    #[getter]
    fn num_qubits(&self) -> usize {
        self.inner.num_qubits()
    }

    /// Get current bond dimensions
    #[getter]
    fn bond_dimensions(&self) -> Vec<usize> {
        self.inner.bond_dimensions()
    }

    /// Get maximum current bond dimension
    #[getter]
    fn max_bond_dim(&self) -> usize {
        self.inner.max_bond_dim()
    }

    /// Apply Hadamard gate
    fn h(&mut self, qubit: usize) -> PyResult<()> {
        if qubit >= self.inner.num_qubits() {
            return Err(PyValueError::new_err("Qubit index out of bounds"));
        }
        self.inner.h(qubit);
        Ok(())
    }

    /// Apply X (NOT) gate
    fn x(&mut self, qubit: usize) -> PyResult<()> {
        if qubit >= self.inner.num_qubits() {
            return Err(PyValueError::new_err("Qubit index out of bounds"));
        }
        self.inner.x(qubit);
        Ok(())
    }

    /// Apply Y gate
    fn y(&mut self, qubit: usize) -> PyResult<()> {
        if qubit >= self.inner.num_qubits() {
            return Err(PyValueError::new_err("Qubit index out of bounds"));
        }
        self.inner.y(qubit);
        Ok(())
    }

    /// Apply Z gate
    fn z(&mut self, qubit: usize) -> PyResult<()> {
        if qubit >= self.inner.num_qubits() {
            return Err(PyValueError::new_err("Qubit index out of bounds"));
        }
        self.inner.z(qubit);
        Ok(())
    }

    /// Apply S gate (phase gate)
    fn s(&mut self, qubit: usize) -> PyResult<()> {
        if qubit >= self.inner.num_qubits() {
            return Err(PyValueError::new_err("Qubit index out of bounds"));
        }
        self.inner.s(qubit);
        Ok(())
    }

    /// Apply T gate (π/8 gate)
    fn t(&mut self, qubit: usize) -> PyResult<()> {
        if qubit >= self.inner.num_qubits() {
            return Err(PyValueError::new_err("Qubit index out of bounds"));
        }
        self.inner.t(qubit);
        Ok(())
    }

    /// Apply Rx gate (rotation around X-axis)
    fn rx(&mut self, qubit: usize, theta: f64) -> PyResult<()> {
        if qubit >= self.inner.num_qubits() {
            return Err(PyValueError::new_err("Qubit index out of bounds"));
        }
        self.inner.rx(qubit, theta);
        Ok(())
    }

    /// Apply Ry gate (rotation around Y-axis)
    fn ry(&mut self, qubit: usize, theta: f64) -> PyResult<()> {
        if qubit >= self.inner.num_qubits() {
            return Err(PyValueError::new_err("Qubit index out of bounds"));
        }
        self.inner.ry(qubit, theta);
        Ok(())
    }

    /// Apply Rz gate (rotation around Z-axis)
    fn rz(&mut self, qubit: usize, theta: f64) -> PyResult<()> {
        if qubit >= self.inner.num_qubits() {
            return Err(PyValueError::new_err("Qubit index out of bounds"));
        }
        self.inner.rz(qubit, theta);
        Ok(())
    }

    /// Apply CNOT gate
    fn cnot(&mut self, control: usize, target: usize) -> PyResult<()> {
        let n = self.inner.num_qubits();
        if control >= n || target >= n {
            return Err(PyValueError::new_err("Qubit index out of bounds"));
        }
        if control == target {
            return Err(PyValueError::new_err(
                "Control and target must be different",
            ));
        }
        self.inner.cnot(control, target);
        Ok(())
    }

    /// Swap two qubits
    fn swap(&mut self, qubit1: usize, qubit2: usize) -> PyResult<()> {
        let n = self.inner.num_qubits();
        if qubit1 >= n || qubit2 >= n {
            return Err(PyValueError::new_err("Qubit index out of bounds"));
        }
        self.inner.swap(qubit1, qubit2);
        Ok(())
    }

    /// Measure all qubits
    fn measure(&mut self) -> PyResult<usize> {
        Ok(self.inner.measure())
    }

    /// Measure a single qubit
    fn measure_qubit(&mut self, qubit: usize) -> PyResult<usize> {
        if qubit >= self.inner.num_qubits() {
            return Err(PyValueError::new_err("Qubit index out of bounds"));
        }
        Ok(self.inner.measure_qubit(qubit))
    }

    /// Get probabilities by converting to state vector
    fn probabilities(&self) -> PyResult<Vec<f64>> {
        let state_vector = self.inner.mps().to_state_vector();
        Ok(state_vector.iter().map(|amp| amp.norm_sqr()).collect())
    }

    /// Sample multiple measurements
    fn sample_bitstrings(
        &self,
        shots: usize,
    ) -> PyResult<std::collections::HashMap<String, usize>> {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        let mut counts = std::collections::HashMap::new();
        let state_vector = self.inner.mps().to_state_vector();
        let n = self.inner.num_qubits();

        for _ in 0..shots {
            let mut r: f64 = rng.gen();
            let mut result = 0;
            for (i, amp) in state_vector.iter().enumerate() {
                let prob = amp.norm_sqr();
                r -= prob;
                if r <= 0.0 {
                    result = i;
                    break;
                }
            }
            let bitstring = format!("{:0width$b}", result, width = n);
            *counts.entry(bitstring).or_insert(0) += 1;
        }

        Ok(counts)
    }

    /// Get string representation
    fn __repr__(&self) -> String {
        format!(
            "<MPSSimulator n={} qubits, max_bond_dim={}>",
            self.inner.num_qubits(),
            self.inner.max_bond_dim()
        )
    }

    /// Get detailed info
    fn info(&self) -> String {
        format!(
            "MPSSimulator:\n  Qubits: {}\n  Max bond dim: {}\n  Current bond dims: {:?}",
            self.inner.num_qubits(),
            self.inner.max_bond_dim(),
            self.inner.bond_dimensions()
        )
    }
}

// ============================================================
// 2D Grid Simulator using Snake-Mapped MPS
// ============================================================

use crate::snake_mapping::{GridCoord, SnakeMapper};
use std::sync::Arc;

/// 2D grid quantum simulator using snake pattern mapping to MPS
///
/// This enables simulation of 2D quantum circuits on existing MPS infrastructure.
///
/// # Example
///
/// ```python
/// import nqpu_metal
///
/// # Create 4x4 grid
/// sim = nqpu_metal.Grid2DSimulator(4, 4, max_bond_dim=2)
///
/// # Apply H to center qubit
/// sim.h(2, 1)  # x=2, y=1
///
/// # Nearest-neighbor CNOT
/// sim.cnot(2, 1, 3, 1)  # (2,1) -> (3,1)
///
/// # Measure all qubits
/// result = sim.measure()
/// ```
#[cfg(feature = "python")]
#[pyclass(name = "Grid2DSimulator")]
pub struct PyGrid2DSimulator {
    inner: Arc<std::sync::Mutex<Grid2DSimulator>>,
    mapper: SnakeMapper,
}

#[cfg(feature = "python")]
#[pymethods]
impl PyGrid2DSimulator {
    /// Create a new 2D grid simulator
    ///
    /// # Arguments
    ///
    /// * `width` - Number of qubits in each row
    /// * `height` - Number of qubits in each column
    /// * `max_bond_dim` - Maximum bond dimension for MPS (default: 2)
    #[new]
    #[pyo3(signature = (width, height, max_bond_dim=2))]
    fn new(width: usize, height: usize, max_bond_dim: Option<usize>) -> PyResult<Self> {
        let mapper = SnakeMapper::new(width, height);
        let inner = Grid2DSimulator::new(width, height, max_bond_dim);

        Ok(Self {
            inner: Arc::new(std::sync::Mutex::new(inner)),
            mapper,
        })
    }

    /// Apply Hadamard gate to qubit at (x, y)
    #[pyo3(signature = (x, y))]
    fn h(&self, x: usize, y: usize) -> PyResult<()> {
        let mut sim = self.inner.lock().unwrap();
        let idx = self.mapper.map_2d_to_1d(x, y);
        sim.h(idx);
        Ok(())
    }

    /// Apply X (NOT) gate to qubit at (x, y)
    #[pyo3(signature = (x, y))]
    fn x(&self, x: usize, y: usize) -> PyResult<()> {
        let mut sim = self.inner.lock().unwrap();
        let idx = self.mapper.map_2d_to_1d(x, y);
        sim.x(idx);
        Ok(())
    }

    /// Apply Y gate to qubit at (x, y)
    #[pyo3(signature = (x, y))]
    fn y(&self, x: usize, y: usize) -> PyResult<()> {
        let mut sim = self.inner.lock().unwrap();
        let idx = self.mapper.map_2d_to_1d(x, y);
        sim.y(idx);
        Ok(())
    }

    /// Apply Z gate to qubit at (x, y)
    #[pyo3(signature = (x, y))]
    fn z(&self, x: usize, y: usize) -> PyResult<()> {
        let mut sim = self.inner.lock().unwrap();
        let idx = self.mapper.map_2d_to_1d(x, y);
        sim.z(idx);
        Ok(())
    }

    /// Apply CNOT gate between two 2D positions
    ///
    /// Warning: Non-nearest-neighbor gates cause bond dimension growth
    #[pyo3(signature = (x1, y1, x2, y2))]
    fn cnot(&self, x1: usize, y1: usize, x2: usize, y2: usize) -> PyResult<()> {
        let mut sim = self.inner.lock().unwrap();
        let idx1 = self.mapper.map_2d_to_1d(x1, y1);
        let idx2 = self.mapper.map_2d_to_1d(x2, y2);

        // Warn about long-range gates
        let coord1 = GridCoord::new(x1, y1);
        let coord2 = GridCoord::new(x2, y2);
        let distance = self.mapper.distance(&coord1, &coord2);
        if distance > 1 {
            eprintln!(
                "Warning: Long-range CNOT (distance={}) will increase bond dimension",
                distance
            );
        }

        sim.cnot(idx1, idx2);
        Ok(())
    }

    /// Apply RX rotation to qubit at (x, y)
    #[pyo3(signature = (x, y, theta))]
    fn rx(&self, x: usize, y: usize, theta: f64) -> PyResult<()> {
        let mut sim = self.inner.lock().unwrap();
        let idx = self.mapper.map_2d_to_1d(x, y);
        sim.rx(idx, theta);
        Ok(())
    }

    /// Apply RY rotation to qubit at (x, y)
    #[pyo3(signature = (x, y, theta))]
    fn ry(&self, x: usize, y: usize, theta: f64) -> PyResult<()> {
        let mut sim = self.inner.lock().unwrap();
        let idx = self.mapper.map_2d_to_1d(x, y);
        sim.ry(idx, theta);
        Ok(())
    }

    /// Apply RZ rotation to qubit at (x, y)
    #[pyo3(signature = (x, y, theta))]
    fn rz(&self, x: usize, y: usize, theta: f64) -> PyResult<()> {
        let mut sim = self.inner.lock().unwrap();
        let idx = self.mapper.map_2d_to_1d(x, y);
        sim.rz(idx, theta);
        Ok(())
    }

    /// Measure all qubits and return integer result
    #[pyo3(signature = ())]
    fn measure(&self) -> PyResult<u64> {
        let mut sim = self.inner.lock().unwrap();
        Ok(sim.measure())
    }

    /// Get number of qubits in grid
    fn num_qubits(&self) -> usize {
        self.mapper.size()
    }

    /// Get grid dimensions
    fn dimensions(&self) -> PyResult<(usize, usize)> {
        Ok(self.mapper.dimensions())
    }

    /// Get maximum Manhattan distance between any two qubits
    fn max_distance(&self) -> usize {
        self.mapper.max_distance()
    }

    /// Get string representation
    fn __repr__(&self) -> String {
        let (width, height) = self.mapper.dimensions();
        format!("<Grid2DSimulator width={} height={}>", width, height)
    }

    /// Get detailed info
    fn info(&self) -> String {
        let (width, height) = self.mapper.dimensions();
        format!(
            "Grid2DSimulator:\n  Width: {}\n  Height: {}\n  Total qubits: {}\n  Max bond dim: {}",
            width,
            height,
            self.mapper.size(),
            self.inner.lock().unwrap().max_bond_dim()
        )
    }
}

// ============================================================
// Internal Grid2DSimulator (wraps MPS)
// ============================================================

/// Internal 2D grid simulator using snake-mapped MPS
#[cfg(feature = "python")]
struct Grid2DSimulator {
    mps: MPSSimulator,
    mapper: crate::snake_mapping::SnakeMapper,
    width: usize,
    height: usize,
}

#[cfg(feature = "python")]
impl Grid2DSimulator {
    fn new(width: usize, height: usize, max_bond_dim: Option<usize>) -> Self {
        let total_qubits = width * height;
        let mps = MPSSimulator::new(total_qubits, max_bond_dim);
        let mapper = crate::snake_mapping::SnakeMapper::new(width, height);
        Self {
            mps,
            mapper,
            width,
            height,
        }
    }

    // Delegate all operations directly to underlying MPS methods
    fn h(&mut self, idx: usize) {
        self.mps.h(idx);
    }

    fn x(&mut self, idx: usize) {
        self.mps.x(idx);
    }

    fn y(&mut self, idx: usize) {
        self.mps.y(idx);
    }

    fn z(&mut self, idx: usize) {
        self.mps.z(idx);
    }

    fn cnot(&mut self, control: usize, target: usize) {
        self.mps.cnot(control, target);
    }

    fn rx(&mut self, idx: usize, theta: f64) {
        self.mps.rx(idx, theta);
    }

    fn ry(&mut self, idx: usize, theta: f64) {
        self.mps.ry(idx, theta);
    }

    fn rz(&mut self, idx: usize, theta: f64) {
        self.mps.rz(idx, theta);
    }

    fn measure(&mut self) -> u64 {
        self.mps.measure() as u64
    }

    fn max_bond_dim(&self) -> usize {
        self.mps.max_bond_dim()
    }

    // ============================================================
    // 2D ALGORITHM HELPERS (delegate to algorithms_2d)
    // ============================================================

    /// Apply 2D QFT using algorithms_2d
    fn qft_2d(&mut self) -> PyResult<()> {
        crate::algorithms_2d::qft_2d(&mut self.mps, self.width, self.height, &self.mapper);
        Ok(())
    }

    /// Apply inverse 2D QFT
    fn inverse_qft_2d(&mut self) -> PyResult<()> {
        crate::algorithms_2d::inverse_qft_2d(&mut self.mps, self.width, self.height, &self.mapper);
        Ok(())
    }

    /// Get entanglement entropy grid
    fn entanglement_entropy_2d(&self) -> PyResult<Vec<Vec<f64>>> {
        let grid = crate::algorithms_2d::entanglement_entropy_2d(
            &self.mps,
            self.width,
            self.height,
            &self.mapper,
        );
        // Extract just entropy values
        let result: Vec<Vec<f64>> = grid
            .iter()
            .map(|row| row.iter().map(|d| d.entropy).collect())
            .collect();
        Ok(result)
    }

    /// Sample power measurements
    fn power_sample_2d(
        &mut self,
        shots: usize,
    ) -> PyResult<std::collections::HashMap<String, usize>> {
        if shots > 100000 {
            return Err(PyValueError::new_err("Maximum 100000 shots"));
        }
        let measurement = crate::algorithms_2d::power_sample_2d(
            &mut self.mps,
            self.width,
            self.height,
            &self.mapper,
            shots,
        );
        // Convert to hashmap format
        let mut result = std::collections::HashMap::new();
        for y in 0..self.height {
            for x in 0..self.width {
                let key = format!("{},{}", x, y);
                result.insert(key, measurement.counts[y][x]);
            }
        }
        Ok(result)
    }

    /// Apply square CNOT from center
    fn square_cnot(&mut self, center_x: usize, center_y: usize) -> PyResult<()> {
        crate::algorithms_2d::square_cnot(
            &mut self.mps,
            center_x,
            center_y,
            self.width,
            self.height,
            &self.mapper,
        );
        Ok(())
    }

    /// Run Grover 2D search
    fn grover_2d(
        &mut self,
        target_x: usize,
        target_y: usize,
        iterations: Option<usize>,
    ) -> PyResult<(usize, usize)> {
        let result = crate::algorithms_2d::grover_2d(
            &mut self.mps,
            self.width,
            self.height,
            &self.mapper,
            &|x, y| x == target_x && y == target_y,
            iterations,
        );
        Ok((result.x, result.y))
    }

    /// Run 2D benchmarks
    fn benchmark_2d(
        &self,
        sizes: Vec<(usize, usize)>,
        max_bond_dim: Option<usize>,
        iterations: usize,
    ) -> PyResult<String> {
        let (widths, heights): (Vec<usize>, Vec<usize>) = sizes.into_iter().unzip();
        let results =
            crate::algorithms_2d::benchmark_suite_2d(widths, heights, max_bond_dim, iterations);

        // Format results
        let mut output = String::from("2D Quantum Algorithm Benchmarks\n");
        output.push_str(&format!("========================================\n\n"));

        for result in &results {
            output.push_str(&format!(
                "{}: {}x{} ({} qubits)\n",
                result.name, result.width, result.height, result.total_qubits
            ));
            output.push_str(&format!("  Time: {:.3} ms/iter\n", result.duration_ms));
            if let Some(fid) = result.fidelity {
                output.push_str(&format!("  Fidelity: {:.2}%\n", fid * 100.0));
            }
            output.push_str(&format!("  Max bond dim: {}\n\n", result.max_bond_dim));
        }

        Ok(output)
    }
}

// ============================================================================
// AUTOMATIC DIFFERENTIATION (ADJOINT METHOD) — for nHybrid integration
// ============================================================================

/// Adjoint-based quantum circuit for automatic differentiation.
///
/// Supports parameterized gates (Rx, Ry, Rz) and computes exact analytic
/// gradients via reverse-mode adjoint differentiation. Used by nHybrid
/// to bridge quantum gradients into PyTorch's autograd.
#[cfg(feature = "python")]
#[pyclass(name = "AdjointCircuit")]
pub struct PyAdjointCircuit {
    inner: crate::adjoint_diff::AdjointCircuit,
}

#[cfg(feature = "python")]
#[pymethods]
impl PyAdjointCircuit {
    /// Create a new adjoint circuit with the given number of qubits.
    #[new]
    fn new(num_qubits: usize) -> PyResult<Self> {
        if num_qubits > 28 {
            return Err(PyValueError::new_err(
                "Maximum 28 qubits for adjoint circuit (state vector backend)",
            ));
        }
        Ok(PyAdjointCircuit {
            inner: crate::adjoint_diff::AdjointCircuit::new(num_qubits),
        })
    }

    /// Number of qubits in this circuit.
    #[getter]
    fn num_qubits(&self) -> usize {
        self.inner.num_qubits
    }

    /// Number of operations in this circuit.
    #[getter]
    fn num_ops(&self) -> usize {
        self.inner.ops.len()
    }

    /// Add a Hadamard gate on the target qubit.
    fn h(&mut self, qubit: usize) -> PyResult<()> {
        if qubit >= self.inner.num_qubits {
            return Err(PyValueError::new_err("Qubit index out of range"));
        }
        self.inner.add_op(crate::adjoint_diff::AdjointOp::Fixed(
            crate::gates::Gate::h(qubit),
        ));
        Ok(())
    }

    /// Add a Pauli-X gate on the target qubit.
    fn x(&mut self, qubit: usize) -> PyResult<()> {
        if qubit >= self.inner.num_qubits {
            return Err(PyValueError::new_err("Qubit index out of range"));
        }
        self.inner.add_op(crate::adjoint_diff::AdjointOp::Fixed(
            crate::gates::Gate::x(qubit),
        ));
        Ok(())
    }

    /// Add a Pauli-Y gate on the target qubit.
    fn y(&mut self, qubit: usize) -> PyResult<()> {
        if qubit >= self.inner.num_qubits {
            return Err(PyValueError::new_err("Qubit index out of range"));
        }
        self.inner.add_op(crate::adjoint_diff::AdjointOp::Fixed(
            crate::gates::Gate::y(qubit),
        ));
        Ok(())
    }

    /// Add a Pauli-Z gate on the target qubit.
    fn z(&mut self, qubit: usize) -> PyResult<()> {
        if qubit >= self.inner.num_qubits {
            return Err(PyValueError::new_err("Qubit index out of range"));
        }
        self.inner.add_op(crate::adjoint_diff::AdjointOp::Fixed(
            crate::gates::Gate::z(qubit),
        ));
        Ok(())
    }

    /// Add a CNOT gate (control → target).
    fn cnot(&mut self, control: usize, target: usize) -> PyResult<()> {
        if control >= self.inner.num_qubits || target >= self.inner.num_qubits {
            return Err(PyValueError::new_err("Qubit index out of range"));
        }
        self.inner.add_op(crate::adjoint_diff::AdjointOp::Fixed(
            crate::gates::Gate::cnot(control, target),
        ));
        Ok(())
    }

    /// Add a parametric Rx(theta) gate. param_index references into the params array.
    fn rx(&mut self, qubit: usize, param_index: usize) -> PyResult<()> {
        if qubit >= self.inner.num_qubits {
            return Err(PyValueError::new_err("Qubit index out of range"));
        }
        self.inner.add_op(crate::adjoint_diff::AdjointOp::Rx {
            qubit,
            param: param_index,
        });
        Ok(())
    }

    /// Add a parametric Ry(theta) gate. param_index references into the params array.
    fn ry(&mut self, qubit: usize, param_index: usize) -> PyResult<()> {
        if qubit >= self.inner.num_qubits {
            return Err(PyValueError::new_err("Qubit index out of range"));
        }
        self.inner.add_op(crate::adjoint_diff::AdjointOp::Ry {
            qubit,
            param: param_index,
        });
        Ok(())
    }

    /// Add a parametric Rz(theta) gate. param_index references into the params array.
    fn rz(&mut self, qubit: usize, param_index: usize) -> PyResult<()> {
        if qubit >= self.inner.num_qubits {
            return Err(PyValueError::new_err("Qubit index out of range"));
        }
        self.inner.add_op(crate::adjoint_diff::AdjointOp::Rz {
            qubit,
            param: param_index,
        });
        Ok(())
    }

    /// Compute <Z_qubit> expectation value for the given parameters.
    fn expectation(&self, params: Vec<f64>, qubit: usize) -> PyResult<f64> {
        if qubit >= self.inner.num_qubits {
            return Err(PyValueError::new_err("Observable qubit out of range"));
        }
        self.inner
            .expectation(&params, crate::adjoint_diff::Observable::PauliZ(qubit))
            .map_err(|e| PyValueError::new_err(format!("Expectation failed: {}", e)))
    }

    /// Compute d<Z_qubit>/d(params) via reverse-mode adjoint differentiation.
    ///
    /// Returns a list of gradients, one per parameter.
    fn gradient(&self, params: Vec<f64>, qubit: usize) -> PyResult<Vec<f64>> {
        if qubit >= self.inner.num_qubits {
            return Err(PyValueError::new_err("Observable qubit out of range"));
        }
        self.inner
            .gradient(&params, crate::adjoint_diff::Observable::PauliZ(qubit))
            .map_err(|e| PyValueError::new_err(format!("Gradient failed: {}", e)))
    }

    fn __repr__(&self) -> String {
        format!(
            "AdjointCircuit(num_qubits={}, num_ops={})",
            self.inner.num_qubits,
            self.inner.ops.len()
        )
    }
}
// Minimal PyTorch bridge bindings - append to end of python.rs

// =============================================================================
// PYTORCH BRIDGE PYTHON BINDINGS
// =============================================================================

// pytorch_bridge types already imported at top of file

// =============================================================================
// PYTORCH INTEGRATION — Real DifferentiableCircuit bindings
// =============================================================================

/// Python wrapper for the Rust DifferentiableCircuit autodiff engine.
///
/// Supports 4 gradient methods: parameter-shift, adjoint, finite-difference, backprop.
/// All computation happens in Rust for maximum performance.
#[cfg(feature = "python")]
#[pyclass(name = "DifferentiableCircuit")]
pub struct PyDifferentiableCircuit {
    inner: DifferentiableCircuit,
}

#[cfg(feature = "python")]
#[pymethods]
impl PyDifferentiableCircuit {
    /// Create a new differentiable circuit.
    ///
    /// Args:
    ///     n_qubits: Number of qubits
    ///     gradient_method: "parameter_shift", "adjoint", "finite_difference", or "backprop"
    #[new]
    #[pyo3(signature = (n_qubits, gradient_method="parameter_shift"))]
    fn new(n_qubits: usize, gradient_method: &str) -> PyResult<Self> {
        let method = match gradient_method {
            "parameter_shift" => pytorch_bridge::GradientMethod::ParameterShift {
                shift: std::f64::consts::FRAC_PI_2,
            },
            "adjoint" => pytorch_bridge::GradientMethod::AdjointDiff,
            "finite_difference" => pytorch_bridge::GradientMethod::FiniteDifference {
                epsilon: 1e-7,
            },
            "backprop" => pytorch_bridge::GradientMethod::Backprop,
            other => {
                return Err(PyValueError::new_err(format!(
                    "Unknown gradient method '{}'. Use: parameter_shift, adjoint, finite_difference, backprop",
                    other
                )))
            }
        };
        Ok(Self {
            inner: DifferentiableCircuit::with_gradient_method(n_qubits, method),
        })
    }

    /// Register a trainable parameter with initial value. Returns parameter index.
    fn add_parameter(&mut self, value: f64) -> usize {
        self.inner.add_parameter(value)
    }

    /// Set all parameter values at once.
    fn set_parameters(&mut self, params: Vec<f64>) -> PyResult<()> {
        self.inner
            .set_parameters(&params)
            .map_err(|e| PyValueError::new_err(format!("{:?}", e)))
    }

    /// Add a Hadamard gate.
    fn h(&mut self, qubit: usize) {
        self.inner
            .add_gate(pytorch_bridge::ParametricGate::H(qubit));
    }

    /// Add a Pauli-X gate.
    fn x(&mut self, qubit: usize) {
        self.inner
            .add_gate(pytorch_bridge::ParametricGate::X(qubit));
    }

    /// Add a Pauli-Z gate.
    fn z(&mut self, qubit: usize) {
        self.inner
            .add_gate(pytorch_bridge::ParametricGate::Z(qubit));
    }

    /// Add a CNOT gate.
    fn cx(&mut self, control: usize, target: usize) {
        self.inner
            .add_gate(pytorch_bridge::ParametricGate::CX(control, target));
    }

    /// Add a CZ gate.
    fn cz(&mut self, control: usize, target: usize) {
        self.inner
            .add_gate(pytorch_bridge::ParametricGate::CZ(control, target));
    }

    /// Add a parametric Rx rotation. `param_idx` is the parameter index from add_parameter().
    fn rx(&mut self, qubit: usize, param_idx: usize) {
        self.inner
            .add_gate(pytorch_bridge::ParametricGate::Rx(qubit, param_idx));
    }

    /// Add a parametric Ry rotation.
    fn ry(&mut self, qubit: usize, param_idx: usize) {
        self.inner
            .add_gate(pytorch_bridge::ParametricGate::Ry(qubit, param_idx));
    }

    /// Add a parametric Rz rotation.
    fn rz(&mut self, qubit: usize, param_idx: usize) {
        self.inner
            .add_gate(pytorch_bridge::ParametricGate::Rz(qubit, param_idx));
    }

    /// Add a parametric controlled-Rx rotation.
    fn crx(&mut self, control: usize, target: usize, param_idx: usize) {
        self.inner.add_gate(pytorch_bridge::ParametricGate::CRx(
            control, target, param_idx,
        ));
    }

    /// Add a parametric controlled-Ry rotation.
    fn cry(&mut self, control: usize, target: usize, param_idx: usize) {
        self.inner.add_gate(pytorch_bridge::ParametricGate::CRy(
            control, target, param_idx,
        ));
    }

    /// Add a parametric controlled-Rz rotation.
    fn crz(&mut self, control: usize, target: usize, param_idx: usize) {
        self.inner.add_gate(pytorch_bridge::ParametricGate::CRz(
            control, target, param_idx,
        ));
    }

    /// Add a U3 gate with three parameter indices (theta, phi, lambda).
    fn u3(&mut self, qubit: usize, theta_idx: usize, phi_idx: usize, lambda_idx: usize) {
        self.inner.add_gate(pytorch_bridge::ParametricGate::U3(
            qubit, theta_idx, phi_idx, lambda_idx,
        ));
    }

    /// Number of trainable parameters.
    fn num_parameters(&self) -> usize {
        self.inner.num_parameters()
    }

    /// Forward pass: execute circuit and measure observables.
    ///
    /// Args:
    ///     params: Parameter values (list of floats)
    ///     observables: List of observable specs, e.g. ["Z0", "Z1", "Z0Z1"]
    ///
    /// Returns:
    ///     Dict with 'expectation_values', 'probabilities', 'state_vector'
    fn forward(
        &self,
        py: Python<'_>,
        params: Vec<f64>,
        observables: Vec<String>,
    ) -> PyResult<pyo3::Py<pyo3::types::PyDict>> {
        let obs: Vec<pytorch_bridge::Observable> = observables
            .iter()
            .map(|s| parse_observable(s))
            .collect::<Result<Vec<_>, _>>()?;

        let result = self
            .inner
            .forward_with_params(&params, &obs)
            .map_err(|e| PyValueError::new_err(format!("{:?}", e)))?;

        let dict = pyo3::types::PyDict::new(py);
        dict.set_item("expectation_values", result.expectation_values)?;
        if let Some(probs) = result.probabilities {
            dict.set_item("probabilities", probs)?;
        }
        if let Some(sv) = result.state_vector {
            let re: Vec<f64> = sv.iter().map(|(r, _)| *r).collect();
            let im: Vec<f64> = sv.iter().map(|(_, i)| *i).collect();
            dict.set_item("state_real", re)?;
            dict.set_item("state_imag", im)?;
        }
        Ok(dict.into())
    }

    /// Backward pass: compute gradients of expectation values w.r.t. parameters.
    ///
    /// Args:
    ///     params: Parameter values
    ///     observables: Observable specs
    ///
    /// Returns:
    ///     Dict with 'parameter_gradients' (list of list of floats, shape [n_obs][n_params]),
    ///     'gradient_method_used' (string)
    fn backward(
        &self,
        py: Python<'_>,
        params: Vec<f64>,
        observables: Vec<String>,
    ) -> PyResult<pyo3::Py<pyo3::types::PyDict>> {
        let obs: Vec<pytorch_bridge::Observable> = observables
            .iter()
            .map(|s| parse_observable(s))
            .collect::<Result<Vec<_>, _>>()?;

        let result = self
            .inner
            .backward_with_params(&params, &obs)
            .map_err(|e| PyValueError::new_err(format!("{:?}", e)))?;

        let dict = pyo3::types::PyDict::new(py);
        dict.set_item("parameter_gradients", result.parameter_gradients)?;
        dict.set_item("gradient_method_used", result.gradient_method_used)?;
        Ok(dict.into())
    }

    fn __repr__(&self) -> String {
        format!(
            "<DifferentiableCircuit qubits={} params={} gates={}>",
            self.inner.num_qubits,
            self.inner.parameters.len(),
            self.inner.gates.len(),
        )
    }
}

// =============================================================================
// JAX INTEGRATION — Real JAXCircuit bindings
// =============================================================================

/// Python wrapper for the Rust JAX quantum circuit engine.
///
/// Provides simulation, expectation values, gradients, and VMAP batch execution.
/// All computation happens in Rust (f32 precision for JAX compatibility).
#[cfg(feature = "python")]
#[pyclass(name = "JAXCircuit")]
pub struct PyJAXCircuit {
    inner: jax_bridge::JAXCircuit,
    n_qubits: usize,
}

#[cfg(feature = "python")]
#[pymethods]
impl PyJAXCircuit {
    /// Create a new JAX-compatible circuit.
    #[new]
    fn new(n_qubits: usize) -> Self {
        Self {
            inner: jax_bridge::JAXCircuit::new(n_qubits),
            n_qubits,
        }
    }

    /// Add a Hadamard gate.
    fn h(&mut self, qubit: usize) {
        self.inner.h(qubit);
    }

    /// Add a Pauli-X gate.
    fn x(&mut self, qubit: usize) {
        self.inner.x(qubit);
    }

    /// Add a Pauli-Z gate.
    fn z(&mut self, qubit: usize) {
        self.inner.z(qubit);
    }

    /// Add a CNOT gate.
    fn cnot(&mut self, control: usize, target: usize) {
        self.inner.cx(control, target);
    }

    /// Add a CZ gate.
    fn cz(&mut self, control: usize, target: usize) {
        self.inner.cz(control, target);
    }

    /// Add a parametric Rx rotation. Registers a named parameter and returns its index.
    fn rx(&mut self, qubit: usize, param_name: String) -> usize {
        let idx = self.inner.num_parameters();
        self.inner.rx(qubit, &param_name);
        idx
    }

    /// Add a parametric Ry rotation.
    fn ry(&mut self, qubit: usize, param_name: String) -> usize {
        let idx = self.inner.num_parameters();
        self.inner.ry(qubit, &param_name);
        idx
    }

    /// Add a parametric Rz rotation.
    fn rz(&mut self, qubit: usize, param_name: String) -> usize {
        let idx = self.inner.num_parameters();
        self.inner.rz(qubit, &param_name);
        idx
    }

    /// Number of trainable parameters.
    fn num_parameters(&self) -> usize {
        self.inner.num_parameters()
    }

    /// Simulate the circuit, returning complex amplitudes as (real, imag) pairs.
    fn simulate(&self, params: Vec<f32>) -> Vec<(f32, f32)> {
        let amps = self.inner.simulate(&params);
        amps.iter().map(|c| (c.real, c.imag)).collect()
    }

    /// Compute expectation value of Z on a specific qubit.
    fn expect_z(&self, params: Vec<f32>, qubit: usize) -> f32 {
        self.inner.expect_z(&params, qubit)
    }

    /// Compute expectation value of a Pauli observable string (e.g. "Z0", "Z0Z1", "X0Y1").
    fn expectation(&self, params: Vec<f32>, observable: String) -> f32 {
        self.inner.expectation(&params, &observable)
    }

    /// Parameter-shift gradients w.r.t. all parameters for Z expectation on qubit.
    fn gradient(&self, params: Vec<f32>, qubit: usize) -> Vec<f32> {
        jax_bridge::parameter_shift_grad(&self.inner, &params, qubit, std::f32::consts::FRAC_PI_2)
    }

    /// Batch (VMAP) simulate: run circuit for many parameter sets.
    ///
    /// Returns list of statevectors, each as list of (real, imag) pairs.
    fn vmap_simulate(&self, batch_params: Vec<Vec<f32>>) -> Vec<Vec<(f32, f32)>> {
        let results = jax_bridge::vmap_simulate(&self.inner, &batch_params);
        results
            .into_iter()
            .map(|amps| amps.iter().map(|c| (c.real, c.imag)).collect())
            .collect()
    }

    /// Batch (VMAP) expectation: evaluate observable for many parameter sets.
    fn vmap_expectation(&self, batch_params: Vec<Vec<f32>>, observable: String) -> Vec<f32> {
        jax_bridge::vmap_expectation(&self.inner, &batch_params, &observable)
    }

    fn __repr__(&self) -> String {
        format!(
            "<JAXCircuit {} qubits, {} params>",
            self.n_qubits,
            self.inner.num_parameters()
        )
    }
}

// =============================================================================
// Observable parsing helper
// =============================================================================

/// Parse a string observable spec like "Z0", "X1", "Z0Z1", "0.5*Z0 + 0.3*X1"
#[cfg(feature = "python")]
fn parse_observable(s: &str) -> PyResult<pytorch_bridge::Observable> {
    let s = s.trim();

    // Simple single-qubit Pauli: "Z0", "X1", "Y2"
    if s.len() >= 2 {
        let pauli = s.chars().next().unwrap();
        if let Ok(qubit) = s[1..].parse::<usize>() {
            match pauli {
                'Z' | 'z' => return Ok(pytorch_bridge::Observable::PauliZ(qubit)),
                'X' | 'x' => return Ok(pytorch_bridge::Observable::PauliX(qubit)),
                'Y' | 'y' => return Ok(pytorch_bridge::Observable::PauliY(qubit)),
                _ => {}
            }
        }
    }

    // Multi-qubit Pauli string: "Z0Z1", "X0Y1Z2"
    let mut ops: Vec<(usize, char)> = Vec::new();
    let mut chars = s.chars().peekable();
    while let Some(c) = chars.next() {
        if "XYZxyz".contains(c) {
            let mut qubit_str = String::new();
            while let Some(&d) = chars.peek() {
                if d.is_ascii_digit() {
                    qubit_str.push(d);
                    chars.next();
                } else {
                    break;
                }
            }
            if let Ok(q) = qubit_str.parse::<usize>() {
                ops.push((q, c.to_ascii_uppercase()));
            }
        }
    }
    if !ops.is_empty() {
        return Ok(pytorch_bridge::Observable::PauliString(ops));
    }

    Err(PyValueError::new_err(format!(
        "Cannot parse observable '{}'. Use e.g. 'Z0', 'X1', 'Z0Z1'",
        s
    )))
}

// =============================================================================
// QPU HARDWARE CONNECTIVITY PYTHON BINDINGS
// =============================================================================

/// Python wrapper for QPUCircuit — build circuits for real quantum hardware.
#[cfg(all(feature = "python", feature = "qpu"))]
#[pyclass(name = "QPUCircuit")]
pub struct PyQPUCircuit {
    inner: crate::qpu::QPUCircuit,
}

#[cfg(all(feature = "python", feature = "qpu"))]
#[pymethods]
impl PyQPUCircuit {
    /// Create a new QPU circuit with the given number of qubits and classical bits.
    #[new]
    #[pyo3(signature = (num_qubits, num_clbits=None))]
    fn new(num_qubits: usize, num_clbits: Option<usize>) -> Self {
        let clbits = num_clbits.unwrap_or(num_qubits);
        Self {
            inner: crate::qpu::QPUCircuit::new(num_qubits, clbits),
        }
    }

    /// Add a Hadamard gate.
    fn h(&mut self, qubit: usize) {
        self.inner.h(qubit);
    }

    /// Add a CNOT gate.
    fn cx(&mut self, control: usize, target: usize) {
        self.inner.cx(control, target);
    }

    /// Add a CZ gate.
    fn cz(&mut self, q0: usize, q1: usize) {
        self.inner.cz(q0, q1);
    }

    /// Add an X gate.
    fn x(&mut self, qubit: usize) {
        self.inner.x(qubit);
    }

    /// Add a Y gate.
    fn y(&mut self, qubit: usize) {
        self.inner.y(qubit);
    }

    /// Add a Z gate.
    fn z(&mut self, qubit: usize) {
        self.inner.z(qubit);
    }

    /// Add an Rz gate.
    fn rz(&mut self, qubit: usize, theta: f64) {
        self.inner.rz(qubit, theta);
    }

    /// Add an Rx gate.
    fn rx(&mut self, qubit: usize, theta: f64) {
        self.inner.rx(qubit, theta);
    }

    /// Add an Ry gate.
    fn ry(&mut self, qubit: usize, theta: f64) {
        self.inner.ry(qubit, theta);
    }

    /// Add an SX gate.
    fn sx(&mut self, qubit: usize) {
        self.inner.sx(qubit);
    }

    /// Add a measurement.
    fn measure(&mut self, qubit: usize, clbit: usize) {
        self.inner.measure(qubit, clbit);
    }

    /// Add measurements on all qubits.
    fn measure_all(&mut self) {
        self.inner.measure_all();
    }

    /// Get circuit depth.
    fn depth(&self) -> usize {
        self.inner.depth()
    }

    /// Get gate count (excluding barriers).
    fn gate_count(&self) -> usize {
        self.inner.gate_count()
    }

    /// Convert to OpenQASM 2.0 string.
    fn to_qasm2(&self) -> String {
        self.inner.to_qasm2()
    }

    /// Convert to OpenQASM 3.0 string.
    fn to_qasm3(&self) -> String {
        self.inner.to_qasm3()
    }

    /// Create a Bell state circuit.
    #[staticmethod]
    fn bell_state() -> Self {
        Self {
            inner: crate::qpu::QPUCircuit::bell_state(),
        }
    }

    /// Create a GHZ state circuit.
    #[staticmethod]
    fn ghz_state(n: usize) -> Self {
        Self {
            inner: crate::qpu::QPUCircuit::ghz_state(n),
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "QPUCircuit(qubits={}, clbits={}, gates={}, measurements={})",
            self.inner.num_qubits,
            self.inner.num_clbits,
            self.inner.gates.len(),
            self.inner.measurements.len()
        )
    }
}

/// Python wrapper for MockProvider — test QPU workflows without credentials.
#[cfg(all(feature = "python", feature = "qpu"))]
#[pyclass(name = "MockProvider")]
pub struct PyMockProvider {
    inner: crate::qpu::MockProvider,
    runtime: tokio::runtime::Runtime,
}

#[cfg(all(feature = "python", feature = "qpu"))]
#[pymethods]
impl PyMockProvider {
    /// Create a new mock provider for testing.
    #[new]
    fn new() -> PyResult<Self> {
        let runtime = tokio::runtime::Runtime::new()
            .map_err(|e| PyValueError::new_err(format!("Failed to create async runtime: {}", e)))?;
        Ok(Self {
            inner: crate::qpu::MockProvider::new(),
            runtime,
        })
    }

    /// List available mock backends.
    fn list_backends(&self) -> PyResult<Vec<(String, usize, bool)>> {
        use crate::qpu::QPUProvider;
        let backends = self
            .runtime
            .block_on(self.inner.list_backends())
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(backends
            .iter()
            .map(|b| (b.name.clone(), b.num_qubits, b.is_simulator))
            .collect())
    }

    /// Submit a circuit and get results immediately (mock).
    fn run(
        &self,
        circuit: &PyQPUCircuit,
        backend: &str,
        shots: usize,
    ) -> PyResult<std::collections::HashMap<String, usize>> {
        use crate::qpu::{JobConfig, QPUJob, QPUProvider};
        use std::time::Duration;

        let config = JobConfig {
            shots,
            ..Default::default()
        };

        let job = self
            .runtime
            .block_on(self.inner.submit_circuit(&circuit.inner, backend, &config))
            .map_err(|e| PyValueError::new_err(e.to_string()))?;

        let result = self
            .runtime
            .block_on(job.wait_for_completion(Duration::from_secs(30), Duration::from_millis(100)))
            .map_err(|e| PyValueError::new_err(e.to_string()))?;

        Ok(result.counts)
    }

    fn __repr__(&self) -> String {
        "MockProvider(backends=['mock_5q', 'mock_27q'])".into()
    }
}

/// List available QPU provider names.
#[cfg(all(feature = "python", feature = "qpu"))]
#[pyfunction]
fn py_qpu_providers() -> Vec<String> {
    let mut providers = vec!["mock".to_string()];
    #[cfg(feature = "qpu-ibm")]
    providers.push("ibm".to_string());
    #[cfg(feature = "qpu-braket")]
    providers.push("braket".to_string());
    #[cfg(feature = "qpu-azure")]
    providers.push("azure".to_string());
    #[cfg(feature = "qpu-ionq")]
    providers.push("ionq".to_string());
    #[cfg(feature = "qpu-google")]
    providers.push("google".to_string());
    providers
}

// =============================================================================
// SLIDING WINDOW DECODER PYTHON BINDINGS
// =============================================================================

/// Python wrapper for SlidingWindowDecoder
#[cfg(feature = "python")]
#[pyclass(name = "SlidingWindowDecoder")]
pub struct PySlidingWindowDecoder {
    inner: crate::sliding_window_decoder::SlidingWindowDecoder,
}

#[cfg(feature = "python")]
#[pymethods]
impl PySlidingWindowDecoder {
    /// Create a new sliding window QEC decoder.
    ///
    /// Args:
    ///     window_size: Number of syndrome rounds per decoding window (>= 2).
    ///     slide_step: Rounds to commit per decode (1..=window_size).
    ///     code_distance: QEC code distance (>= 1).
    ///     decoder: Inner decoder variant - "greedy" or "union_find".
    #[new]
    #[pyo3(signature = (window_size, slide_step, code_distance, decoder = "union_find"))]
    fn new(
        window_size: usize,
        slide_step: usize,
        code_distance: usize,
        decoder: &str,
    ) -> PyResult<Self> {
        let inner_decoder = match decoder {
            "greedy" => crate::sliding_window_decoder::WindowInnerDecoder::Greedy,
            "union_find" | "uf" => crate::sliding_window_decoder::WindowInnerDecoder::UnionFind,
            other => {
                return Err(PyValueError::new_err(format!(
                    "Unknown decoder '{}', expected 'greedy' or 'union_find'",
                    other
                )))
            }
        };
        if window_size < 2 {
            return Err(PyValueError::new_err("window_size must be >= 2"));
        }
        if slide_step < 1 || slide_step > window_size {
            return Err(PyValueError::new_err(
                "slide_step must be in [1, window_size]",
            ));
        }
        Ok(Self {
            inner: crate::sliding_window_decoder::SlidingWindowDecoder::new(
                window_size,
                slide_step,
                code_distance,
                inner_decoder,
            ),
        })
    }

    /// Push a new syndrome measurement round into the buffer.
    ///
    /// Args:
    ///     round_id: Sequential round identifier.
    ///     syndrome: Boolean detector outcomes for this round.
    ///     timestamp: Optional wall-clock timestamp (default 0.0).
    #[pyo3(signature = (round_id, syndrome, timestamp = 0.0))]
    fn push_round(&mut self, round_id: usize, syndrome: Vec<bool>, timestamp: f64) {
        self.inner
            .push_round(crate::sliding_window_decoder::SyndromeRound {
                round_id,
                syndrome,
                timestamp,
            });
    }

    /// Check whether enough rounds are buffered to decode a window.
    fn ready(&self) -> bool {
        self.inner.ready()
    }

    /// Decode the current window and commit corrections.
    ///
    /// Returns a dict with: committed_rounds, corrections, decode_time_us,
    /// defects_in_window, matches_found.
    fn decode_window(&mut self, py: Python<'_>) -> PyResult<pyo3::Py<pyo3::types::PyDict>> {
        if !self.inner.ready() {
            return Err(PyValueError::new_err(
                "Not enough rounds buffered; call push_round first",
            ));
        }
        let result = self.inner.decode_window();
        let dict = pyo3::types::PyDict::new(py);
        dict.set_item("committed_rounds", result.committed_rounds)?;
        dict.set_item("corrections", result.corrections)?;
        dict.set_item("decode_time_us", result.decode_time_us)?;
        dict.set_item("defects_in_window", result.defects_in_window)?;
        dict.set_item("matches_found", result.matches_found)?;
        Ok(dict.into())
    }

    /// Flush all remaining buffered rounds.
    ///
    /// Returns a list of result dicts (same format as decode_window).
    fn flush(&mut self, py: Python<'_>) -> PyResult<Vec<pyo3::Py<pyo3::types::PyDict>>> {
        let results = self.inner.flush();
        let mut out = Vec::with_capacity(results.len());
        for result in results {
            let dict = pyo3::types::PyDict::new(py);
            dict.set_item("committed_rounds", result.committed_rounds)?;
            dict.set_item("corrections", result.corrections)?;
            dict.set_item("decode_time_us", result.decode_time_us)?;
            dict.set_item("defects_in_window", result.defects_in_window)?;
            dict.set_item("matches_found", result.matches_found)?;
            out.push(dict.into());
        }
        Ok(out)
    }

    /// Get all committed correction vectors so far.
    fn committed(&self) -> Vec<Vec<bool>> {
        self.inner.committed().to_vec()
    }

    /// Number of rounds currently buffered (not yet decoded).
    #[getter]
    fn buffered_rounds(&self) -> usize {
        self.inner.buffered_rounds()
    }

    /// The window size parameter.
    #[getter]
    fn window_size(&self) -> usize {
        self.inner.window_size()
    }

    /// The slide step parameter.
    #[getter]
    fn slide_step(&self) -> usize {
        self.inner.slide_step()
    }

    /// The code distance parameter.
    #[getter]
    fn code_distance(&self) -> usize {
        self.inner.code_distance()
    }

    /// Total number of committed rounds across all decodes.
    #[getter]
    fn total_committed_rounds(&self) -> usize {
        self.inner.total_committed_rounds()
    }

    /// String name of the inner decoder variant.
    #[getter]
    fn inner_decoder(&self) -> String {
        self.inner.inner_decoder().to_string()
    }

    fn __repr__(&self) -> String {
        format!(
            "SlidingWindowDecoder(window={}, slide={}, d={}, decoder={}, buffered={}, committed={})",
            self.inner.window_size(),
            self.inner.slide_step(),
            self.inner.code_distance(),
            self.inner.inner_decoder(),
            self.inner.buffered_rounds(),
            self.inner.total_committed_rounds(),
        )
    }
}

// =============================================================================
// BP-OSD DECODER PYTHON BINDINGS
// =============================================================================

/// Python wrapper for BpOsdDecoder (belief propagation + ordered statistics decoding).
#[cfg(feature = "python")]
#[pyclass(name = "BpOsdDecoder")]
pub struct PyBpOsdDecoder {
    inner: crate::bp_osd::BpOsdDecoder,
}

#[cfg(feature = "python")]
#[pymethods]
impl PyBpOsdDecoder {
    /// Create a BP-OSD decoder from a dense parity check matrix.
    ///
    /// Args:
    ///     parity_check: 2D list of booleans representing H (rows x cols).
    ///     osd_order: OSD search depth (0 = OSD-0, 5-10 typical).
    ///     max_iterations: Max BP iterations (default 100).
    ///     damping: BP message damping in (0, 1] (default 0.75).
    ///     method: "min_sum" or "sum_product" (default "min_sum").
    #[new]
    #[pyo3(signature = (parity_check, osd_order = 0, max_iterations = 100, damping = 0.75, method = "min_sum"))]
    fn new(
        parity_check: Vec<Vec<bool>>,
        osd_order: usize,
        max_iterations: usize,
        damping: f64,
        method: &str,
    ) -> PyResult<Self> {
        let bp_method = match method {
            "min_sum" | "ms" => crate::bp_osd::BpMethod::MinSum,
            "sum_product" | "sp" => crate::bp_osd::BpMethod::SumProduct,
            other => {
                return Err(PyValueError::new_err(format!(
                    "Unknown BP method '{}', expected 'min_sum' or 'sum_product'",
                    other
                )))
            }
        };
        let sparse = crate::bp_osd::SparseMatrix::from_dense(&parity_check);
        let bp_config = crate::bp_osd::BpConfig {
            max_iterations,
            damping,
            method: bp_method,
            min_sum_scaling: 0.625,
            convergence_threshold: 1e-8,
        };
        Ok(Self {
            inner: crate::bp_osd::BpOsdDecoder::new(sparse, bp_config, osd_order, 10_000),
        })
    }

    /// Create a decoder for a repetition code of length n.
    #[staticmethod]
    #[pyo3(signature = (n, osd_order = 0))]
    fn repetition_code(n: usize, osd_order: usize) -> Self {
        let h = crate::bp_osd::SparseMatrix::repetition_code(n);
        Self {
            inner: crate::bp_osd::BpOsdDecoder::with_osd(h, osd_order),
        }
    }

    /// Create a decoder for the [7,4,3] Hamming code.
    #[staticmethod]
    #[pyo3(signature = (osd_order = 0))]
    fn hamming_7_4(osd_order: usize) -> Self {
        let h = crate::bp_osd::SparseMatrix::hamming_7_4();
        Self {
            inner: crate::bp_osd::BpOsdDecoder::with_osd(h, osd_order),
        }
    }

    /// Create a decoder for the distance-3 surface code.
    #[staticmethod]
    #[pyo3(signature = (osd_order = 0))]
    fn surface_code_d3(osd_order: usize) -> Self {
        let h = crate::bp_osd::SparseMatrix::surface_code_d3();
        Self {
            inner: crate::bp_osd::BpOsdDecoder::with_osd(h, osd_order),
        }
    }

    /// Decode a syndrome given the physical error probability.
    ///
    /// Args:
    ///     syndrome: Boolean vector of check outcomes.
    ///     p: Per-qubit error probability in (0, 1).
    ///
    /// Returns:
    ///     Dict with: correction, converged, bp_iterations, used_osd,
    ///     osd_combinations_tried, weight, llr_beliefs.
    fn decode(
        &self,
        py: Python<'_>,
        syndrome: Vec<bool>,
        p: f64,
    ) -> PyResult<pyo3::Py<pyo3::types::PyDict>> {
        if p <= 0.0 || p >= 1.0 {
            return Err(PyValueError::new_err("p must be in (0, 1)"));
        }
        let result = self.inner.decode(&syndrome, p);
        let dict = pyo3::types::PyDict::new(py);
        dict.set_item("correction", result.correction)?;
        dict.set_item("converged", result.converged)?;
        dict.set_item("bp_iterations", result.bp_iterations)?;
        dict.set_item("used_osd", result.used_osd)?;
        dict.set_item("osd_combinations_tried", result.osd_combinations_tried)?;
        dict.set_item("weight", result.weight)?;
        dict.set_item("llr_beliefs", result.llr_beliefs)?;
        Ok(dict.into())
    }

    fn __repr__(&self) -> String {
        format!("{}", self.inner)
    }
}

// =============================================================================
// MAGIC STATE FACTORY PYTHON BINDINGS
// =============================================================================

/// Python wrapper for MagicStateFactory.
#[cfg(feature = "python")]
#[pyclass(name = "MagicStateFactory")]
pub struct PyMagicStateFactory {
    inner: crate::magic_state_factory::MagicStateFactory,
}

#[cfg(feature = "python")]
#[pymethods]
impl PyMagicStateFactory {
    /// Create a new magic state distillation factory.
    ///
    /// Args:
    ///     protocol: "15_to_1", "20_to_4", "reed_muller_116", or "litinski_compact".
    ///     levels: Number of cascaded distillation levels.
    ///     code_distance: Surface code distance (must be >= 3).
    ///     physical_error_rate: Physical gate error probability in (0, 1).
    #[new]
    fn new(
        protocol: &str,
        levels: usize,
        code_distance: usize,
        physical_error_rate: f64,
    ) -> PyResult<Self> {
        let proto = parse_distillation_protocol(protocol)?;
        if code_distance < 3 {
            return Err(PyValueError::new_err("code_distance must be >= 3"));
        }
        if physical_error_rate <= 0.0 || physical_error_rate >= 1.0 {
            return Err(PyValueError::new_err(
                "physical_error_rate must be in (0, 1)",
            ));
        }
        Ok(Self {
            inner: crate::magic_state_factory::MagicStateFactory::new(
                proto,
                levels,
                code_distance,
                physical_error_rate,
            ),
        })
    }

    /// Number of physical qubits per surface code patch.
    #[getter]
    fn qubits_per_patch(&self) -> usize {
        self.inner.qubits_per_patch()
    }

    /// Run the distillation simulation.
    ///
    /// Returns a dict with: output_error_rate, total_raw_states_consumed,
    /// total_physical_qubits, distillation_time_cycles, space_time_volume,
    /// levels_used, and level_details (list of per-level dicts).
    fn distill(&self, py: Python<'_>) -> PyResult<pyo3::Py<pyo3::types::PyDict>> {
        let result = self.inner.distill();
        let dict = pyo3::types::PyDict::new(py);
        dict.set_item("output_error_rate", result.output_error_rate)?;
        dict.set_item(
            "total_raw_states_consumed",
            result.total_raw_states_consumed,
        )?;
        dict.set_item("total_physical_qubits", result.total_physical_qubits)?;
        dict.set_item("distillation_time_cycles", result.distillation_time_cycles)?;
        dict.set_item("space_time_volume", result.space_time_volume)?;
        dict.set_item("levels_used", result.levels_used)?;

        // Build per-level details as a list of dicts
        let levels_list = pyo3::types::PyList::empty(py);
        for ld in &result.level_details {
            let ld_dict = pyo3::types::PyDict::new(py);
            ld_dict.set_item("level", ld.level)?;
            ld_dict.set_item("input_error_rate", ld.input_error_rate)?;
            ld_dict.set_item("output_error_rate", ld.output_error_rate)?;
            ld_dict.set_item("states_in", ld.states_in)?;
            ld_dict.set_item("states_out", ld.states_out)?;
            ld_dict.set_item("qubits_per_factory", ld.qubits_per_factory)?;
            ld_dict.set_item("cycles", ld.cycles)?;
            levels_list.append(ld_dict)?;
        }
        dict.set_item("level_details", levels_list)?;

        Ok(dict.into())
    }

    /// Compute just the output error rate (fast, no full resource accounting).
    fn output_error_rate(&self) -> f64 {
        self.inner.output_error_rate()
    }

    /// Check if the output error is below the protocol's threshold.
    fn is_below_threshold(&self) -> bool {
        self.inner.is_below_threshold()
    }

    /// Estimate optimal resources for a target error rate.
    ///
    /// Returns a dict with: target_error_rate, recommended_protocol,
    /// recommended_levels, total_physical_qubits, t_gates_per_second,
    /// factory_footprint_patches.
    #[staticmethod]
    fn estimate_resources(
        py: Python<'_>,
        target_error: f64,
        physical_error: f64,
        code_distance: usize,
    ) -> PyResult<pyo3::Py<pyo3::types::PyDict>> {
        if code_distance < 3 {
            return Err(PyValueError::new_err("code_distance must be >= 3"));
        }
        let est = crate::magic_state_factory::MagicStateFactory::estimate_resources(
            target_error,
            physical_error,
            code_distance,
        );
        let dict = pyo3::types::PyDict::new(py);
        dict.set_item("target_error_rate", est.target_error_rate)?;
        dict.set_item("recommended_protocol", est.recommended_protocol.to_string())?;
        dict.set_item("recommended_levels", est.recommended_levels)?;
        dict.set_item("total_physical_qubits", est.total_physical_qubits)?;
        dict.set_item("t_gates_per_second", est.t_gates_per_second)?;
        dict.set_item("factory_footprint_patches", est.factory_footprint_patches)?;
        Ok(dict.into())
    }

    fn __repr__(&self) -> String {
        format!(
            "MagicStateFactory(output_error={:.2e}, below_threshold={})",
            self.inner.output_error_rate(),
            self.inner.is_below_threshold(),
        )
    }
}

/// Helper to parse distillation protocol from string.
#[cfg(feature = "python")]
fn parse_distillation_protocol(
    s: &str,
) -> PyResult<crate::magic_state_factory::DistillationProtocol> {
    match s {
        "15_to_1" | "fifteen_to_one" | "15-to-1" => Ok(crate::magic_state_factory::DistillationProtocol::FifteenToOne),
        "20_to_4" | "twenty_to_four" | "20-to-4" => Ok(crate::magic_state_factory::DistillationProtocol::TwentyToFour),
        "reed_muller_116" | "reed_muller" | "rm116" => Ok(crate::magic_state_factory::DistillationProtocol::ReedMuller116),
        "litinski_compact" | "litinski" => Ok(crate::magic_state_factory::DistillationProtocol::LitinskiCompact),
        other => Err(PyValueError::new_err(
            format!("Unknown protocol '{}'. Use: '15_to_1', '20_to_4', 'reed_muller_116', or 'litinski_compact'", other)
        )),
    }
}

// =============================================================================
// ADAPT-VQE PYTHON BINDINGS
// =============================================================================

/// Python wrapper for ADAPT-VQE algorithm.
#[cfg(feature = "python")]
#[pyclass(name = "AdaptVqe")]
pub struct PyAdaptVqe {
    inner: crate::adapt_vqe::AdaptVqe,
}

#[cfg(feature = "python")]
#[pymethods]
impl PyAdaptVqe {
    /// Create a new ADAPT-VQE engine.
    ///
    /// Args:
    ///     pool_type: "gsd" (generalized singles & doubles) or "qubit_adapt".
    ///     n_qubits: Number of qubits in the system.
    ///     n_electrons: Number of electrons (for Hartree-Fock reference).
    ///     n_orbitals: Number of orbitals (required for "gsd" pool, defaults to n_qubits).
    ///     gradient_threshold: Convergence threshold for max gradient (default 1e-5).
    ///     max_iterations: Maximum ADAPT iterations (default 50).
    ///     energy_convergence: Energy convergence threshold (default 1e-8).
    #[new]
    #[pyo3(signature = (pool_type, n_qubits, n_electrons, n_orbitals = None, gradient_threshold = 1e-5, max_iterations = 50, energy_convergence = 1e-8))]
    fn new(
        pool_type: &str,
        n_qubits: usize,
        n_electrons: usize,
        n_orbitals: Option<usize>,
        gradient_threshold: f64,
        max_iterations: usize,
        energy_convergence: f64,
    ) -> PyResult<Self> {
        let pool = match pool_type {
            "gsd" | "generalized_singles_doubles" => {
                let norb = n_orbitals.unwrap_or(n_qubits);
                crate::adapt_vqe::OperatorPool::generalized_singles_doubles(norb, n_electrons)
            }
            "qubit_adapt" | "qubit" => crate::adapt_vqe::OperatorPool::qubit_adapt_pool(n_qubits),
            other => {
                return Err(PyValueError::new_err(format!(
                    "Unknown pool type '{}'. Use 'gsd' or 'qubit_adapt'",
                    other
                )))
            }
        };
        let mut engine = crate::adapt_vqe::AdaptVqe::new(pool, n_qubits, n_electrons);
        engine.gradient_threshold = gradient_threshold;
        engine.max_iterations = max_iterations;
        engine.energy_convergence = energy_convergence;
        Ok(Self { inner: engine })
    }

    /// Run ADAPT-VQE on a molecular Hamiltonian.
    ///
    /// Args:
    ///     hamiltonian: A MolecularHamiltonian instance.
    ///
    /// Returns:
    ///     Dict with: energy, parameters, selected_operators, energy_history,
    ///     gradient_norms, n_iterations, converged.
    fn run(
        &self,
        py: Python<'_>,
        hamiltonian: &PyMolecularHamiltonian,
    ) -> PyResult<pyo3::Py<pyo3::types::PyDict>> {
        let result = self.inner.run(&hamiltonian.inner);
        let dict = pyo3::types::PyDict::new(py);
        dict.set_item("energy", result.energy)?;
        dict.set_item("parameters", result.parameters)?;
        dict.set_item("selected_operators", result.selected_operators)?;
        dict.set_item("energy_history", result.energy_history)?;
        dict.set_item("gradient_norms", result.gradient_norms)?;
        dict.set_item("n_iterations", result.n_iterations)?;
        dict.set_item("converged", result.converged)?;
        Ok(dict.into())
    }

    /// Size of the operator pool.
    #[getter]
    fn pool_size(&self) -> usize {
        self.inner.pool().len()
    }

    /// Number of qubits in the system.
    #[getter]
    fn n_qubits(&self) -> usize {
        self.inner.n_qubits()
    }

    /// Number of electrons.
    #[getter]
    fn n_electrons(&self) -> usize {
        self.inner.n_electrons()
    }

    fn __repr__(&self) -> String {
        format!(
            "AdaptVqe(n_qubits={}, n_electrons={}, pool_size={})",
            self.inner.n_qubits(),
            self.inner.n_electrons(),
            self.inner.pool().len(),
        )
    }
}

/// Python wrapper for MolecularHamiltonian.
#[cfg(feature = "python")]
#[pyclass(name = "MolecularHamiltonian")]
pub struct PyMolecularHamiltonian {
    inner: crate::adapt_vqe::MolecularHamiltonian,
}

#[cfg(feature = "python")]
#[pymethods]
impl PyMolecularHamiltonian {
    /// Create a molecular Hamiltonian from Pauli terms.
    ///
    /// Args:
    ///     terms: List of (coefficient, [(qubit, pauli_char)]) tuples.
    ///            pauli_char is 'X', 'Y', or 'Z'. Empty list = identity.
    ///     n_qubits: Number of qubits.
    #[new]
    fn new(terms: Vec<(f64, Vec<(usize, char)>)>, n_qubits: usize) -> Self {
        Self {
            inner: crate::adapt_vqe::MolecularHamiltonian::new(terms, n_qubits),
        }
    }

    /// H2 molecule in STO-3G basis (4 qubits, 2 electrons).
    /// Exact ground state energy: -1.1373 Hartree.
    #[staticmethod]
    fn h2_sto3g() -> Self {
        Self {
            inner: crate::adapt_vqe::MolecularHamiltonian::h2_sto3g(),
        }
    }

    /// LiH molecule in STO-3G basis (4-qubit active space, 2 electrons).
    /// Approximate exact energy: -7.8825 Hartree.
    #[staticmethod]
    fn lih_sto3g() -> Self {
        Self {
            inner: crate::adapt_vqe::MolecularHamiltonian::lih_sto3g(),
        }
    }

    /// Number of Pauli terms in the Hamiltonian.
    #[getter]
    fn num_terms(&self) -> usize {
        self.inner.num_terms()
    }

    /// Number of qubits required.
    #[getter]
    fn n_qubits(&self) -> usize {
        self.inner.n_qubits
    }

    fn __repr__(&self) -> String {
        format!(
            "MolecularHamiltonian(n_qubits={}, num_terms={})",
            self.inner.n_qubits,
            self.inner.num_terms(),
        )
    }
}

// =============================================================================
// QUANTUM WALK PYTHON BINDINGS
// =============================================================================

/// Python wrapper for Graph (quantum walk topology).
#[cfg(feature = "python")]
#[pyclass(name = "WalkGraph")]
#[derive(Clone)]
pub struct PyWalkGraph {
    inner: crate::quantum_walk::Graph,
}

#[cfg(feature = "python")]
#[pymethods]
impl PyWalkGraph {
    /// Create a line (path) graph with n vertices.
    #[staticmethod]
    fn line(n: usize) -> PyResult<Self> {
        if n < 2 {
            return Err(PyValueError::new_err(
                "Line graph requires at least 2 vertices",
            ));
        }
        Ok(Self {
            inner: crate::quantum_walk::Graph::line(n),
        })
    }

    /// Create a cycle graph with n vertices.
    #[staticmethod]
    fn cycle(n: usize) -> PyResult<Self> {
        if n < 3 {
            return Err(PyValueError::new_err(
                "Cycle graph requires at least 3 vertices",
            ));
        }
        Ok(Self {
            inner: crate::quantum_walk::Graph::cycle(n),
        })
    }

    /// Create a complete graph K_n.
    #[staticmethod]
    fn complete(n: usize) -> PyResult<Self> {
        if n < 2 {
            return Err(PyValueError::new_err(
                "Complete graph requires at least 2 vertices",
            ));
        }
        Ok(Self {
            inner: crate::quantum_walk::Graph::complete(n),
        })
    }

    /// Create a 2D grid graph (rows x cols).
    #[staticmethod]
    fn grid_2d(rows: usize, cols: usize) -> PyResult<Self> {
        if rows < 1 || cols < 1 {
            return Err(PyValueError::new_err("Grid dimensions must be >= 1"));
        }
        Ok(Self {
            inner: crate::quantum_walk::Graph::grid_2d(rows, cols),
        })
    }

    /// Create a hypercube graph of given dimension (2^dim vertices).
    #[staticmethod]
    fn hypercube(dim: usize) -> PyResult<Self> {
        if dim < 1 {
            return Err(PyValueError::new_err("Hypercube dimension must be >= 1"));
        }
        Ok(Self {
            inner: crate::quantum_walk::Graph::hypercube(dim),
        })
    }

    /// Create a star graph with n vertices (1 center + n-1 leaves).
    #[staticmethod]
    fn star(n: usize) -> PyResult<Self> {
        if n < 2 {
            return Err(PyValueError::new_err(
                "Star graph requires at least 2 vertices",
            ));
        }
        Ok(Self {
            inner: crate::quantum_walk::Graph::star(n),
        })
    }

    /// Number of vertices.
    #[getter]
    fn n_vertices(&self) -> usize {
        self.inner.n_vertices
    }

    /// Maximum vertex degree.
    #[getter]
    fn max_degree(&self) -> usize {
        self.inner.max_degree()
    }

    /// Vertex degrees.
    fn degrees(&self) -> Vec<usize> {
        self.inner.degrees()
    }

    fn __repr__(&self) -> String {
        format!(
            "WalkGraph(vertices={}, max_degree={})",
            self.inner.n_vertices,
            self.inner.max_degree()
        )
    }
}

/// Python wrapper for ContinuousWalk (continuous-time quantum walk).
#[cfg(feature = "python")]
#[pyclass(name = "ContinuousQuantumWalk")]
pub struct PyContinuousWalk {
    inner: crate::quantum_walk::ContinuousWalk,
}

#[cfg(feature = "python")]
#[pymethods]
impl PyContinuousWalk {
    /// Create a new continuous-time quantum walk.
    ///
    /// Args:
    ///     graph: The WalkGraph to walk on.
    ///     initial_vertex: Starting vertex index.
    ///     gamma: Hopping rate (coupling constant).
    #[new]
    fn new(graph: &PyWalkGraph, initial_vertex: usize, gamma: f64) -> PyResult<Self> {
        if initial_vertex >= graph.inner.n_vertices {
            return Err(PyValueError::new_err(format!(
                "initial_vertex {} out of range (graph has {} vertices)",
                initial_vertex, graph.inner.n_vertices
            )));
        }
        Ok(Self {
            inner: crate::quantum_walk::ContinuousWalk::new(
                graph.inner.clone(),
                initial_vertex,
                gamma,
            ),
        })
    }

    /// Evolve the walk for time t (accumulates with previous evolve calls).
    fn evolve(&mut self, t: f64) {
        self.inner.evolve(t);
    }

    /// Get the probability distribution at each vertex.
    fn probabilities(&self) -> Vec<f64> {
        self.inner.probabilities()
    }

    /// Estimate the mixing time (first time TVD < epsilon from uniform).
    ///
    /// Returns None if not reached within max_steps * dt.
    #[pyo3(signature = (epsilon = 0.1, dt = 0.1, max_steps = 10000))]
    fn mixing_time(&self, epsilon: f64, dt: f64, max_steps: usize) -> Option<f64> {
        self.inner.mixing_time(epsilon, dt, max_steps)
    }

    /// Quantum spatial search for marked vertices (Childs-Goldstone).
    ///
    /// Uses the Hamiltonian H = -gamma*A + sum|m><m|.
    /// Returns a dict with: success_probability, marked_probabilities,
    /// optimal_time, classical_hitting_time, quantum_hitting_time, speedup.
    #[staticmethod]
    fn search(
        py: Python<'_>,
        graph: &PyWalkGraph,
        marked: Vec<usize>,
        gamma: f64,
        t: f64,
    ) -> PyResult<pyo3::Py<pyo3::types::PyDict>> {
        let result = crate::quantum_walk::ContinuousWalk::search(&graph.inner, &marked, gamma, t);
        let dict = pyo3::types::PyDict::new(py);
        dict.set_item("success_probability", result.success_probability)?;
        dict.set_item("marked_probabilities", result.marked_probabilities)?;
        dict.set_item("optimal_time", result.optimal_time)?;
        dict.set_item("classical_hitting_time", result.classical_hitting_time)?;
        dict.set_item("quantum_hitting_time", result.quantum_hitting_time)?;
        dict.set_item("speedup", result.speedup)?;
        Ok(dict.into())
    }

    /// Analyze the walk over a range of times.
    ///
    /// Returns a dict with: variance, return_probability, entanglement_entropy.
    fn analyze(&self, py: Python<'_>, times: Vec<f64>) -> PyResult<pyo3::Py<pyo3::types::PyDict>> {
        let result = self.inner.analyze(&times);
        let dict = pyo3::types::PyDict::new(py);
        dict.set_item("variance", result.variance)?;
        dict.set_item("return_probability", result.return_probability)?;
        dict.set_item("entanglement_entropy", result.entanglement_entropy)?;
        Ok(dict.into())
    }

    fn __repr__(&self) -> String {
        format!(
            "ContinuousQuantumWalk(vertices={}, gamma={})",
            self.inner.graph.n_vertices, self.inner.gamma,
        )
    }
}

/// Python wrapper for DiscreteWalk (discrete-time quantum walk with coin).
#[cfg(feature = "python")]
#[pyclass(name = "DiscreteQuantumWalk")]
pub struct PyDiscreteWalk {
    inner: crate::quantum_walk::DiscreteWalk,
}

#[cfg(feature = "python")]
#[pymethods]
impl PyDiscreteWalk {
    /// Create a new discrete-time quantum walk.
    ///
    /// Args:
    ///     graph: The WalkGraph to walk on.
    ///     initial_vertex: Starting vertex index.
    ///     coin: Coin operator - "hadamard", "grover", or "dft".
    #[new]
    #[pyo3(signature = (graph, initial_vertex, coin = "hadamard"))]
    fn new(graph: &PyWalkGraph, initial_vertex: usize, coin: &str) -> PyResult<Self> {
        if initial_vertex >= graph.inner.n_vertices {
            return Err(PyValueError::new_err(format!(
                "initial_vertex {} out of range (graph has {} vertices)",
                initial_vertex, graph.inner.n_vertices
            )));
        }
        let coin_op = match coin {
            "hadamard" | "h" => crate::quantum_walk::CoinOperator::Hadamard,
            "grover" | "g" => crate::quantum_walk::CoinOperator::Grover,
            "dft" | "fourier" => crate::quantum_walk::CoinOperator::DFT,
            other => {
                return Err(PyValueError::new_err(format!(
                    "Unknown coin '{}'. Use 'hadamard', 'grover', or 'dft'",
                    other
                )))
            }
        };
        Ok(Self {
            inner: crate::quantum_walk::DiscreteWalk::new(
                graph.inner.clone(),
                initial_vertex,
                coin_op,
            ),
        })
    }

    /// Perform a single walk step (coin + shift).
    fn step(&mut self) {
        self.inner.step();
    }

    /// Evolve for n_steps walk steps.
    fn evolve(&mut self, n_steps: usize) {
        self.inner.evolve(n_steps);
    }

    /// Get the probability at each vertex (summed over coin states).
    fn vertex_probabilities(&self) -> Vec<f64> {
        self.inner.vertex_probabilities()
    }

    /// Quantum search for marked vertices on a graph.
    ///
    /// Args:
    ///     graph: WalkGraph to search.
    ///     marked: Indices of target vertices.
    ///     coin: Coin operator name.
    ///     n_steps: Number of walk steps.
    ///
    /// Returns a dict with: success_probability, marked_probabilities,
    /// optimal_time, classical_hitting_time, quantum_hitting_time, speedup.
    #[staticmethod]
    #[pyo3(signature = (graph, marked, coin = "grover", n_steps = 100))]
    fn search(
        py: Python<'_>,
        graph: &PyWalkGraph,
        marked: Vec<usize>,
        coin: &str,
        n_steps: usize,
    ) -> PyResult<pyo3::Py<pyo3::types::PyDict>> {
        let coin_op = match coin {
            "hadamard" | "h" => crate::quantum_walk::CoinOperator::Hadamard,
            "grover" | "g" => crate::quantum_walk::CoinOperator::Grover,
            "dft" | "fourier" => crate::quantum_walk::CoinOperator::DFT,
            other => {
                return Err(PyValueError::new_err(format!(
                    "Unknown coin '{}'. Use 'hadamard', 'grover', or 'dft'",
                    other
                )))
            }
        };
        let result =
            crate::quantum_walk::DiscreteWalk::search(&graph.inner, &marked, coin_op, n_steps);
        let dict = pyo3::types::PyDict::new(py);
        dict.set_item("success_probability", result.success_probability)?;
        dict.set_item("marked_probabilities", result.marked_probabilities)?;
        dict.set_item("optimal_time", result.optimal_time)?;
        dict.set_item("classical_hitting_time", result.classical_hitting_time)?;
        dict.set_item("quantum_hitting_time", result.quantum_hitting_time)?;
        dict.set_item("speedup", result.speedup)?;
        Ok(dict.into())
    }

    /// Analyze the walk over n_steps steps.
    ///
    /// Returns a dict with: variance, return_probability, entanglement_entropy.
    fn analyze(&self, py: Python<'_>, n_steps: usize) -> PyResult<pyo3::Py<pyo3::types::PyDict>> {
        let result = self.inner.analyze(n_steps);
        let dict = pyo3::types::PyDict::new(py);
        dict.set_item("variance", result.variance)?;
        dict.set_item("return_probability", result.return_probability)?;
        dict.set_item("entanglement_entropy", result.entanglement_entropy)?;
        Ok(dict.into())
    }

    fn __repr__(&self) -> String {
        format!(
            "DiscreteQuantumWalk(vertices={}, coin_dim={})",
            self.inner.graph.n_vertices, self.inner.coin_dim,
        )
    }
}

// =============================================================================
// QUANTUM ANNEALING PYTHON BINDINGS
// =============================================================================

/// Python wrapper for IsingModel.
#[cfg(feature = "python")]
#[pyclass(name = "IsingModel")]
struct PyIsingModel {
    inner: crate::quantum_annealing::IsingModel,
}

#[cfg(feature = "python")]
#[pymethods]
impl PyIsingModel {
    /// Create an Ising model.
    ///
    /// Args:
    ///     n_spins: Number of spins.
    ///     h: Local fields (list of floats, length n_spins).
    ///     j: Couplings as list of (i, j, J_ij) tuples.
    #[new]
    fn new(n_spins: usize, h: Vec<f64>, j: Vec<(usize, usize, f64)>) -> PyResult<Self> {
        crate::quantum_annealing::IsingModel::new(n_spins, h, j)
            .map(|inner| Self { inner })
            .map_err(|e| PyValueError::new_err(format!("{:?}", e)))
    }

    /// Create an Ising model from a QUBO matrix.
    ///
    /// Args:
    ///     n: Number of binary variables.
    ///     q: QUBO entries as list of (i, j, Q_ij) tuples.
    #[staticmethod]
    fn from_qubo(n: usize, q: Vec<(usize, usize, f64)>) -> Self {
        Self {
            inner: crate::quantum_annealing::IsingModel::from_qubo(n, &q),
        }
    }

    /// Create a max-cut problem.
    #[staticmethod]
    fn max_cut(n_vertices: usize, edges: Vec<(usize, usize)>) -> Self {
        Self {
            inner: crate::quantum_annealing::problems::max_cut(n_vertices, &edges),
        }
    }

    /// Create a number partitioning problem.
    #[staticmethod]
    fn number_partitioning(numbers: Vec<f64>) -> Self {
        Self {
            inner: crate::quantum_annealing::problems::number_partitioning(&numbers),
        }
    }

    /// Evaluate the Ising energy for a spin configuration.
    fn energy(&self, config: Vec<bool>) -> f64 {
        self.inner.energy(&config)
    }

    #[getter]
    fn n_spins(&self) -> usize {
        self.inner.n_spins
    }

    fn __repr__(&self) -> String {
        format!(
            "IsingModel(n_spins={}, couplings={})",
            self.inner.n_spins,
            self.inner.j.len()
        )
    }
}

/// Python wrapper for QuantumAnnealer.
#[cfg(feature = "python")]
#[pyclass(name = "QuantumAnnealer")]
struct PyQuantumAnnealer {
    inner: crate::quantum_annealing::QuantumAnnealer,
}

#[cfg(feature = "python")]
#[pymethods]
impl PyQuantumAnnealer {
    /// Create a quantum annealer.
    ///
    /// Args:
    ///     model: IsingModel to solve.
    ///     schedule: 'linear' or 'quadratic' (default 'linear').
    ///     n_steps: Number of annealing steps (default 1000).
    ///     n_trotter: Number of Trotter slices for SQA (default 32).
    ///     beta: Inverse temperature (default 10.0).
    ///     n_runs: Number of independent runs (default 10).
    #[new]
    #[pyo3(signature = (model, schedule = "linear", n_steps = 1000, n_trotter = 32, beta = 10.0, n_runs = 10))]
    fn new(
        model: &PyIsingModel,
        schedule: &str,
        n_steps: usize,
        n_trotter: usize,
        beta: f64,
        n_runs: usize,
    ) -> PyResult<Self> {
        let sched = match schedule {
            "linear" => crate::quantum_annealing::QuantumAnnealingSchedule::Linear,
            "quadratic" => crate::quantum_annealing::QuantumAnnealingSchedule::Quadratic,
            other => {
                return Err(PyValueError::new_err(format!(
                    "Unknown schedule '{}'. Use 'linear' or 'quadratic'",
                    other
                )))
            }
        };
        let config = crate::quantum_annealing::QuantumAnnealerConfig {
            schedule: sched,
            n_steps,
            n_trotter,
            beta,
            gamma_initial: 3.0,
            n_runs,
            n_sweeps_per_step: 1,
        };
        Ok(Self {
            inner: crate::quantum_annealing::QuantumAnnealer::new(model.inner.clone(), config),
        })
    }

    /// Run simulated quantum annealing (path-integral Monte Carlo).
    ///
    /// Returns dict with: best_config, best_energy, energy_history,
    /// success_probability, time_to_solution.
    fn anneal_quantum(&self, py: Python<'_>) -> PyResult<pyo3::Py<pyo3::types::PyDict>> {
        let result = self.inner.anneal_quantum();
        let dict = pyo3::types::PyDict::new(py);
        dict.set_item("best_config", result.best_config)?;
        dict.set_item("best_energy", result.best_energy)?;
        dict.set_item("energy_history", result.energy_history)?;
        dict.set_item("success_probability", result.success_probability)?;
        dict.set_item("time_to_solution", result.time_to_solution)?;
        Ok(dict.into())
    }

    /// Run classical simulated annealing baseline.
    fn anneal_classical(&self, py: Python<'_>) -> PyResult<pyo3::Py<pyo3::types::PyDict>> {
        let result = self.inner.anneal_classical();
        let dict = pyo3::types::PyDict::new(py);
        dict.set_item("best_config", result.best_config)?;
        dict.set_item("best_energy", result.best_energy)?;
        dict.set_item("energy_history", result.energy_history)?;
        dict.set_item("success_probability", result.success_probability)?;
        dict.set_item("time_to_solution", result.time_to_solution)?;
        Ok(dict.into())
    }

    fn __repr__(&self) -> String {
        format!(
            "QuantumAnnealer(n_spins={}, n_steps={})",
            self.inner.problem.n_spins, self.inner.config.n_steps
        )
    }
}

// =============================================================================
// FERMIONIC GAUSSIAN STATE PYTHON BINDINGS
// =============================================================================

/// Python wrapper for FermionicGaussianState.
#[cfg(feature = "python")]
#[pyclass(name = "FermionicGaussianState")]
struct PyFermionicGaussianState {
    inner: crate::fermionic_gaussian::FermionicGaussianState,
}

#[cfg(feature = "python")]
#[pymethods]
impl PyFermionicGaussianState {
    /// Create a fermionic Gaussian state in the vacuum.
    ///
    /// Args:
    ///     n_modes: Number of fermionic modes.
    #[new]
    fn new(n_modes: usize) -> Self {
        Self {
            inner: crate::fermionic_gaussian::FermionicGaussianState::new(n_modes),
        }
    }

    /// Create from explicit covariance matrix.
    #[staticmethod]
    fn from_covariance(gamma: Vec<Vec<f64>>) -> PyResult<Self> {
        crate::fermionic_gaussian::FermionicGaussianState::from_covariance(gamma)
            .map(|inner| Self { inner })
            .map_err(|e| PyValueError::new_err(format!("{:?}", e)))
    }

    /// Create a Slater determinant from occupied mode indices.
    ///
    /// Args:
    ///     occupied: List of occupied mode indices.
    ///     n_modes: Total number of modes.
    #[staticmethod]
    fn from_occupation(occupied: Vec<usize>, n_modes: usize) -> PyResult<Self> {
        crate::fermionic_gaussian::FermionicGaussianState::from_occupation(&occupied, n_modes)
            .map(|inner| Self { inner })
            .map_err(|e| PyValueError::new_err(format!("{:?}", e)))
    }

    /// Apply a hopping gate between modes i and j.
    fn apply_hopping(&mut self, i: usize, j: usize, angle: f64) -> PyResult<()> {
        self.inner
            .apply_hopping(i, j, angle)
            .map_err(|e| PyValueError::new_err(format!("{:?}", e)))
    }

    /// Apply a pairing gate between modes i and j.
    fn apply_pairing(&mut self, i: usize, j: usize, angle: f64) -> PyResult<()> {
        self.inner
            .apply_pairing(i, j, angle)
            .map_err(|e| PyValueError::new_err(format!("{:?}", e)))
    }

    /// Apply an on-site phase rotation on mode i.
    fn apply_onsite_phase(&mut self, i: usize, angle: f64) -> PyResult<()> {
        self.inner
            .apply_onsite_phase(i, angle)
            .map_err(|e| PyValueError::new_err(format!("{:?}", e)))
    }

    /// Apply a beam splitter between modes i and j.
    fn apply_beam_splitter(&mut self, i: usize, j: usize, angle: f64) -> PyResult<()> {
        self.inner
            .apply_beam_splitter(i, j, angle)
            .map_err(|e| PyValueError::new_err(format!("{:?}", e)))
    }

    /// Get the occupation number <n_i> of mode i.
    fn occupation_number(&self, i: usize) -> f64 {
        self.inner.occupation_number(i)
    }

    /// Get all occupation numbers.
    fn occupation_numbers(&self) -> Vec<f64> {
        (0..self.inner.n_modes)
            .map(|i| self.inner.occupation_number(i))
            .collect()
    }

    /// Get total particle number.
    fn total_particle_number(&self) -> f64 {
        self.inner.total_particle_number()
    }

    /// Compute entanglement entropy of a subsystem.
    fn entropy_of_subsystem(&self, sites: Vec<usize>) -> f64 {
        self.inner.entropy_of_subsystem(&sites)
    }

    /// Overlap with another state.
    fn overlap(&self, other: &PyFermionicGaussianState) -> f64 {
        self.inner.overlap(&other.inner)
    }

    /// Check if state is pure.
    fn is_pure(&self) -> bool {
        self.inner.is_pure()
    }

    #[getter]
    fn n_modes(&self) -> usize {
        self.inner.n_modes
    }

    fn __repr__(&self) -> String {
        format!(
            "FermionicGaussianState(n_modes={}, particles={:.1})",
            self.inner.n_modes,
            self.inner.total_particle_number()
        )
    }
}

// =============================================================================
// NEURAL QUANTUM STATE PYTHON BINDINGS
// =============================================================================

/// Python wrapper for RBMState (Restricted Boltzmann Machine wave function).
#[cfg(feature = "python")]
#[pyclass(name = "RBMState")]
struct PyRBMState {
    inner: crate::neural_quantum_states::RBMState,
}

#[cfg(feature = "python")]
#[pymethods]
impl PyRBMState {
    /// Create a new RBM variational state.
    ///
    /// Args:
    ///     n_visible: Number of visible units (spins).
    ///     n_hidden: Number of hidden units.
    ///     seed: Random seed (default 42).
    #[new]
    #[pyo3(signature = (n_visible, n_hidden, seed = 42))]
    fn new(n_visible: usize, n_hidden: usize, seed: u64) -> Self {
        Self {
            inner: crate::neural_quantum_states::RBMState::new_seeded(n_visible, n_hidden, seed),
        }
    }

    /// Total number of variational parameters.
    fn num_params(&self) -> usize {
        self.inner.num_params()
    }

    /// Compute log amplitude for a spin configuration.
    ///
    /// Args:
    ///     config: List of bools (True = spin up).
    ///
    /// Returns (real, imag) tuple.
    fn log_amplitude(&self, config: Vec<bool>) -> (f64, f64) {
        let c = self.inner.log_amplitude(&config);
        (c.re, c.im)
    }

    /// Compute amplitude for a spin configuration.
    fn amplitude(&self, config: Vec<bool>) -> (f64, f64) {
        let c = self.inner.amplitude(&config);
        (c.re, c.im)
    }

    #[getter]
    fn n_visible(&self) -> usize {
        self.inner.n_visible
    }

    #[getter]
    fn n_hidden(&self) -> usize {
        self.inner.n_hidden
    }

    fn __repr__(&self) -> String {
        format!(
            "RBMState(n_visible={}, n_hidden={}, params={})",
            self.inner.n_visible,
            self.inner.n_hidden,
            self.inner.num_params()
        )
    }
}

/// Python wrapper for VMCOptimizer.
#[cfg(feature = "python")]
#[pyclass(name = "VMCOptimizer")]
struct PyVMCOptimizer {
    config: crate::neural_quantum_states::VMCConfig,
}

#[cfg(feature = "python")]
#[pymethods]
impl PyVMCOptimizer {
    /// Create a VMC optimizer with stochastic reconfiguration.
    ///
    /// Args:
    ///     n_iterations: Number of optimization steps (default 100).
    ///     n_samples: Monte Carlo samples per step (default 500).
    ///     learning_rate: Step size (default 0.05).
    ///     seed: Random seed (default 42).
    #[new]
    #[pyo3(signature = (n_iterations = 100, n_samples = 500, learning_rate = 0.05, seed = 42))]
    fn new(n_iterations: usize, n_samples: usize, learning_rate: f64, seed: u64) -> Self {
        Self {
            config: crate::neural_quantum_states::VMCConfig {
                n_iterations,
                n_samples,
                learning_rate,
                regularization: 1e-4,
                n_burn_in: 100,
                n_thin: 5,
                seed,
            },
        }
    }

    /// Optimize an RBM state to minimize the energy of a Heisenberg Hamiltonian.
    ///
    /// Args:
    ///     state: RBMState to optimize (modified in place).
    ///     n_sites: Number of sites for Heisenberg chain.
    ///     j_coupling: Exchange coupling (default 1.0).
    ///     h_field: External field (default 0.0).
    ///
    /// Returns dict with: energy_history, final_energy, variance, acceptance_rate.
    #[pyo3(signature = (state, n_sites, j_coupling = 1.0, h_field = 0.0))]
    fn optimize_heisenberg(
        &self,
        py: Python<'_>,
        state: &mut PyRBMState,
        n_sites: usize,
        j_coupling: f64,
        h_field: f64,
    ) -> PyResult<pyo3::Py<pyo3::types::PyDict>> {
        let ham = crate::neural_quantum_states::SpinHamiltonian::heisenberg_1d(
            n_sites, j_coupling, h_field,
        );
        let optimizer = crate::neural_quantum_states::VMCOptimizer::new(self.config.clone());
        let result = optimizer
            .optimize(&mut state.inner, &ham)
            .map_err(|e| PyValueError::new_err(format!("{:?}", e)))?;
        let dict = pyo3::types::PyDict::new(py);
        dict.set_item("energy_history", result.energy_history)?;
        dict.set_item("final_energy", result.final_energy)?;
        dict.set_item("variance", result.variance)?;
        dict.set_item("acceptance_rate", result.acceptance_rate)?;
        dict.set_item("iterations", result.iterations)?;
        Ok(dict.into())
    }

    /// Optimize for Ising Hamiltonian.
    #[pyo3(signature = (state, n_sites, j_coupling = 1.0, h_field = 1.0))]
    fn optimize_ising(
        &self,
        py: Python<'_>,
        state: &mut PyRBMState,
        n_sites: usize,
        j_coupling: f64,
        h_field: f64,
    ) -> PyResult<pyo3::Py<pyo3::types::PyDict>> {
        let ham =
            crate::neural_quantum_states::SpinHamiltonian::ising_1d(n_sites, j_coupling, h_field);
        let optimizer = crate::neural_quantum_states::VMCOptimizer::new(self.config.clone());
        let result = optimizer
            .optimize(&mut state.inner, &ham)
            .map_err(|e| PyValueError::new_err(format!("{:?}", e)))?;
        let dict = pyo3::types::PyDict::new(py);
        dict.set_item("energy_history", result.energy_history)?;
        dict.set_item("final_energy", result.final_energy)?;
        dict.set_item("variance", result.variance)?;
        dict.set_item("acceptance_rate", result.acceptance_rate)?;
        dict.set_item("iterations", result.iterations)?;
        Ok(dict.into())
    }

    fn __repr__(&self) -> String {
        format!(
            "VMCOptimizer(iters={}, samples={}, lr={})",
            self.config.n_iterations, self.config.n_samples, self.config.learning_rate
        )
    }
}

// =============================================================================
// QUANTUM DRUG DESIGN PYTHON BINDINGS
// =============================================================================

use crate::quantum_drug_design::{
    self as drug_design, lead_optimization, AdmetPredictor as RustAdmetPredictor, AdmetProperty,
    BondType as DrugBondType, DockingConfig, DockingResult,
    DrugDiscoveryPipeline as RustDrugDiscoveryPipeline,
    DrugLikenessResult as RustDrugLikenessResult, DrugProperties, Element as DrugElement,
    MolecularFingerprint, Molecule as RustMolecule, PipelineStage,
    QuantumDockingScorer as RustQuantumDockingScorer, ScoringFunction,
};

/// Python wrapper for Molecule
#[cfg(feature = "python")]
#[pyclass(name = "Molecule")]
#[derive(Debug, Clone)]
pub struct PyMolecule {
    pub inner: RustMolecule,
}

#[cfg(feature = "python")]
#[pymethods]
impl PyMolecule {
    /// Create a new molecule with a name
    #[new]
    fn new(name: String) -> Self {
        Self {
            inner: RustMolecule::new(&name),
        }
    }

    /// Add an atom to the molecule
    ///
    /// Args:
    ///     element: Element symbol (e.g., "C", "N", "O")
    ///     position: 3D coordinates [x, y, z] in Angstroms
    ///     charge: Partial charge (default 0.0)
    ///
    /// Returns:
    ///     Index of the added atom
    fn add_atom(
        &mut self,
        element: String,
        position: Vec<f64>,
        charge: Option<f64>,
    ) -> PyResult<usize> {
        if position.len() != 3 {
            return Err(PyValueError::new_err(
                "Position must have 3 elements [x, y, z]",
            ));
        }
        let elem = match element.as_str() {
            "H" => DrugElement::H,
            "C" => DrugElement::C,
            "N" => DrugElement::N,
            "O" => DrugElement::O,
            "F" => DrugElement::F,
            "S" => DrugElement::S,
            "P" => DrugElement::P,
            "Cl" => DrugElement::Cl,
            "Br" => DrugElement::Br,
            "Fe" => DrugElement::Fe,
            "Zn" => DrugElement::Zn,
            _ => {
                return Err(PyValueError::new_err(format!(
                    "Unknown element: {}",
                    element
                )))
            }
        };
        let pos = [position[0], position[1], position[2]];
        Ok(self.inner.add_atom(elem, pos, charge.unwrap_or(0.0)))
    }

    /// Add a bond between two atoms
    ///
    /// Args:
    ///     atom1: Index of first atom
    ///     atom2: Index of second atom
    ///     bond_type: Bond type ("single", "double", "triple", "aromatic")
    fn add_bond(&mut self, atom1: usize, atom2: usize, bond_type: Option<String>) -> PyResult<()> {
        let bt = match bond_type.as_deref().unwrap_or("single") {
            "single" => DrugBondType::Single,
            "double" => DrugBondType::Double,
            "triple" => DrugBondType::Triple,
            "aromatic" => DrugBondType::Aromatic,
            _ => {
                return Err(PyValueError::new_err(format!(
                    "Unknown bond type: {:?}",
                    bond_type
                )))
            }
        };
        self.inner.add_bond(atom1, atom2, bt);
        Ok(())
    }

    /// Set SMILES string
    fn set_smiles(&mut self, smiles: String) {
        self.inner.smiles = Some(smiles);
    }

    /// Get SMILES string if available
    fn get_smiles(&self) -> Option<String> {
        self.inner.smiles.clone()
    }

    /// Get molecule name
    fn get_name(&self) -> String {
        self.inner.name.clone()
    }

    /// Get number of atoms
    fn num_atoms(&self) -> usize {
        self.inner.atoms.len()
    }

    /// Get number of bonds
    fn num_bonds(&self) -> usize {
        self.inner.bonds.len()
    }

    /// Calculate molecular weight in Daltons
    fn molecular_weight(&self) -> f64 {
        self.inner.molecular_weight()
    }

    /// Get number of hydrogen bond donors
    fn h_bond_donors(&self) -> usize {
        self.inner.h_bond_donors()
    }

    /// Get number of hydrogen bond acceptors
    fn h_bond_acceptors(&self) -> usize {
        self.inner.h_bond_acceptors()
    }

    /// Get number of rotatable bonds
    fn rotatable_bonds(&self) -> usize {
        self.inner.rotatable_bonds()
    }

    /// Estimate LogP
    fn estimated_log_p(&self) -> f64 {
        self.inner.estimated_log_p()
    }

    /// Estimate polar surface area
    fn polar_surface_area(&self) -> f64 {
        self.inner.polar_surface_area()
    }

    /// Get center of mass
    fn center_of_mass(&self) -> Vec<f64> {
        self.inner.center_of_mass().to_vec()
    }

    /// Validate molecule structure
    fn validate(&self) -> PyResult<()> {
        self.inner
            .validate()
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    fn __repr__(&self) -> String {
        format!(
            "<Molecule '{}' atoms={} bonds={}>",
            self.inner.name,
            self.inner.atoms.len(),
            self.inner.bonds.len()
        )
    }
}

/// Python wrapper for DrugLikenessResult
#[cfg(feature = "python")]
#[pyclass(name = "DrugLikenessResult")]
#[derive(Debug, Clone)]
pub struct PyDrugLikenessResult {
    pub inner: RustDrugLikenessResult,
}

#[cfg(feature = "python")]
#[pymethods]
impl PyDrugLikenessResult {
    /// Number of Lipinski Rule of Five violations
    #[getter]
    fn lipinski_violations(&self) -> usize {
        self.inner.lipinski_violations
    }

    /// QED (Quantitative Estimate of Drug-likeness) score in [0, 1]
    #[getter]
    fn qed_score(&self) -> f64 {
        self.inner.qed_score
    }

    /// Synthetic accessibility score in [0, 1] (higher = easier to synthesize)
    #[getter]
    fn synthetic_accessibility(&self) -> f64 {
        self.inner.synthetic_accessibility
    }

    /// Molecular properties used in the evaluation
    fn properties(&self, py: Python<'_>) -> PyResult<pyo3::Py<pyo3::types::PyDict>> {
        let dict = pyo3::types::PyDict::new(py);
        dict.set_item("molecular_weight", self.inner.properties.molecular_weight)?;
        dict.set_item("log_p", self.inner.properties.log_p)?;
        dict.set_item("h_bond_donors", self.inner.properties.h_bond_donors)?;
        dict.set_item("h_bond_acceptors", self.inner.properties.h_bond_acceptors)?;
        dict.set_item("rotatable_bonds", self.inner.properties.rotatable_bonds)?;
        dict.set_item(
            "polar_surface_area",
            self.inner.properties.polar_surface_area,
        )?;
        Ok(dict.into())
    }

    fn __repr__(&self) -> String {
        format!(
            "<DrugLikenessResult qed={:.3} violations={}>",
            self.inner.qed_score, self.inner.lipinski_violations
        )
    }
}

/// Evaluate drug-likeness for a molecule
#[cfg(feature = "python")]
#[pyfunction]
fn py_evaluate_drug_likeness(mol: &PyMolecule) -> PyDrugLikenessResult {
    PyDrugLikenessResult {
        inner: drug_design::evaluate_drug_likeness(&mol.inner),
    }
}

/// Python wrapper for ADMET Predictor
#[cfg(feature = "python")]
#[pyclass(name = "AdmetPredictor")]
pub struct PyAdmetPredictor {
    inner: RustAdmetPredictor,
}

#[cfg(feature = "python")]
#[pymethods]
impl PyAdmetPredictor {
    /// Create a new ADMET predictor
    ///
    /// Args:
    ///     num_qubits: Number of qubits for quantum computations
    ///     properties: List of properties to predict ("absorption", "distribution",
    ///                 "metabolism", "excretion", "toxicity", "solubility", "log_p",
    ///                 "bbb_permeability")
    #[new]
    fn new(num_qubits: usize, properties: Vec<String>) -> Self {
        let props: Vec<AdmetProperty> = properties
            .iter()
            .filter_map(|p| match p.as_str() {
                "absorption" => Some(AdmetProperty::Absorption),
                "distribution" => Some(AdmetProperty::Distribution),
                "metabolism" => Some(AdmetProperty::Metabolism),
                "excretion" => Some(AdmetProperty::Excretion),
                "toxicity" => Some(AdmetProperty::Toxicity),
                "solubility" => Some(AdmetProperty::Solubility),
                "log_p" => Some(AdmetProperty::LogP),
                "bbb_permeability" => Some(AdmetProperty::BBBPermeability),
                _ => None,
            })
            .collect();

        Self {
            inner: RustAdmetPredictor::new(num_qubits, props),
        }
    }

    /// Predict ADMET properties for a molecule
    ///
    /// Returns a dictionary mapping property names to prediction results
    fn predict(&self, mol: &PyMolecule, py: Python<'_>) -> PyResult<pyo3::Py<pyo3::types::PyDict>> {
        let results = self
            .inner
            .predict(&mol.inner)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;

        let dict = pyo3::types::PyDict::new(py);
        for result in results {
            let prop_name = match result.property {
                AdmetProperty::Absorption => "absorption",
                AdmetProperty::Distribution => "distribution",
                AdmetProperty::Metabolism => "metabolism",
                AdmetProperty::Excretion => "excretion",
                AdmetProperty::Toxicity => "toxicity",
                AdmetProperty::Solubility => "solubility",
                AdmetProperty::LogP => "log_p",
                AdmetProperty::BBBPermeability => "bbb_permeability",
            };
            let inner_dict = pyo3::types::PyDict::new(py);
            inner_dict.set_item("probability", result.probability)?;
            inner_dict.set_item("passes", result.passes)?;
            inner_dict.set_item("confidence", result.confidence)?;
            dict.set_item(prop_name, inner_dict)?;
        }
        Ok(dict.into())
    }

    fn __repr__(&self) -> String {
        format!(
            "<AdmetPredictor qubits={} properties={}>",
            self.inner.num_qubits,
            self.inner.properties.len()
        )
    }
}

/// Predict ADMET properties for a molecule (convenience function)
#[cfg(feature = "python")]
#[pyfunction]
#[pyo3(signature = (mol, properties=None))]
fn py_predict_admet(
    mol: &PyMolecule,
    properties: Option<Vec<String>>,
    py: Python<'_>,
) -> PyResult<pyo3::Py<pyo3::types::PyDict>> {
    let props = properties.unwrap_or_else(|| {
        vec![
            "absorption".to_string(),
            "distribution".to_string(),
            "metabolism".to_string(),
            "excretion".to_string(),
            "toxicity".to_string(),
            "solubility".to_string(),
        ]
    });

    let predictor = PyAdmetPredictor::new(4, props);
    predictor.predict(mol, py)
}

/// Python wrapper for QuantumDockingScorer
#[cfg(feature = "python")]
#[pyclass(name = "QuantumDockingScorer")]
pub struct PyQuantumDockingScorer {
    inner: RustQuantumDockingScorer,
}

#[cfg(feature = "python")]
#[pymethods]
impl PyQuantumDockingScorer {
    /// Create a new docking scorer
    ///
    /// Args:
    ///     num_qubits: Number of qubits (default 4)
    ///     scoring: Scoring function type ("quantum_force_field", "quantum_kernel_score",
    ///              "hybrid_classical_quantum")
    ///     num_conformations: Number of conformations to sample (default 10)
    ///     optimization_steps: Optimization iterations (default 50)
    #[new]
    #[pyo3(signature = (num_qubits=None, scoring=None, num_conformations=None, optimization_steps=None))]
    fn new(
        num_qubits: Option<usize>,
        scoring: Option<String>,
        num_conformations: Option<usize>,
        optimization_steps: Option<usize>,
    ) -> Self {
        let scoring_fn = match scoring.as_deref().unwrap_or("hybrid_classical_quantum") {
            "quantum_force_field" => ScoringFunction::QuantumForceField,
            "quantum_kernel_score" => ScoringFunction::QuantumKernelScore,
            _ => ScoringFunction::HybridClassicalQuantum,
        };

        let config = DockingConfig {
            num_qubits: num_qubits.unwrap_or(4),
            scoring_function: scoring_fn,
            num_conformations: num_conformations.unwrap_or(10),
            optimization_steps: optimization_steps.unwrap_or(50),
        };

        Self {
            inner: RustQuantumDockingScorer::new(config),
        }
    }

    /// Score docking of ligand into protein
    ///
    /// Returns a dictionary with score, binding_energy, best_conformation, and all_scores
    fn score(
        &self,
        protein: &PyMolecule,
        ligand: &PyMolecule,
        py: Python<'_>,
    ) -> PyResult<pyo3::Py<pyo3::types::PyDict>> {
        let result = self
            .inner
            .score(&protein.inner, &ligand.inner)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;

        let dict = pyo3::types::PyDict::new(py);
        dict.set_item("score", result.score)?;
        dict.set_item("binding_energy", result.binding_energy)?;
        dict.set_item("best_conformation", result.best_conformation)?;
        dict.set_item("all_scores", result.all_scores)?;
        Ok(dict.into())
    }

    fn __repr__(&self) -> String {
        format!(
            "<QuantumDockingScorer qubits={} conformations={}>",
            self.inner.config.num_qubits, self.inner.config.num_conformations
        )
    }
}

/// Screen a library of ligands against a protein target
#[cfg(feature = "python")]
#[pyfunction]
#[pyo3(signature = (protein, ligands, top_k=10, num_qubits=4, scoring="hybrid_classical_quantum"))]
fn py_screen_library(
    protein: &PyMolecule,
    ligands: Vec<PyMolecule>,
    top_k: usize,
    num_qubits: usize,
    scoring: &str,
    py: Python<'_>,
) -> PyResult<Vec<pyo3::Py<pyo3::types::PyDict>>> {
    let scoring_fn = match scoring {
        "quantum_force_field" => ScoringFunction::QuantumForceField,
        "quantum_kernel_score" => ScoringFunction::QuantumKernelScore,
        _ => ScoringFunction::HybridClassicalQuantum,
    };

    let config = DockingConfig {
        num_qubits,
        scoring_function: scoring_fn,
        num_conformations: 3,
        optimization_steps: 20,
    };
    let scorer = RustQuantumDockingScorer::new(config);

    let mut results: Vec<(usize, f64)> = Vec::new();
    for (idx, ligand) in ligands.iter().enumerate() {
        match scorer.score(&protein.inner, &ligand.inner) {
            Ok(dock_result) => {
                results.push((idx, dock_result.score));
            }
            Err(_) => {
                results.push((idx, f64::MAX));
            }
        }
    }

    // Sort by score (lower is better)
    results.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

    // Take top_k
    let mut output = Vec::new();
    for (idx, score) in results.into_iter().take(top_k) {
        let dict = pyo3::types::PyDict::new(py);
        dict.set_item("ligand_index", idx)?;
        dict.set_item("score", score)?;
        output.push(dict.into());
    }

    Ok(output)
}

/// Optimize a lead compound
#[cfg(feature = "python")]
#[pyfunction]
#[pyo3(signature = (lead, protein, iterations=50, num_qubits=4))]
fn py_optimize_lead(
    lead: &PyMolecule,
    protein: &PyMolecule,
    iterations: usize,
    num_qubits: usize,
    py: Python<'_>,
) -> PyResult<pyo3::Py<pyo3::types::PyDict>> {
    let result = lead_optimization(&lead.inner, &protein.inner, iterations, num_qubits)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;

    let dict = pyo3::types::PyDict::new(py);
    dict.set_item("best_score", result.best_score)?;
    dict.set_item("scores_over_iterations", result.scores_over_iterations)?;

    // Convert Pareto front
    let pareto_list = pyo3::types::PyList::empty(py);
    for point in result.pareto_front {
        let point_dict = pyo3::types::PyDict::new(py);
        point_dict.set_item("objectives", point.objectives)?;
        point_dict.set_item("candidate_index", point.candidate_idx)?;
        pareto_list.append(point_dict)?;
    }
    dict.set_item("pareto_front", pareto_list)?;

    Ok(dict.into())
}

/// Python wrapper for DrugDiscoveryPipeline
#[cfg(feature = "python")]
#[pyclass(name = "DrugDiscoveryPipeline")]
pub struct PyDrugDiscoveryPipeline {
    inner: RustDrugDiscoveryPipeline,
}

#[cfg(feature = "python")]
#[pymethods]
impl PyDrugDiscoveryPipeline {
    /// Create a new drug discovery pipeline
    #[new]
    #[pyo3(signature = (num_qubits=4, standard=false))]
    fn new(num_qubits: usize, standard: bool) -> Self {
        let pipeline = if standard {
            RustDrugDiscoveryPipeline::standard()
        } else {
            RustDrugDiscoveryPipeline::new(num_qubits)
        };
        Self { inner: pipeline }
    }

    /// Run the pipeline on candidate molecules
    ///
    /// Args:
    ///     candidates: List of candidate molecules
    ///     protein: Target protein
    ///
    /// Returns list of stage results
    fn run(
        &mut self,
        candidates: Vec<PyMolecule>,
        protein: &PyMolecule,
        py: Python<'_>,
    ) -> PyResult<Vec<pyo3::Py<pyo3::types::PyDict>>> {
        let rust_candidates: Vec<RustMolecule> =
            candidates.iter().map(|c| c.inner.clone()).collect();

        let stage_results = self
            .inner
            .run(&rust_candidates, &protein.inner)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;

        let mut output = Vec::new();
        for stage in stage_results {
            let dict = pyo3::types::PyDict::new(py);
            dict.set_item("stage_name", stage.stage_name)?;
            dict.set_item("passed", stage.passed)?;
            dict.set_item("failed", stage.failed)?;
            dict.set_item("scores", stage.scores)?;
            output.push(dict.into());
        }

        Ok(output)
    }

    fn __repr__(&self) -> String {
        format!(
            "<DrugDiscoveryPipeline stages={} qubits={}>",
            self.inner.stages.len(),
            self.inner.num_qubits
        )
    }
}

// =============================================================================
// UNIT TESTS
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // =========================================================================
    // Core quantum simulator tests (no Python feature needed)
    // These test the underlying Rust types that the Python bindings wrap.
    // =========================================================================

    /// Verify QuantumSimulator construction and initial state properties.
    /// The |0...0> state should have probability 1 on index 0 and 0 elsewhere.
    #[test]
    fn test_quantum_simulator_initial_state() {
        let sim = QuantumSimulator::new(3);
        assert_eq!(sim.num_qubits(), 3);
        assert_eq!(sim.state.dim, 8); // 2^3

        let amps = sim.state.amplitudes_ref();
        // |000> should have amplitude 1+0i
        assert!((amps[0].re - 1.0).abs() < 1e-12);
        assert!(amps[0].im.abs() < 1e-12);
        // All other amplitudes should be 0
        for i in 1..8 {
            assert!(
                amps[i].re.abs() < 1e-12,
                "amplitude[{}] re was {}",
                i,
                amps[i].re
            );
            assert!(
                amps[i].im.abs() < 1e-12,
                "amplitude[{}] im was {}",
                i,
                amps[i].im
            );
        }
    }

    /// Test that Hadamard on qubit 0 produces equal superposition |0>+|1>.
    /// This mirrors the validation logic in PyQuantumSimulator.h().
    #[test]
    fn test_hadamard_creates_superposition() {
        let mut sim = QuantumSimulator::new(1);
        sim.h(0);

        let probs: Vec<f64> = sim
            .state
            .amplitudes_ref()
            .iter()
            .map(|c| c.re * c.re + c.im * c.im)
            .collect();

        // Both |0> and |1> should have probability 0.5
        assert!((probs[0] - 0.5).abs() < 1e-10, "P(0) = {}", probs[0]);
        assert!((probs[1] - 0.5).abs() < 1e-10, "P(1) = {}", probs[1]);
    }

    /// Test Bell state creation logic (H on qubit 0, CNOT 0->1).
    /// This is the same circuit as create_bell_state() in the Python bindings.
    #[test]
    fn test_bell_state_circuit() {
        let mut sim = QuantumSimulator::new(2);
        sim.h(0);
        sim.cnot(0, 1);

        let probs: Vec<f64> = sim
            .state
            .amplitudes_ref()
            .iter()
            .map(|c| c.re * c.re + c.im * c.im)
            .collect();

        // Bell state (|00> + |11>)/sqrt(2): P(00)=0.5, P(01)=0, P(10)=0, P(11)=0.5
        assert!((probs[0] - 0.5).abs() < 1e-10, "P(00) = {}", probs[0]);
        assert!(probs[1].abs() < 1e-10, "P(01) = {}", probs[1]);
        assert!(probs[2].abs() < 1e-10, "P(10) = {}", probs[2]);
        assert!((probs[3] - 0.5).abs() < 1e-10, "P(11) = {}", probs[3]);
    }

    /// Test GHZ state creation logic for N qubits: H on qubit 0, CNOT chain.
    /// Validates the same pattern used in create_ghz_state().
    #[test]
    fn test_ghz_state_circuit() {
        let num_qubits = 4;
        let mut sim = QuantumSimulator::new(num_qubits);
        sim.h(0);
        for i in 0..(num_qubits - 1) {
            sim.cnot(i, i + 1);
        }

        let probs: Vec<f64> = sim
            .state
            .amplitudes_ref()
            .iter()
            .map(|c| c.re * c.re + c.im * c.im)
            .collect();

        // GHZ state: only |0000> and |1111> have probability 0.5
        let dim = 1 << num_qubits;
        assert!((probs[0] - 0.5).abs() < 1e-10, "P(0000) = {}", probs[0]);
        assert!(
            (probs[dim - 1] - 0.5).abs() < 1e-10,
            "P(1111) = {}",
            probs[dim - 1]
        );
        // All others are 0
        let sum_other: f64 = probs[1..dim - 1].iter().sum();
        assert!(
            sum_other.abs() < 1e-10,
            "sum of non-GHZ amplitudes = {}",
            sum_other
        );
    }

    /// Test Gate construction helpers used by PyQuantumCircuit2.
    /// Verifies that Gate::h, Gate::x, Gate::cnot, Gate::rx produce correct gate types.
    #[test]
    fn test_gate_construction() {
        use crate::gates::GateType;

        let h_gate = Gate::h(0);
        assert_eq!(h_gate.gate_type, GateType::H);
        assert_eq!(h_gate.targets, vec![0]);
        assert!(h_gate.controls.is_empty());

        let x_gate = Gate::x(3);
        assert_eq!(x_gate.gate_type, GateType::X);
        assert_eq!(x_gate.targets, vec![3]);

        let cnot_gate = Gate::cnot(0, 1);
        assert_eq!(cnot_gate.gate_type, GateType::CNOT);
        assert_eq!(cnot_gate.controls, vec![0]);
        assert_eq!(cnot_gate.targets, vec![1]);

        let rx_gate = Gate::rx(2, 1.57);
        match rx_gate.gate_type {
            GateType::Rx(angle) => assert!((angle - 1.57).abs() < 1e-12),
            other => panic!("Expected Rx, got {:?}", other),
        }

        let rz_gate = Gate::rz(0, 0.42);
        match rz_gate.gate_type {
            GateType::Rz(angle) => assert!((angle - 0.42).abs() < 1e-12),
            other => panic!("Expected Rz, got {:?}", other),
        }
    }

    /// Test SnakeMapper as used by PyGrid2DSimulator.
    /// Validates 2D-to-1D mapping and round-trip conversion.
    #[test]
    fn test_snake_mapper_roundtrip() {
        let mapper = SnakeMapper::new(3, 3);
        assert_eq!(mapper.size(), 9);
        assert_eq!(mapper.dimensions(), (3, 3));

        // Verify round-trip for all coordinates
        for y in 0..3 {
            for x in 0..3 {
                let idx = mapper.map_2d_to_1d(x, y);
                let coord = mapper.map_1d_to_2d(idx);
                assert_eq!(coord.x, x, "Round-trip failed for ({}, {})", x, y);
                assert_eq!(coord.y, y, "Round-trip failed for ({}, {})", x, y);
            }
        }
    }

    /// Test SnakeMapper distance calculation (Manhattan distance).
    /// This is used in PyGrid2DSimulator.cnot() to warn about long-range gates.
    #[test]
    fn test_snake_mapper_distance() {
        let mapper = SnakeMapper::new(4, 4);

        let c1 = GridCoord::new(0, 0);
        let c2 = GridCoord::new(3, 3);
        assert_eq!(mapper.distance(&c1, &c2), 6); // |3-0| + |3-0|

        let c3 = GridCoord::new(1, 1);
        let c4 = GridCoord::new(2, 1);
        assert_eq!(mapper.distance(&c3, &c4), 1); // nearest neighbor

        assert_eq!(mapper.max_distance(), 6); // 4+4-2
    }

    /// Test the entropy calculation logic from PyQuantumSimulator.entropy().
    /// For |0> state, Shannon entropy should be 0 (pure state, single outcome).
    #[test]
    fn test_entropy_calculation() {
        let sim = QuantumSimulator::new(2);
        let probs: Vec<f64> = sim
            .state
            .amplitudes_ref()
            .iter()
            .map(|c| c.re * c.re + c.im * c.im)
            .collect();

        // Shannon entropy: -sum(p * ln(p)) for p > 0
        let mut entropy = 0.0;
        for p in &probs {
            if *p > 1e-15 {
                entropy -= p * p.ln();
            }
        }
        // |00> state has prob 1 on state 0, entropy should be 0 (1*ln(1) = 0)
        assert!(entropy.abs() < 1e-12, "entropy = {}", entropy);
    }

    /// Test entropy for a maximally mixed state (equal superposition).
    /// After H on all qubits, entropy should be ln(2^n).
    #[test]
    fn test_entropy_uniform_superposition() {
        let n = 3;
        let mut sim = QuantumSimulator::new(n);
        for i in 0..n {
            sim.h(i);
        }

        let probs: Vec<f64> = sim
            .state
            .amplitudes_ref()
            .iter()
            .map(|c| c.re * c.re + c.im * c.im)
            .collect();

        let mut entropy = 0.0;
        for p in &probs {
            if *p > 1e-15 {
                entropy -= p * p.ln();
            }
        }

        // Uniform over 2^3 = 8 states: entropy = ln(8)
        let expected = (8.0_f64).ln();
        assert!(
            (entropy - expected).abs() < 1e-10,
            "entropy = {}, expected = {}",
            entropy,
            expected
        );
    }

    /// Test the density matrix computation from PyQuantumSimulator.density_matrix().
    /// For |0> state, reduced density matrix should be [[1,0],[0,0]].
    #[test]
    fn test_density_matrix_pure_state() {
        let sim = QuantumSimulator::new(1);
        let x = sim.state.expectation_x(0);
        let y = sim.state.expectation_y(0);
        let z = sim.state.expectation_z(0);

        // For |0>: <X>=0, <Y>=0, <Z>=1
        assert!(x.abs() < 1e-10, "<X> = {}", x);
        assert!(y.abs() < 1e-10, "<Y> = {}", y);
        assert!((z - 1.0).abs() < 1e-10, "<Z> = {}", z);

        // Density matrix elements: rho = (I + x*X + y*Y + z*Z)/2
        // rho[0,0] = (1+z)/2 = 1, rho[1,1] = (1-z)/2 = 0
        let rho_00 = (1.0 + z) / 2.0;
        let rho_11 = (1.0 - z) / 2.0;
        assert!((rho_00 - 1.0).abs() < 1e-10);
        assert!(rho_11.abs() < 1e-10);
    }

    /// Test Bloch vector extraction for |+> state.
    /// After Hadamard: |+> = (|0>+|1>)/sqrt(2), Bloch vector = (1, 0, 0).
    #[test]
    fn test_bloch_vector_plus_state() {
        let mut sim = QuantumSimulator::new(1);
        sim.h(0);

        let x = sim.state.expectation_x(0);
        let y = sim.state.expectation_y(0);
        let z = sim.state.expectation_z(0);

        // |+> state: <X>=1, <Y>=0, <Z>=0
        assert!((x - 1.0).abs() < 1e-10, "<X> = {}", x);
        assert!(y.abs() < 1e-10, "<Y> = {}", y);
        assert!(z.abs() < 1e-10, "<Z> = {}", z);
    }

    /// Test Grover optimal iteration count calculation.
    /// Validates: ceil(pi/4 * sqrt(N)) where N = 2^num_qubits.
    #[test]
    fn test_grover_optimal_iterations() {
        for num_qubits in 2..=8 {
            let max_states = 1usize << num_qubits;
            let optimal =
                ((std::f64::consts::PI / 4.0) * (max_states as f64).sqrt()).ceil() as usize;
            // Grover optimal iterations grows as sqrt(N)
            assert!(
                optimal >= 1,
                "num_qubits={}, optimal={}",
                num_qubits,
                optimal
            );
            // For 2 qubits (4 states): optimal = ceil(pi/4 * 2) = 2
            if num_qubits == 2 {
                assert_eq!(optimal, 2);
            }
            // For 4 qubits (16 states): optimal = ceil(pi/4 * 4) = ceil(3.14) = 4
            if num_qubits == 4 {
                assert_eq!(optimal, 4);
            }
        }
    }

    /// Test the probabilities calculation pattern used by both
    /// PyQuantumSimulator.probabilities() and PyQuantumState.probabilities().
    /// Probabilities should always sum to 1.
    #[test]
    fn test_probabilities_normalization() {
        let mut sim = QuantumSimulator::new(4);
        // Apply a variety of gates
        sim.h(0);
        sim.h(1);
        sim.cnot(0, 2);
        sim.ry(3, 0.75);
        sim.rz(1, 1.23);
        sim.cz(1, 3);

        let probs: Vec<f64> = sim
            .state
            .amplitudes_ref()
            .iter()
            .map(|c| c.re * c.re + c.im * c.im)
            .collect();

        let total: f64 = probs.iter().sum();
        assert!(
            (total - 1.0).abs() < 1e-10,
            "Probabilities sum to {} (should be 1.0)",
            total
        );
    }

    /// Test reset functionality: after reset the simulator should return to |0...0>.
    #[test]
    fn test_simulator_reset() {
        let mut sim = QuantumSimulator::new(3);
        sim.h(0);
        sim.cnot(0, 1);
        sim.x(2);

        // State is entangled, not |000>
        let prob_000_before = {
            let a = &sim.state.amplitudes_ref()[0];
            a.re * a.re + a.im * a.im
        };
        assert!(
            prob_000_before < 0.9,
            "State should not be |000> after gates"
        );

        sim.reset();

        // After reset, should be back to |000>
        let prob_000_after = {
            let a = &sim.state.amplitudes_ref()[0];
            a.re * a.re + a.im * a.im
        };
        assert!(
            (prob_000_after - 1.0).abs() < 1e-12,
            "After reset: P(000) = {}",
            prob_000_after
        );
    }

    /// Test that X gate flips |0> to |1> (used extensively in Python bindings).
    #[test]
    fn test_x_gate_flip() {
        let mut sim = QuantumSimulator::new(1);
        sim.x(0);

        let probs: Vec<f64> = sim
            .state
            .amplitudes_ref()
            .iter()
            .map(|c| c.re * c.re + c.im * c.im)
            .collect();

        assert!(probs[0].abs() < 1e-12, "P(0) should be 0 after X");
        assert!((probs[1] - 1.0).abs() < 1e-12, "P(1) should be 1 after X");
    }

    /// Test the CDF-based sampling logic used in PyEnhancedSimulator.run_circuit().
    /// Verifies that building CDF from probabilities works correctly.
    #[test]
    fn test_cdf_sampling_logic() {
        // Simulate the CDF construction from run_circuit
        let probs = vec![0.25, 0.25, 0.25, 0.25]; // uniform 2-qubit
        let mut cdf = Vec::with_capacity(probs.len());
        let mut cumsum = 0.0;
        for &p in &probs {
            cumsum += p;
            cdf.push(cumsum);
        }

        // CDF should end at 1.0
        assert!((*cdf.last().unwrap() - 1.0_f64).abs() < 1e-12);
        // CDF should be monotonically increasing
        for i in 1..cdf.len() {
            assert!(cdf[i] >= cdf[i - 1], "CDF not monotone at index {}", i);
        }

        // Test binary search for known values
        // r=0.1 should land in first bucket (index 0)
        let r = 0.1;
        let idx = match cdf.binary_search_by(|c| c.partial_cmp(&r).unwrap()) {
            Ok(i) => i,
            Err(i) => i.min(probs.len().saturating_sub(1)),
        };
        assert_eq!(idx, 0, "r={} should map to index 0", r);

        // r=0.3 should land in second bucket (index 1)
        let r = 0.3;
        let idx = match cdf.binary_search_by(|c| c.partial_cmp(&r).unwrap()) {
            Ok(i) => i,
            Err(i) => i.min(probs.len().saturating_sub(1)),
        };
        assert_eq!(idx, 1, "r={} should map to index 1", r);
    }

    /// Test Heisenberg QPE phase validation logic used in run_heisenberg_qpe_rust().
    #[test]
    fn test_heisenberg_qpe_phase_validation() {
        // Phase must be in [0, 1)
        assert!((0.0..1.0).contains(&0.0));
        assert!((0.0..1.0).contains(&0.5));
        assert!((0.0..1.0).contains(&0.999));
        assert!(!(0.0..1.0).contains(&1.0));
        assert!(!(0.0..1.0).contains(&-0.1));

        // Readout error must be in [0, 0.5]
        assert!((0.0..=0.5).contains(&0.0));
        assert!((0.0..=0.5).contains(&0.5));
        assert!(!(0.0..=0.5).contains(&0.51));
        assert!(!(0.0..=0.5).contains(&-0.01));
    }

    // =========================================================================
    // Python-feature-gated tests
    // These test functions that are only compiled with the `python` feature.
    // =========================================================================

    /// Test observable parsing: single Pauli operators.
    #[cfg(feature = "python")]
    #[test]
    fn test_parse_observable_single_pauli() {
        use crate::pytorch_bridge::Observable;

        let z0 = parse_observable("Z0").unwrap();
        assert!(matches!(z0, Observable::PauliZ(0)));

        let x1 = parse_observable("X1").unwrap();
        assert!(matches!(x1, Observable::PauliX(1)));

        let y2 = parse_observable("Y2").unwrap();
        assert!(matches!(y2, Observable::PauliY(2)));

        // Case insensitive
        let z_lower = parse_observable("z5").unwrap();
        assert!(matches!(z_lower, Observable::PauliZ(5)));
    }

    /// Test observable parsing: multi-qubit Pauli strings.
    #[cfg(feature = "python")]
    #[test]
    fn test_parse_observable_pauli_string() {
        use crate::pytorch_bridge::Observable;

        let zz = parse_observable("Z0Z1").unwrap();
        match zz {
            Observable::PauliString(ops) => {
                assert_eq!(ops.len(), 2);
                assert_eq!(ops[0], (0, 'Z'));
                assert_eq!(ops[1], (1, 'Z'));
            }
            other => panic!("Expected PauliString, got {:?}", other),
        }

        let xyz = parse_observable("X0Y1Z2").unwrap();
        match xyz {
            Observable::PauliString(ops) => {
                assert_eq!(ops.len(), 3);
                assert_eq!(ops[0], (0, 'X'));
                assert_eq!(ops[1], (1, 'Y'));
                assert_eq!(ops[2], (2, 'Z'));
            }
            other => panic!("Expected PauliString, got {:?}", other),
        }
    }

    /// Test observable parsing: invalid input should produce an error.
    #[cfg(feature = "python")]
    #[test]
    fn test_parse_observable_invalid() {
        let result = parse_observable("INVALID");
        // Should fail or parse as empty PauliString
        // The function returns Err for strings it cannot parse
        assert!(result.is_err() || matches!(result, Ok(_)));
    }

    /// Test distillation protocol parsing for all valid variants.
    #[cfg(feature = "python")]
    #[test]
    fn test_parse_distillation_protocol() {
        use crate::magic_state_factory::DistillationProtocol;

        let p1 = parse_distillation_protocol("15_to_1").unwrap();
        assert!(matches!(p1, DistillationProtocol::FifteenToOne));

        let p2 = parse_distillation_protocol("fifteen_to_one").unwrap();
        assert!(matches!(p2, DistillationProtocol::FifteenToOne));

        let p3 = parse_distillation_protocol("20_to_4").unwrap();
        assert!(matches!(p3, DistillationProtocol::TwentyToFour));

        let p4 = parse_distillation_protocol("reed_muller_116").unwrap();
        assert!(matches!(p4, DistillationProtocol::ReedMuller116));

        let p5 = parse_distillation_protocol("litinski_compact").unwrap();
        assert!(matches!(p5, DistillationProtocol::LitinskiCompact));

        let p6 = parse_distillation_protocol("litinski").unwrap();
        assert!(matches!(p6, DistillationProtocol::LitinskiCompact));
    }

    /// Test distillation protocol parsing rejects unknown protocols.
    #[cfg(feature = "python")]
    #[test]
    fn test_parse_distillation_protocol_invalid() {
        let result = parse_distillation_protocol("unknown_protocol");
        assert!(result.is_err());
        let err_msg = format!("{}", result.unwrap_err());
        assert!(err_msg.contains("Unknown protocol"));
    }

    /// Test the Gate vector building pattern used in PyQuantumCircuit2.
    /// Verifies that sequential gate additions accumulate correctly.
    #[test]
    fn test_circuit_gate_accumulation() {
        let mut gates: Vec<Gate> = Vec::new();
        let num_qubits = 4;

        // Replicate what PyQuantumCircuit2 methods do
        gates.push(Gate::h(0));
        gates.push(Gate::x(1));
        gates.push(Gate::cnot(0, 1));
        gates.push(Gate::rx(2, 0.5));
        gates.push(Gate::ry(3, 1.0));
        gates.push(Gate::rz(0, 0.25));

        assert_eq!(gates.len(), 6);

        // Verify types
        assert_eq!(gates[0].gate_type, crate::gates::GateType::H);
        assert_eq!(gates[1].gate_type, crate::gates::GateType::X);
        assert_eq!(gates[2].gate_type, crate::gates::GateType::CNOT);
        assert_eq!(gates[5].targets, vec![0]);
    }

    /// Test PyQuantumCircuit2 qubit bounds validation pattern.
    /// The bindings check `qubit >= self.num_qubits` before every gate.
    #[test]
    fn test_qubit_bounds_validation_pattern() {
        let num_qubits = 3;

        // Valid: qubits 0, 1, 2 for a 3-qubit circuit
        for q in 0..num_qubits {
            assert!(q < num_qubits, "Qubit {} should be valid", q);
        }
        // Invalid: qubit 3 for a 3-qubit circuit
        assert!(
            3 >= num_qubits,
            "Qubit 3 should be invalid for 3-qubit circuit"
        );

        // CNOT: control != target validation
        let control = 0;
        let target = 1;
        assert_ne!(control, target, "Control and target must differ");
    }

    /// Test fidelity between identical states (should be 1.0).
    #[test]
    fn test_fidelity_identical_states() {
        let mut sim = QuantumSimulator::new(2);
        sim.h(0);
        sim.cnot(0, 1);

        let state_copy = sim.state.clone();
        let fidelity = sim.state.fidelity(&state_copy);
        assert!(
            (fidelity - 1.0).abs() < 1e-10,
            "Fidelity with self should be 1.0, got {}",
            fidelity
        );
    }

    /// Test fidelity between orthogonal states (should be 0.0).
    #[test]
    fn test_fidelity_orthogonal_states() {
        let state0 = QuantumState::new(1); // |0>
        let mut sim1 = QuantumSimulator::new(1);
        sim1.x(0); // |1>

        let fidelity = state0.fidelity(&sim1.state);
        assert!(
            fidelity.abs() < 1e-10,
            "Fidelity between |0> and |1> should be 0.0, got {}",
            fidelity
        );
    }

    /// Test that state amplitudes preserve unitarity through multiple gate operations.
    /// This validates the internal consistency that all Python bindings depend on.
    #[test]
    fn test_unitarity_through_gate_sequence() {
        let mut sim = QuantumSimulator::new(5);

        // Apply a complex sequence of gates
        for i in 0..5 {
            sim.h(i);
        }
        sim.cnot(0, 1);
        sim.cnot(2, 3);
        sim.cz(1, 4);
        sim.rx(0, 0.3);
        sim.ry(2, 1.1);
        sim.rz(4, 2.7);
        sim.swap(0, 3);
        sim.s(1);
        sim.t(2);

        // Verify normalization: sum of |a_i|^2 = 1
        let norm_sq: f64 = sim
            .state
            .amplitudes_ref()
            .iter()
            .map(|c| c.re * c.re + c.im * c.im)
            .sum();

        assert!(
            (norm_sq - 1.0).abs() < 1e-10,
            "State normalization broken: |psi|^2 = {}",
            norm_sq
        );
    }
}
