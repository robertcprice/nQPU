//! Distributed gradients for variational circuits.
//!
//! Supports:
//! - Parameter-shift over distributed world execution (works for multi-rank).
//! - Adjoint-mode gradients when `world_size == 1`.
//! - Auto strategy selection.

use crate::adjoint_diff::{AdjointCircuit, AdjointOp, Observable};
use crate::distributed_metal_mpi::{
    CommunicationCostModel, DistributedMetalConfig, DistributedMetalMetrics,
    DistributedMetalWorldExecutor,
};
use crate::gates::Gate;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DistributedGradientMethod {
    /// Always use distributed parameter-shift.
    ParameterShift,
    /// Use adjoint-mode gradient. Requires `world_size == 1`.
    Adjoint,
    /// Use mirrored adjoint gradient for `world_size > 1`.
    ///
    /// This runs distributed world execution for expectation/metrics and
    /// evaluates gradients with the full-state adjoint engine.
    AdjointMirror,
    /// Use adjoint for `world_size == 1`, otherwise mirrored adjoint.
    Auto,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DistributedGradientMethodUsed {
    ParameterShift,
    Adjoint,
    AdjointMirror,
}

#[derive(Clone, Debug)]
pub struct DistributedAdjointConfig {
    pub world_size: usize,
    pub distributed_config: DistributedMetalConfig,
    pub cost_model: CommunicationCostModel,
    /// Parameter-shift amount. For Pauli rotations, `pi/2` yields exact gradients.
    pub shift: f64,
    pub method: DistributedGradientMethod,
    /// If `true`, `Adjoint` mode falls back to parameter-shift when `world_size > 1`.
    pub allow_parameter_shift_fallback: bool,
}

impl Default for DistributedAdjointConfig {
    fn default() -> Self {
        Self {
            world_size: 1,
            distributed_config: DistributedMetalConfig::default(),
            cost_model: CommunicationCostModel::default(),
            shift: std::f64::consts::FRAC_PI_2,
            method: DistributedGradientMethod::Auto,
            allow_parameter_shift_fallback: true,
        }
    }
}

#[derive(Clone, Debug)]
pub struct DistributedAdjointResult {
    pub expectation: f64,
    pub gradients: Vec<f64>,
    pub num_evaluations: usize,
    pub last_metrics: DistributedMetalMetrics,
    pub method_used: DistributedGradientMethodUsed,
    pub fallback_reason: Option<String>,
}

fn infer_num_params(circuit: &AdjointCircuit) -> usize {
    circuit
        .ops
        .iter()
        .filter_map(|op| match op {
            AdjointOp::Rx { param, .. } => Some(*param),
            AdjointOp::Ry { param, .. } => Some(*param),
            AdjointOp::Rz { param, .. } => Some(*param),
            AdjointOp::Fixed(_) => None,
        })
        .max()
        .map(|i| i + 1)
        .unwrap_or(0)
}

fn validate_num_params(circuit: &AdjointCircuit, params: &[f64]) -> Result<usize, String> {
    let num_params = infer_num_params(circuit);
    if params.len() < num_params {
        return Err(format!(
            "insufficient params: got {}, expected at least {}",
            params.len(),
            num_params
        ));
    }
    Ok(num_params)
}

fn compile_gates(circuit: &AdjointCircuit, params: &[f64]) -> Result<Vec<Gate>, String> {
    let mut gates = Vec::with_capacity(circuit.ops.len());
    for op in &circuit.ops {
        match op {
            AdjointOp::Fixed(g) => gates.push(g.clone()),
            AdjointOp::Rx { qubit, param } => {
                let theta = params
                    .get(*param)
                    .ok_or_else(|| format!("missing parameter {} for Rx", param))?;
                gates.push(Gate::rx(*qubit, *theta));
            }
            AdjointOp::Ry { qubit, param } => {
                let theta = params
                    .get(*param)
                    .ok_or_else(|| format!("missing parameter {} for Ry", param))?;
                gates.push(Gate::ry(*qubit, *theta));
            }
            AdjointOp::Rz { qubit, param } => {
                let theta = params
                    .get(*param)
                    .ok_or_else(|| format!("missing parameter {} for Rz", param))?;
                gates.push(Gate::rz(*qubit, *theta));
            }
        }
    }
    Ok(gates)
}

fn expectation_from_probabilities(probs: &[f64], obs: Observable) -> f64 {
    match obs {
        Observable::PauliZ(q) => {
            let mask = 1usize << q;
            probs
                .iter()
                .enumerate()
                .map(|(i, p)| if (i & mask) == 0 { *p } else { -*p })
                .sum()
        }
    }
}

fn distributed_parameter_shift_impl(
    circuit: &AdjointCircuit,
    params: &[f64],
    obs: Observable,
    cfg: &DistributedAdjointConfig,
    fallback_reason: Option<String>,
) -> Result<DistributedAdjointResult, String> {
    let num_params = validate_num_params(circuit, params)?;

    let mut world = DistributedMetalWorldExecutor::new(
        circuit.num_qubits,
        cfg.world_size,
        cfg.distributed_config.clone(),
    )?;
    world.set_cost_model(cfg.cost_model.clone());

    let base_gates = compile_gates(circuit, params)?;
    let base = world.execute_partitioned(&base_gates)?;
    let expectation = expectation_from_probabilities(&base.global_probabilities, obs);

    let mut gradients = vec![0.0; num_params];
    let mut last_metrics = base.metrics.clone();
    let mut eval_count = 1usize;

    for p in 0..num_params {
        let mut plus = params.to_vec();
        let mut minus = params.to_vec();
        plus[p] += cfg.shift;
        minus[p] -= cfg.shift;

        let plus_gates = compile_gates(circuit, &plus)?;
        let minus_gates = compile_gates(circuit, &minus)?;

        let plus_out = world.execute_partitioned(&plus_gates)?;
        let minus_out = world.execute_partitioned(&minus_gates)?;

        let e_plus = expectation_from_probabilities(&plus_out.global_probabilities, obs);
        let e_minus = expectation_from_probabilities(&minus_out.global_probabilities, obs);

        gradients[p] = (e_plus - e_minus) / (2.0 * cfg.shift.sin());
        last_metrics = minus_out.metrics;
        eval_count += 2;
    }

    Ok(DistributedAdjointResult {
        expectation,
        gradients,
        num_evaluations: eval_count,
        last_metrics,
        method_used: DistributedGradientMethodUsed::ParameterShift,
        fallback_reason,
    })
}

fn distributed_adjoint_single_rank(
    circuit: &AdjointCircuit,
    params: &[f64],
    obs: Observable,
    cfg: &DistributedAdjointConfig,
) -> Result<DistributedAdjointResult, String> {
    validate_num_params(circuit, params)?;

    // Keep expectation + metrics path on world executor so callers get the
    // same telemetry surface as parameter-shift.
    let mut world = DistributedMetalWorldExecutor::new(
        circuit.num_qubits,
        cfg.world_size,
        cfg.distributed_config.clone(),
    )?;
    world.set_cost_model(cfg.cost_model.clone());
    let base_gates = compile_gates(circuit, params)?;
    let base = world.execute_partitioned(&base_gates)?;

    let expectation = expectation_from_probabilities(&base.global_probabilities, obs);
    let gradients = circuit.gradient(params, obs)?;
    Ok(DistributedAdjointResult {
        expectation,
        gradients,
        num_evaluations: 1,
        last_metrics: base.metrics,
        method_used: DistributedGradientMethodUsed::Adjoint,
        fallback_reason: None,
    })
}

fn distributed_adjoint_mirror_multi_rank(
    circuit: &AdjointCircuit,
    params: &[f64],
    obs: Observable,
    cfg: &DistributedAdjointConfig,
) -> Result<DistributedAdjointResult, String> {
    validate_num_params(circuit, params)?;

    let mut world = DistributedMetalWorldExecutor::new(
        circuit.num_qubits,
        cfg.world_size,
        cfg.distributed_config.clone(),
    )?;
    world.set_cost_model(cfg.cost_model.clone());
    let base_gates = compile_gates(circuit, params)?;
    let base = world.execute_partitioned(&base_gates)?;

    let expectation = expectation_from_probabilities(&base.global_probabilities, obs);
    let gradients = circuit.gradient(params, obs)?;

    Ok(DistributedAdjointResult {
        expectation,
        gradients,
        num_evaluations: 1,
        last_metrics: base.metrics,
        method_used: DistributedGradientMethodUsed::AdjointMirror,
        fallback_reason: None,
    })
}

/// Compute distributed gradients with configurable method selection.
pub fn distributed_gradient(
    circuit: &AdjointCircuit,
    params: &[f64],
    obs: Observable,
    cfg: &DistributedAdjointConfig,
) -> Result<DistributedAdjointResult, String> {
    match cfg.method {
        DistributedGradientMethod::ParameterShift => {
            distributed_parameter_shift_impl(circuit, params, obs, cfg, None)
        }
        DistributedGradientMethod::Adjoint => {
            if cfg.world_size == 1 {
                distributed_adjoint_single_rank(circuit, params, obs, cfg)
            } else if cfg.allow_parameter_shift_fallback {
                distributed_parameter_shift_impl(
                    circuit,
                    params,
                    obs,
                    cfg,
                    Some("adjoint requested with world_size>1; fell back to parameter-shift"
                        .to_string()),
                )
            } else {
                Err("adjoint distributed gradient currently requires world_size == 1 (or use AdjointMirror)".to_string())
            }
        }
        DistributedGradientMethod::AdjointMirror => {
            if cfg.world_size == 1 {
                distributed_adjoint_single_rank(circuit, params, obs, cfg)
            } else {
                distributed_adjoint_mirror_multi_rank(circuit, params, obs, cfg)
            }
        }
        DistributedGradientMethod::Auto => {
            if cfg.world_size == 1 {
                distributed_adjoint_single_rank(circuit, params, obs, cfg)
            } else {
                distributed_adjoint_mirror_multi_rank(circuit, params, obs, cfg)
            }
        }
    }
}

/// Compatibility wrapper that always uses parameter-shift.
pub fn distributed_parameter_shift_gradient(
    circuit: &AdjointCircuit,
    params: &[f64],
    obs: Observable,
    cfg: &DistributedAdjointConfig,
) -> Result<DistributedAdjointResult, String> {
    distributed_parameter_shift_impl(circuit, params, obs, cfg, None)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_distributed_parameter_shift_matches_adjoint_reference_single_rank() {
        let mut c = AdjointCircuit::new(2);
        c.add_op(AdjointOp::Ry { qubit: 0, param: 0 });
        c.add_op(AdjointOp::Fixed(Gate::cnot(0, 1)));
        c.add_op(AdjointOp::Rz { qubit: 1, param: 1 });

        let params = vec![0.31, -0.22];
        let ref_grad = c
            .gradient(&params, Observable::PauliZ(0))
            .expect("adjoint reference");

        let cfg = DistributedAdjointConfig::default();
        let out = distributed_parameter_shift_gradient(&c, &params, Observable::PauliZ(0), &cfg)
            .expect("distributed gradient");

        assert_eq!(out.gradients.len(), ref_grad.len());
        assert_eq!(out.method_used, DistributedGradientMethodUsed::ParameterShift);
        for (a, b) in out.gradients.iter().zip(ref_grad.iter()) {
            assert!((a - b).abs() < 1e-6, "grad mismatch {} vs {}", a, b);
        }
    }

    #[test]
    fn test_distributed_parameter_shift_world_size_two_with_remote_fixed_gate() {
        let mut c = AdjointCircuit::new(3);
        c.add_op(AdjointOp::Fixed(Gate::h(2))); // shard-domain for world_size=2
        c.add_op(AdjointOp::Rx { qubit: 0, param: 0 });
        c.add_op(AdjointOp::Fixed(Gate::cnot(2, 1))); // remote-control reducible path

        let params = vec![0.47];
        let ref_grad = c
            .gradient(&params, Observable::PauliZ(0))
            .expect("adjoint reference");

        let cfg = DistributedAdjointConfig {
            world_size: 2,
            ..DistributedAdjointConfig::default()
        };
        let out = distributed_parameter_shift_gradient(&c, &params, Observable::PauliZ(0), &cfg)
            .expect("distributed gradient");

        assert_eq!(out.gradients.len(), 1);
        assert!((out.gradients[0] - ref_grad[0]).abs() < 1e-6);
        assert!(out.num_evaluations >= 3);
        assert_eq!(out.method_used, DistributedGradientMethodUsed::ParameterShift);
    }

    #[test]
    fn test_distributed_gradient_auto_uses_adjoint_single_rank() {
        let mut c = AdjointCircuit::new(2);
        c.add_op(AdjointOp::Ry { qubit: 0, param: 0 });
        c.add_op(AdjointOp::Fixed(Gate::cnot(0, 1)));
        c.add_op(AdjointOp::Rz { qubit: 1, param: 1 });
        let params = vec![0.3, -0.1];

        let cfg = DistributedAdjointConfig {
            world_size: 1,
            method: DistributedGradientMethod::Auto,
            ..DistributedAdjointConfig::default()
        };
        let out = distributed_gradient(&c, &params, Observable::PauliZ(0), &cfg).expect("run");
        let reference = c
            .gradient(&params, Observable::PauliZ(0))
            .expect("adjoint reference");

        assert_eq!(out.method_used, DistributedGradientMethodUsed::Adjoint);
        assert_eq!(out.num_evaluations, 1);
        assert_eq!(out.gradients.len(), reference.len());
        for (a, b) in out.gradients.iter().zip(reference.iter()) {
            assert!((a - b).abs() < 1e-6);
        }
    }

    #[test]
    fn test_distributed_gradient_adjoint_only_rejects_multi_rank_without_fallback() {
        let mut c = AdjointCircuit::new(2);
        c.add_op(AdjointOp::Rx { qubit: 0, param: 0 });
        let params = vec![0.2];

        let cfg = DistributedAdjointConfig {
            world_size: 2,
            method: DistributedGradientMethod::Adjoint,
            allow_parameter_shift_fallback: false,
            ..DistributedAdjointConfig::default()
        };
        let err = distributed_gradient(&c, &params, Observable::PauliZ(0), &cfg)
            .expect_err("should reject adjoint-only multi-rank");
        assert!(err.contains("world_size == 1"));
    }

    #[test]
    fn test_distributed_gradient_adjoint_only_falls_back_when_enabled() {
        let mut c = AdjointCircuit::new(3);
        c.add_op(AdjointOp::Fixed(Gate::h(2)));
        c.add_op(AdjointOp::Rx { qubit: 0, param: 0 });
        c.add_op(AdjointOp::Fixed(Gate::cnot(2, 1)));
        let params = vec![0.47];

        let cfg = DistributedAdjointConfig {
            world_size: 2,
            method: DistributedGradientMethod::Adjoint,
            allow_parameter_shift_fallback: true,
            ..DistributedAdjointConfig::default()
        };
        let out = distributed_gradient(&c, &params, Observable::PauliZ(0), &cfg).expect("run");
        let ref_grad = c
            .gradient(&params, Observable::PauliZ(0))
            .expect("adjoint reference");

        assert_eq!(out.method_used, DistributedGradientMethodUsed::ParameterShift);
        assert!(out
            .fallback_reason
            .as_ref()
            .map(|s| s.contains("fell back"))
            .unwrap_or(false));
        assert!((out.gradients[0] - ref_grad[0]).abs() < 1e-6);
    }

    #[test]
    fn test_distributed_gradient_auto_uses_adjoint_mirror_multi_rank() {
        let mut c = AdjointCircuit::new(3);
        c.add_op(AdjointOp::Fixed(Gate::h(2))); // remote for world_size=2
        c.add_op(AdjointOp::Ry { qubit: 0, param: 0 });
        c.add_op(AdjointOp::Fixed(Gate::cnot(2, 1))); // remote-control reducible
        c.add_op(AdjointOp::Rz { qubit: 1, param: 1 });
        let params = vec![0.33, -0.27];

        let cfg = DistributedAdjointConfig {
            world_size: 2,
            method: DistributedGradientMethod::Auto,
            ..DistributedAdjointConfig::default()
        };
        let out = distributed_gradient(&c, &params, Observable::PauliZ(1), &cfg).expect("run");
        let reference = c
            .gradient(&params, Observable::PauliZ(1))
            .expect("adjoint reference");

        assert_eq!(out.method_used, DistributedGradientMethodUsed::AdjointMirror);
        assert_eq!(out.num_evaluations, 1);
        assert!(out.fallback_reason.is_none());
        assert_eq!(out.gradients.len(), reference.len());
        for (a, b) in out.gradients.iter().zip(reference.iter()) {
            assert!((a - b).abs() < 1e-6);
        }
        assert!(out.last_metrics.remote_gates >= 2);
    }
}
