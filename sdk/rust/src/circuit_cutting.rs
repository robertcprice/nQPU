//! Circuit cutting and fragment stitching.

use crate::gates::{Gate, GateType};
use crate::{GateOperations, QuantumState};
use std::collections::{BTreeSet, HashMap};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CutKind {
    Gate,
    Wire,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CutPoint {
    pub gate_index: usize,
    pub wire: usize,
    pub kind: CutKind,
}

#[derive(Clone, Debug)]
pub struct CircuitFragment {
    pub gates: Vec<Gate>,
    pub local_qubits: Vec<usize>,
    pub input_wires: Vec<usize>,
    pub output_wires: Vec<usize>,
}

#[derive(Clone, Debug)]
pub struct CutPlan {
    pub cut_points: Vec<CutPoint>,
    pub fragments: Vec<CircuitFragment>,
}

/// Reconstruction strategy for fragment-level estimates.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ReconstructionMode {
    /// Multiply fragment estimates (default separable approximation).
    Product,
    /// Arithmetic mean across fragments.
    Mean,
    /// Variance-weighted mean (requires per-fragment variances).
    WeightedMean,
    /// Quasiprobability-style scaling model (product with sampling overhead model).
    QuasiProbability,
}

/// Basis term used for wire-cut quasiprobability reconstruction.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PauliBasisTerm {
    I,
    X,
    Y,
    Z,
}

/// One quasiprobability term spanning all wire cuts in the plan.
#[derive(Clone, Debug, PartialEq)]
pub struct QuasiProbTerm {
    pub basis: Vec<PauliBasisTerm>,
    pub coefficient: f64,
}

/// Rough sampling-cost model for a cut plan.
#[derive(Clone, Debug, PartialEq)]
pub struct SamplingCostEstimate {
    pub num_fragments: usize,
    pub num_cuts: usize,
    pub fragment_evaluations: usize,
    pub quasiprobability_overhead: f64,
    pub estimated_total_shots: u64,
}

/// Configuration for searching candidate automatic cut plans.
#[derive(Clone, Debug)]
pub struct CutSearchConfig {
    pub max_fragment_qubits_candidates: Vec<usize>,
    pub lookahead_candidates: Vec<usize>,
    pub base_shots_per_fragment: u64,
    pub mode: ReconstructionMode,
    /// Linear penalty per cut.
    pub cut_penalty: f64,
    /// Penalty on largest fragment size (quadratic).
    pub fragment_qubit_penalty: f64,
}

impl Default for CutSearchConfig {
    fn default() -> Self {
        Self {
            max_fragment_qubits_candidates: vec![4, 6, 8],
            lookahead_candidates: vec![8, 16],
            base_shots_per_fragment: 1024,
            mode: ReconstructionMode::QuasiProbability,
            cut_penalty: 10.0,
            fragment_qubit_penalty: 2.0,
        }
    }
}

/// Candidate cut plan with objective score.
#[derive(Clone, Debug)]
pub struct ScoredCutPlan {
    pub plan: CutPlan,
    pub auto_config: AutoCutConfig,
    pub sampling_cost: SamplingCostEstimate,
    pub objective: f64,
}

/// Configuration for automatic cut finding.
#[derive(Clone, Debug)]
pub struct AutoCutConfig {
    /// Maximum active qubits allowed per fragment.
    pub max_fragment_qubits: usize,
    /// Number of future gates to inspect when choosing cut wire.
    pub lookahead_gates: usize,
    /// Prefer wire cuts when the selected wire has significant future activity.
    pub prefer_wire_cuts: bool,
    /// Minimum number of near-future uses required to label a cut as `Wire`.
    pub wire_cut_min_future_uses: usize,
}

impl Default for AutoCutConfig {
    fn default() -> Self {
        Self {
            max_fragment_qubits: 8,
            lookahead_gates: 16,
            prefer_wire_cuts: true,
            wire_cut_min_future_uses: 2,
        }
    }
}

/// Build a cut plan so each fragment stays under `max_fragment_qubits`.
pub fn plan_cuts(gates: &[Gate], max_fragment_qubits: usize) -> CutPlan {
    let mut cuts = Vec::new();
    let mut live = BTreeSet::new();

    for (i, g) in gates.iter().enumerate() {
        for &q in g.targets.iter().chain(g.controls.iter()) {
            live.insert(q);
        }

        if live.len() > max_fragment_qubits && g.num_qubits() >= 2 {
            let wire = g.targets[0];
            cuts.push(CutPoint {
                gate_index: i,
                wire,
                kind: CutKind::Gate,
            });
            live.clear();
            for &q in g.targets.iter().chain(g.controls.iter()) {
                live.insert(q);
            }
        }
    }

    let fragments = build_fragments(gates, &cuts);
    CutPlan {
        cut_points: cuts,
        fragments,
    }
}

/// Automatically find cut points using bounded lookahead.
///
/// Heuristic:
/// - trigger a cut when the active wire set exceeds the configured bound
/// - choose the cut wire that appears least in upcoming gates
pub fn plan_cuts_auto(gates: &[Gate], config: &AutoCutConfig) -> CutPlan {
    if gates.is_empty() {
        return CutPlan {
            cut_points: vec![],
            fragments: vec![],
        };
    }

    let mut cuts = Vec::new();
    let mut live = BTreeSet::new();
    let gate_wires: Vec<Vec<usize>> = gates.iter().map(distinct_gate_wires).collect();
    let lookahead = config.lookahead_gates.max(1);
    let mut window_counts: HashMap<usize, usize> = HashMap::new();

    // For gate index i, the lookahead window is (i, i + lookahead].
    // Initialize for i=0 with gates [1, lookahead].
    let init_end = (1 + lookahead).min(gates.len());
    for wires in gate_wires.iter().take(init_end).skip(1) {
        for &q in wires {
            *window_counts.entry(q).or_insert(0) += 1;
        }
    }

    for (i, g) in gates.iter().enumerate() {
        for &q in g.targets.iter().chain(g.controls.iter()) {
            live.insert(q);
        }

        if live.len() <= config.max_fragment_qubits {
            continue;
        }

        let mut candidates: Vec<usize> =
            g.targets.iter().chain(g.controls.iter()).copied().collect();
        candidates.sort_unstable();
        candidates.dedup();

        if candidates.is_empty() {
            continue;
        }

        let mut best_wire = candidates[0];
        let mut best_score = window_counts.get(&best_wire).copied().unwrap_or(0);

        for &wire in candidates.iter().skip(1) {
            let score = window_counts.get(&wire).copied().unwrap_or(0);
            if score < best_score || (score == best_score && wire < best_wire) {
                best_wire = wire;
                best_score = score;
            }
        }

        let kind = if config.prefer_wire_cuts && best_score >= config.wire_cut_min_future_uses {
            CutKind::Wire
        } else {
            CutKind::Gate
        };

        cuts.push(CutPoint {
            gate_index: i,
            wire: best_wire,
            kind,
        });

        live.clear();
        for &q in g.targets.iter().chain(g.controls.iter()) {
            live.insert(q);
        }

        // Slide the lookahead window for the next gate index.
        if i + 1 < gates.len() {
            let leaving = i + 1;
            for &q in &gate_wires[leaving] {
                if let Some(v) = window_counts.get_mut(&q) {
                    *v = v.saturating_sub(1);
                    if *v == 0 {
                        window_counts.remove(&q);
                    }
                }
            }
            let entering = i + lookahead + 1;
            if entering < gates.len() {
                for &q in &gate_wires[entering] {
                    *window_counts.entry(q).or_insert(0) += 1;
                }
            }
        }
    }

    let fragments = build_fragments(gates, &cuts);
    CutPlan {
        cut_points: cuts,
        fragments,
    }
}

/// Automatically plan cuts with wire-cut preference enabled.
pub fn plan_wire_cuts_auto(
    gates: &[Gate],
    max_fragment_qubits: usize,
    lookahead_gates: usize,
) -> CutPlan {
    let cfg = AutoCutConfig {
        max_fragment_qubits,
        lookahead_gates: lookahead_gates.max(1),
        prefer_wire_cuts: true,
        wire_cut_min_future_uses: 1,
    };
    plan_cuts_auto(gates, &cfg)
}

/// Reconstruct an observable by stitching fragment estimates.
///
/// This uses multiplicative stitching for approximately separable cut channels.
pub fn stitch_fragment_estimates(values: &[f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    values.iter().product()
}

/// Reconstruct a global estimate from fragment estimates.
///
/// For `WeightedMean`, provide per-fragment variances via `variances`.
pub fn reconstruct_from_fragment_estimates(
    values: &[f64],
    variances: Option<&[f64]>,
    mode: ReconstructionMode,
) -> Result<f64, String> {
    if values.is_empty() {
        return Ok(0.0);
    }

    match mode {
        ReconstructionMode::Product => Ok(stitch_fragment_estimates(values)),
        ReconstructionMode::Mean => Ok(values.iter().sum::<f64>() / values.len() as f64),
        ReconstructionMode::WeightedMean => {
            let vars = variances.ok_or_else(|| {
                "weighted reconstruction requires per-fragment variances".to_string()
            })?;
            if vars.len() != values.len() {
                return Err("variance length must match values length".to_string());
            }

            let mut weighted_sum = 0.0;
            let mut weight_sum = 0.0;
            for (&v, &var) in values.iter().zip(vars.iter()) {
                if var.is_sign_negative() {
                    return Err("variance must be non-negative".to_string());
                }
                let w = 1.0 / var.max(1e-12);
                weighted_sum += w * v;
                weight_sum += w;
            }
            Ok(weighted_sum / weight_sum.max(1e-12))
        }
        ReconstructionMode::QuasiProbability => Ok(stitch_fragment_estimates(values)),
    }
}

/// Enumerate quasiprobability terms for wire-cut stitching.
///
/// For `k` wire cuts this returns `4^k` Pauli-basis terms with uniform
/// coefficient `0.5^k` (Bloch-basis channel reconstruction proxy).
pub fn wire_cut_quasiprobability_terms(num_wire_cuts: usize) -> Vec<QuasiProbTerm> {
    if num_wire_cuts == 0 {
        return vec![QuasiProbTerm {
            basis: Vec::new(),
            coefficient: 1.0,
        }];
    }

    let basis = [
        PauliBasisTerm::I,
        PauliBasisTerm::X,
        PauliBasisTerm::Y,
        PauliBasisTerm::Z,
    ];
    let coeff = 0.5f64.powi(num_wire_cuts as i32);
    let mut terms = vec![QuasiProbTerm {
        basis: Vec::new(),
        coefficient: coeff,
    }];

    for _ in 0..num_wire_cuts {
        let mut next = Vec::with_capacity(terms.len() * basis.len());
        for t in &terms {
            for &b in &basis {
                let mut bt = t.basis.clone();
                bt.push(b);
                next.push(QuasiProbTerm {
                    basis: bt,
                    coefficient: coeff,
                });
            }
        }
        terms = next;
    }

    terms
}

/// Reconstruct global observable from wire-cut quasiprobability term estimates.
pub fn reconstruct_from_quasiprobability_terms(
    terms: &[QuasiProbTerm],
    term_estimates: &[f64],
) -> Result<f64, String> {
    if terms.len() != term_estimates.len() {
        return Err("term_estimates length must match terms length".to_string());
    }
    if terms.is_empty() {
        return Ok(0.0);
    }
    Ok(terms
        .iter()
        .zip(term_estimates.iter())
        .map(|(t, &e)| t.coefficient * e)
        .sum())
}

/// Estimate shot complexity for a chosen reconstruction mode.
pub fn estimate_sampling_cost(
    plan: &CutPlan,
    base_shots_per_fragment: u64,
    mode: ReconstructionMode,
) -> SamplingCostEstimate {
    let num_fragments = plan.fragments.len();
    let num_cuts = plan.cut_points.len();
    let (num_gate_cuts, num_wire_cuts) = count_cut_kinds(plan);
    let fragment_evaluations = num_fragments.max(1);

    let quasiprobability_overhead = if mode == ReconstructionMode::QuasiProbability {
        // Rough proxy: gate cuts ~3x each, wire cuts ~4x each.
        3f64.powi(num_gate_cuts as i32) * 4f64.powi(num_wire_cuts as i32)
    } else {
        1.0
    };

    let estimated_total_shots = ((base_shots_per_fragment as f64)
        * (fragment_evaluations as f64)
        * quasiprobability_overhead)
        .ceil() as u64;

    SamplingCostEstimate {
        num_fragments,
        num_cuts,
        fragment_evaluations,
        quasiprobability_overhead,
        estimated_total_shots,
    }
}

/// Search over candidate auto-cut configurations and return the best-scoring plan.
pub fn search_best_cut_plan(
    gates: &[Gate],
    search: &CutSearchConfig,
) -> Result<ScoredCutPlan, String> {
    if search.max_fragment_qubits_candidates.is_empty() {
        return Err("max_fragment_qubits_candidates cannot be empty".to_string());
    }
    if search.lookahead_candidates.is_empty() {
        return Err("lookahead_candidates cannot be empty".to_string());
    }
    if search.base_shots_per_fragment == 0 {
        return Err("base_shots_per_fragment must be >= 1".to_string());
    }

    let mut best: Option<ScoredCutPlan> = None;

    for &max_q in &search.max_fragment_qubits_candidates {
        if max_q == 0 {
            continue;
        }
        for &lookahead in &search.lookahead_candidates {
            let cfg = AutoCutConfig {
                max_fragment_qubits: max_q,
                lookahead_gates: lookahead.max(1),
                prefer_wire_cuts: true,
                wire_cut_min_future_uses: 2,
            };
            let plan = plan_cuts_auto(gates, &cfg);
            let sampling_cost =
                estimate_sampling_cost(&plan, search.base_shots_per_fragment, search.mode);
            let objective = cut_plan_objective(&plan, &sampling_cost, search);

            let candidate = ScoredCutPlan {
                plan,
                auto_config: cfg,
                sampling_cost,
                objective,
            };

            match &best {
                None => best = Some(candidate),
                Some(curr) => {
                    if candidate.objective < curr.objective {
                        best = Some(candidate);
                    }
                }
            }
        }
    }

    best.ok_or_else(|| "no valid cut-plan candidates found".to_string())
}

/// Evaluate a cut plan with a user-supplied fragment evaluator.
pub fn evaluate_cut_plan<F>(plan: &CutPlan, mut evaluator: F) -> Result<f64, String>
where
    F: FnMut(&CircuitFragment) -> Result<f64, String>,
{
    let mut vals = Vec::with_capacity(plan.fragments.len());
    for frag in &plan.fragments {
        vals.push(evaluator(frag)?);
    }
    Ok(stitch_fragment_estimates(&vals))
}

/// Evaluate a cut plan with explicit reconstruction mode.
pub fn evaluate_cut_plan_with_mode<F>(
    plan: &CutPlan,
    mut evaluator: F,
    mode: ReconstructionMode,
) -> Result<f64, String>
where
    F: FnMut(&CircuitFragment) -> Result<f64, String>,
{
    let mut vals = Vec::with_capacity(plan.fragments.len());
    for frag in &plan.fragments {
        vals.push(evaluator(frag)?);
    }
    reconstruct_from_fragment_estimates(&vals, None, mode)
}

/// Convenience execution path: evaluate each fragment with a local state-vector
/// and return stitched Z-expectation on each fragment's first local qubit.
pub fn execute_cut_circuit_z(gates: &[Gate], max_fragment_qubits: usize) -> Result<f64, String> {
    let plan = plan_cuts(gates, max_fragment_qubits);
    execute_cut_plan_z_product(&plan)
}

/// Execute a precomputed cut plan and reconstruct using product stitching.
pub fn execute_cut_plan_z_product(plan: &CutPlan) -> Result<f64, String> {
    evaluate_cut_plan(plan, |fragment| {
        let stats = execute_fragment_with_boundary_observables(fragment)?;
        Ok(stats.observable_z)
    })
}

/// Execute a wire-cut aware plan and reconstruct Z-observable using a
/// factorized quasiprobability model.
pub fn execute_wire_cut_plan_z_quasiprobability(plan: &CutPlan) -> Result<f64, String> {
    if plan.fragments.is_empty() {
        return Ok(0.0);
    }

    let mut frag_stats = Vec::with_capacity(plan.fragments.len());
    for frag in &plan.fragments {
        frag_stats.push(execute_fragment_with_boundary_observables(frag)?);
    }

    let base_estimate = frag_stats
        .iter()
        .map(|s| s.observable_z)
        .fold(1.0_f64, |acc, v| acc * v);

    let (_, wire_cuts) = count_cut_kinds(plan);
    if wire_cuts == 0 {
        return Ok(base_estimate);
    }

    let mut cut_factors = Vec::with_capacity(wire_cuts);
    for cut in plan.cut_points.iter().filter(|c| c.kind == CutKind::Wire) {
        let mut count = 0usize;
        let mut sx = 0.0;
        let mut sy = 0.0;
        let mut sz = 0.0;

        for stats in &frag_stats {
            if let Some(&(x, y, z)) = stats.boundary_xyz.get(&cut.wire) {
                count += 1;
                sx += x;
                sy += y;
                sz += z;
            }
        }

        if count == 0 {
            // No boundary estimate found for this cut, treat as maximally mixed.
            cut_factors.push((0.0, 0.0, 0.0));
        } else {
            let inv = 1.0 / count as f64;
            cut_factors.push((sx * inv, sy * inv, sz * inv));
        }
    }

    Ok(reconstruct_quasiprobability_factorized(
        base_estimate,
        &cut_factors,
    ))
}

/// End-to-end convenience path for wire-cut quasiprobability execution.
pub fn execute_wire_cut_circuit_z_quasiprobability(
    gates: &[Gate],
    max_fragment_qubits: usize,
    lookahead_gates: usize,
) -> Result<f64, String> {
    let plan = plan_wire_cuts_auto(gates, max_fragment_qubits, lookahead_gates);
    execute_wire_cut_plan_z_quasiprobability(&plan)
}

fn build_fragments(gates: &[Gate], cuts: &[CutPoint]) -> Vec<CircuitFragment> {
    if gates.is_empty() {
        return vec![];
    }

    let mut boundaries: Vec<usize> = cuts.iter().map(|c| c.gate_index).collect();
    boundaries.sort_unstable();
    boundaries.dedup();

    let mut start = 0usize;
    let mut fragments = Vec::new();

    for &b in &boundaries {
        let end = b.min(gates.len());
        if start < end {
            fragments.push(make_fragment(&gates[start..end], cuts, start, end));
        }
        start = end;
    }

    if start < gates.len() {
        fragments.push(make_fragment(&gates[start..], cuts, start, gates.len()));
    }

    fragments
}

/// Count (gate_cuts, wire_cuts) in a plan.
pub fn count_cut_kinds(plan: &CutPlan) -> (usize, usize) {
    let mut gate = 0usize;
    let mut wire = 0usize;
    for c in &plan.cut_points {
        match c.kind {
            CutKind::Gate => gate += 1,
            CutKind::Wire => wire += 1,
        }
    }
    (gate, wire)
}

#[derive(Clone, Debug)]
struct FragmentObservableStats {
    observable_z: f64,
    boundary_xyz: HashMap<usize, (f64, f64, f64)>,
}

fn execute_fragment_with_boundary_observables(
    fragment: &CircuitFragment,
) -> Result<FragmentObservableStats, String> {
    if fragment.local_qubits.is_empty() {
        return Ok(FragmentObservableStats {
            observable_z: 1.0,
            boundary_xyz: HashMap::new(),
        });
    }

    let local_n = fragment.local_qubits.len();
    let mut remap = HashMap::new();
    for (i, &q) in fragment.local_qubits.iter().enumerate() {
        remap.insert(q, i);
    }

    let mut state = QuantumState::new(local_n);
    for g in &fragment.gates {
        let mapped = remap_gate(g, &remap)?;
        apply_gate(&mut state, &mapped)?;
    }

    let observable_z = state.expectation_z(0);
    let mut boundary_xyz = HashMap::new();
    let mut boundary_wires = BTreeSet::new();
    boundary_wires.extend(fragment.input_wires.iter().copied());
    boundary_wires.extend(fragment.output_wires.iter().copied());

    for wire in boundary_wires {
        if let Some(&local_idx) = remap.get(&wire) {
            boundary_xyz.insert(
                wire,
                (
                    state.expectation_x(local_idx),
                    state.expectation_y(local_idx),
                    state.expectation_z(local_idx),
                ),
            );
        }
    }

    Ok(FragmentObservableStats {
        observable_z,
        boundary_xyz,
    })
}

/// Factorized quasiprobability stitcher for wire cuts.
///
/// Uses per-cut (x,y,z) moments and avoids explicit `4^k` term enumeration:
/// reconstruction factor per cut is `0.5 * (1 + x + y + z)`.
fn reconstruct_quasiprobability_factorized(
    base_estimate: f64,
    per_cut_xyz: &[(f64, f64, f64)],
) -> f64 {
    let mut factor = 1.0f64;
    for &(x, y, z) in per_cut_xyz {
        factor *= 0.5 * (1.0 + x + y + z);
    }
    base_estimate * factor
}

fn future_wire_usage(
    gates: &[Gate],
    start_gate: usize,
    wire: usize,
    lookahead_gates: usize,
) -> usize {
    if gates.is_empty() || start_gate >= gates.len() {
        return 0;
    }

    let end = (start_gate + lookahead_gates + 1).min(gates.len());
    let mut usage = 0usize;

    for g in gates.iter().take(end).skip(start_gate + 1) {
        if g.targets.contains(&wire) || g.controls.contains(&wire) {
            usage += 1;
        }
    }

    usage
}

fn distinct_gate_wires(g: &Gate) -> Vec<usize> {
    let mut wires: Vec<usize> = g.targets.iter().chain(g.controls.iter()).copied().collect();
    wires.sort_unstable();
    wires.dedup();
    wires
}

fn cut_plan_objective(
    plan: &CutPlan,
    sampling_cost: &SamplingCostEstimate,
    search: &CutSearchConfig,
) -> f64 {
    let max_fragment_qubits = plan
        .fragments
        .iter()
        .map(|f| f.local_qubits.len())
        .max()
        .unwrap_or(0) as f64;

    sampling_cost.estimated_total_shots as f64
        + search.cut_penalty * plan.cut_points.len() as f64
        + search.fragment_qubit_penalty * max_fragment_qubits * max_fragment_qubits
}

fn make_fragment(
    slice: &[Gate],
    cuts: &[CutPoint],
    global_start: usize,
    global_end: usize,
) -> CircuitFragment {
    let mut set = BTreeSet::new();
    for g in slice {
        for &q in g.targets.iter().chain(g.controls.iter()) {
            set.insert(q);
        }
    }

    let input_wires = cuts
        .iter()
        .filter(|c| c.gate_index == global_start && c.kind == CutKind::Wire)
        .map(|c| c.wire)
        .collect();
    let output_wires = cuts
        .iter()
        .filter(|c| c.gate_index == global_end && c.kind == CutKind::Wire)
        .map(|c| c.wire)
        .collect();

    CircuitFragment {
        gates: slice.to_vec(),
        local_qubits: set.into_iter().collect(),
        input_wires,
        output_wires,
    }
}

fn remap_gate(g: &Gate, remap: &HashMap<usize, usize>) -> Result<Gate, String> {
    let targets = g
        .targets
        .iter()
        .map(|q| {
            remap
                .get(q)
                .copied()
                .ok_or_else(|| format!("unmapped target qubit {}", q))
        })
        .collect::<Result<Vec<_>, _>>()?;
    let controls = g
        .controls
        .iter()
        .map(|q| {
            remap
                .get(q)
                .copied()
                .ok_or_else(|| format!("unmapped control qubit {}", q))
        })
        .collect::<Result<Vec<_>, _>>()?;

    Ok(Gate {
        gate_type: g.gate_type.clone(),
        targets,
        controls,
        params: g.params.clone(),
    })
}

fn apply_gate(state: &mut QuantumState, gate: &Gate) -> Result<(), String> {
    match &gate.gate_type {
        GateType::H => GateOperations::h(state, gate.targets[0]),
        GateType::X => GateOperations::x(state, gate.targets[0]),
        GateType::Y => GateOperations::y(state, gate.targets[0]),
        GateType::Z => GateOperations::z(state, gate.targets[0]),
        GateType::S => GateOperations::s(state, gate.targets[0]),
        GateType::T => GateOperations::t(state, gate.targets[0]),
        GateType::Rx(theta) => GateOperations::rx(state, gate.targets[0], *theta),
        GateType::Ry(theta) => GateOperations::ry(state, gate.targets[0], *theta),
        GateType::Rz(theta) => GateOperations::rz(state, gate.targets[0], *theta),
        GateType::CNOT => GateOperations::cnot(state, gate.controls[0], gate.targets[0]),
        GateType::CZ => GateOperations::cz(state, gate.controls[0], gate.targets[0]),
        GateType::SWAP => GateOperations::swap(state, gate.targets[0], gate.targets[1]),
        GateType::Toffoli => {
            GateOperations::toffoli(state, gate.controls[0], gate.controls[1], gate.targets[0])
        }
        GateType::CRx(theta) => {
            GateOperations::crx(state, gate.controls[0], gate.targets[0], *theta)
        }
        GateType::CRy(theta) => {
            GateOperations::cry(state, gate.controls[0], gate.targets[0], *theta)
        }
        GateType::CRz(theta) => {
            GateOperations::crz(state, gate.controls[0], gate.targets[0], *theta)
        }
        GateType::Phase(theta) => GateOperations::phase(state, gate.targets[0], *theta),
        _ => {
            return Err(format!(
                "unsupported gate type in circuit-cutting executor: {:?}",
                gate.gate_type
            ))
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_plan_cuts_for_wide_circuit() {
        let mut gates = Vec::new();
        for q in 0..6 {
            gates.push(Gate::h(q));
        }
        gates.push(Gate::cnot(0, 5));
        gates.push(Gate::cnot(1, 4));

        let plan = plan_cuts(&gates, 3);
        assert!(!plan.fragments.is_empty());
        assert!(!plan.cut_points.is_empty());
    }

    #[test]
    fn test_fragment_qubit_bound() {
        let gates = vec![
            Gate::h(0),
            Gate::h(1),
            Gate::h(2),
            Gate::h(3),
            Gate::cnot(0, 3),
            Gate::cnot(1, 2),
        ];

        let plan = plan_cuts(&gates, 2);
        assert!(plan.fragments.iter().all(|f| f.local_qubits.len() <= 4));
    }

    #[test]
    fn test_stitch_product() {
        let vals = vec![0.9, 0.8, 1.1];
        let s = stitch_fragment_estimates(&vals);
        assert!((s - 0.792).abs() < 1e-12);
    }

    #[test]
    fn test_auto_plan_cuts_for_wide_circuit() {
        let gates = vec![
            Gate::h(0),
            Gate::h(1),
            Gate::h(2),
            Gate::h(3),
            Gate::cnot(0, 1),
            Gate::cnot(2, 3),
            Gate::cnot(0, 3),
        ];

        let cfg = AutoCutConfig {
            max_fragment_qubits: 2,
            lookahead_gates: 8,
            prefer_wire_cuts: true,
            wire_cut_min_future_uses: 1,
        };
        let plan = plan_cuts_auto(&gates, &cfg);
        assert!(!plan.cut_points.is_empty());
        assert!(!plan.fragments.is_empty());
    }

    #[test]
    fn test_auto_plan_single_fragment_when_under_bound() {
        let gates = vec![Gate::h(0), Gate::cnot(0, 1), Gate::rz(1, 0.2)];
        let cfg = AutoCutConfig {
            max_fragment_qubits: 4,
            lookahead_gates: 4,
            prefer_wire_cuts: true,
            wire_cut_min_future_uses: 2,
        };
        let plan = plan_cuts_auto(&gates, &cfg);
        assert!(plan.cut_points.is_empty());
        assert_eq!(plan.fragments.len(), 1);
    }

    #[test]
    fn test_weighted_reconstruction_prefers_low_variance() {
        let values = vec![0.2, 1.0];
        let variances = vec![1.0, 1e-6];
        let rec = reconstruct_from_fragment_estimates(
            &values,
            Some(&variances),
            ReconstructionMode::WeightedMean,
        )
        .expect("weighted reconstruction");
        assert!((rec - 1.0).abs() < 1e-3);
    }

    #[test]
    fn test_sampling_cost_quasiprobability_overhead() {
        let gates = vec![
            Gate::h(0),
            Gate::h(1),
            Gate::h(2),
            Gate::h(3),
            Gate::cnot(0, 3),
            Gate::cnot(1, 2),
        ];
        let plan = plan_cuts(&gates, 2);
        let cost = estimate_sampling_cost(&plan, 100, ReconstructionMode::QuasiProbability);
        assert!(cost.quasiprobability_overhead >= 1.0);
        assert!(cost.estimated_total_shots >= 100);
    }

    #[test]
    fn test_wire_cut_quasiprobability_term_count() {
        let terms = wire_cut_quasiprobability_terms(2);
        assert_eq!(terms.len(), 16);
        assert!(
            terms
                .iter()
                .all(|t| (t.coefficient - 0.25).abs() < 1e-12 && t.basis.len() == 2)
        );
    }

    #[test]
    fn test_reconstruct_from_quasiprobability_terms() {
        let terms = wire_cut_quasiprobability_terms(1);
        let estimates = vec![1.0, 0.5, -0.5, 0.0];
        let rec = reconstruct_from_quasiprobability_terms(&terms, &estimates)
            .expect("quasiprob reconstruction");
        // 0.5 * (1.0 + 0.5 - 0.5 + 0.0) = 0.5
        assert!((rec - 0.5).abs() < 1e-12);
    }

    #[test]
    fn test_sampling_cost_mixed_gate_and_wire_cuts() {
        let plan = CutPlan {
            cut_points: vec![
                CutPoint {
                    gate_index: 1,
                    wire: 0,
                    kind: CutKind::Gate,
                },
                CutPoint {
                    gate_index: 2,
                    wire: 1,
                    kind: CutKind::Wire,
                },
            ],
            fragments: vec![
                CircuitFragment {
                    gates: vec![Gate::h(0)],
                    local_qubits: vec![0],
                    input_wires: vec![],
                    output_wires: vec![],
                },
                CircuitFragment {
                    gates: vec![Gate::h(1)],
                    local_qubits: vec![1],
                    input_wires: vec![1],
                    output_wires: vec![],
                },
            ],
        };

        let cost = estimate_sampling_cost(&plan, 100, ReconstructionMode::QuasiProbability);
        // gate_cut=1 => x3, wire_cut=1 => x4 => total x12.
        assert!((cost.quasiprobability_overhead - 12.0).abs() < 1e-12);
        let (g, w) = count_cut_kinds(&plan);
        assert_eq!((g, w), (1, 1));
    }

    #[test]
    fn test_evaluate_cut_plan_with_mode_mean() {
        let plan = CutPlan {
            cut_points: vec![],
            fragments: vec![
                CircuitFragment {
                    gates: vec![],
                    local_qubits: vec![0],
                    input_wires: vec![],
                    output_wires: vec![],
                },
                CircuitFragment {
                    gates: vec![],
                    local_qubits: vec![1],
                    input_wires: vec![],
                    output_wires: vec![],
                },
            ],
        };
        let vals = [0.4, 0.8];
        let mut i = 0usize;
        let out = evaluate_cut_plan_with_mode(
            &plan,
            |_frag| {
                let v = vals[i];
                i += 1;
                Ok(v)
            },
            ReconstructionMode::Mean,
        )
        .expect("mean reconstruction");
        assert!((out - 0.6).abs() < 1e-12);
    }

    #[test]
    fn test_search_best_cut_plan_prefers_fewer_cuts_under_qp_mode() {
        let gates = vec![
            Gate::h(0),
            Gate::h(1),
            Gate::h(2),
            Gate::h(3),
            Gate::cnot(0, 3),
            Gate::cnot(1, 2),
            Gate::cnot(0, 1),
        ];

        let search = CutSearchConfig {
            max_fragment_qubits_candidates: vec![2, 8],
            lookahead_candidates: vec![4],
            base_shots_per_fragment: 100,
            mode: ReconstructionMode::QuasiProbability,
            cut_penalty: 0.0,
            fragment_qubit_penalty: 0.0,
        };
        let best = search_best_cut_plan(&gates, &search).expect("best plan");
        assert_eq!(best.auto_config.max_fragment_qubits, 8);
        assert!(best.plan.cut_points.is_empty());
    }

    #[test]
    fn test_search_best_cut_plan_rejects_empty_candidate_lists() {
        let search = CutSearchConfig {
            max_fragment_qubits_candidates: vec![],
            ..CutSearchConfig::default()
        };
        let err = search_best_cut_plan(&[], &search).expect_err("expected validation error");
        assert!(err.contains("cannot be empty"));
    }

    #[test]
    fn test_plan_wire_cuts_auto_marks_wire_cut() {
        let gates = vec![
            Gate::h(0),
            Gate::cnot(0, 1),
            Gate::cnot(0, 2),
            Gate::cnot(0, 3),
            Gate::rz(0, 0.1),
            Gate::cnot(0, 4),
        ];
        let plan = plan_wire_cuts_auto(&gates, 2, 6);
        assert!(!plan.cut_points.is_empty());
        assert!(plan.cut_points.iter().any(|c| c.kind == CutKind::Wire));
    }

    #[test]
    fn test_execute_cut_plan_z_product_is_finite() {
        let gates = vec![Gate::h(0), Gate::cnot(0, 1), Gate::rz(1, 0.2)];
        let plan = plan_cuts_auto(&gates, &AutoCutConfig::default());
        let out = execute_cut_plan_z_product(&plan).expect("product exec");
        assert!(out.is_finite());
    }

    #[test]
    fn test_execute_wire_cut_plan_z_quasiprobability_is_finite() {
        let gates = vec![
            Gate::h(0),
            Gate::cnot(0, 1),
            Gate::cnot(0, 2),
            Gate::cnot(0, 3),
            Gate::rz(0, 0.1),
        ];
        let plan = plan_wire_cuts_auto(&gates, 2, 6);
        let out = execute_wire_cut_plan_z_quasiprobability(&plan).expect("wire qp exec");
        assert!(out.is_finite());
    }
}
