//! Distributed Metal + MPI shard execution.
//!
//! This module provides:
//! - `DistributedMetalShardExecutor`: per-rank execution with MPI sync stubs.
//! - `DistributedMetalWorldExecutor`: single-process multi-rank execution that
//!   performs real remote-gate updates via gather/apply/scatter exchange.

use crate::distributed_mpi::MPICommunicator;
use crate::gates::{Gate, GateType};
use crate::metal_backend::MetalSimulator;
use crate::{GateOperations, QuantumState, C64};
use std::collections::BTreeSet;
use std::time::Instant;

/// Per-rank shard layout for block-distributed state vectors.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ShardLayout {
    pub global_num_qubits: usize,
    pub world_size: usize,
    pub rank: usize,
    /// Number of high-order index bits used for sharding.
    pub shard_bits: usize,
    /// Number of qubits represented in each local shard.
    pub local_num_qubits: usize,
    /// Number of amplitudes in local shard.
    pub local_amplitudes: usize,
    /// Global start/end amplitude offsets for this rank.
    pub global_start: usize,
    pub global_end: usize,
}

impl ShardLayout {
    pub fn for_rank(
        global_num_qubits: usize,
        world_size: usize,
        rank: usize,
    ) -> Result<Self, String> {
        if world_size == 0 {
            return Err("world_size must be >= 1".to_string());
        }
        if rank >= world_size {
            return Err("rank must be < world_size".to_string());
        }
        if !world_size.is_power_of_two() {
            return Err("world_size must be a power of two for current shard layout".to_string());
        }

        let shard_bits = world_size.trailing_zeros() as usize;
        if shard_bits > global_num_qubits {
            return Err(
                "global_num_qubits must be >= log2(world_size) for distributed sharding"
                    .to_string(),
            );
        }

        let local_num_qubits = global_num_qubits - shard_bits;
        let local_amplitudes = 1usize << local_num_qubits;
        let global_start = rank * local_amplitudes;
        let global_end = global_start + local_amplitudes;

        Ok(Self {
            global_num_qubits,
            world_size,
            rank,
            shard_bits,
            local_num_qubits,
            local_amplitudes,
            global_start,
            global_end,
        })
    }

    pub fn is_local_qubit(&self, qubit: usize) -> bool {
        qubit < self.local_num_qubits
    }

    pub fn gate_is_local(&self, gate: &Gate) -> bool {
        gate.targets
            .iter()
            .chain(gate.controls.iter())
            .all(|&q| self.is_local_qubit(q))
    }

    pub fn localize_gate(&self, gate: &Gate) -> Result<Gate, String> {
        if !self.gate_is_local(gate) {
            return Err("cannot localize gate containing shard-domain qubits".to_string());
        }
        Ok(gate.clone())
    }

    /// Derive communication partners for shard-domain qubits touched by a gate.
    pub fn communication_partners_for_gate(&self, gate: &Gate) -> Vec<usize> {
        let mut partners = BTreeSet::new();
        for &q in gate.targets.iter().chain(gate.controls.iter()) {
            if q >= self.local_num_qubits {
                let shard_bit = q - self.local_num_qubits;
                if shard_bit < self.shard_bits {
                    let partner = self.rank ^ (1usize << shard_bit);
                    if partner < self.world_size && partner != self.rank {
                        partners.insert(partner);
                    }
                }
            }
        }
        partners.into_iter().collect()
    }
}

#[derive(Clone, Debug)]
pub struct DistributedMetalConfig {
    /// Require Metal backend; return error if unavailable.
    pub strict_gpu_only: bool,
    /// If true, return error on first remote gate (instead of synchronization stub).
    pub fail_on_remote_gates: bool,
    /// Remote execution behavior for per-rank shard executor.
    pub remote_execution_mode: ShardRemoteExecutionMode,
    /// Maximum local-gate batch submitted in one backend call.
    pub max_local_batch: usize,
}

impl Default for DistributedMetalConfig {
    fn default() -> Self {
        Self {
            strict_gpu_only: false,
            fail_on_remote_gates: false,
            remote_execution_mode: ShardRemoteExecutionMode::Skip,
            max_local_batch: 256,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ShardRemoteExecutionMode {
    /// Legacy behavior: synchronize metadata and skip unsupported remote updates.
    Skip,
    /// Execute full circuit via single-process world executor and return this rank's shard.
    /// This gives exact remote-gate semantics for shard API users.
    EmulatedWorldExact,
}

/// Lightweight communication cost model used to tune batch scheduling.
#[derive(Clone, Debug)]
pub struct CommunicationCostModel {
    /// Fixed per-communication event overhead.
    pub latency_cost: f64,
    /// Per-remote-gate transfer/processing overhead.
    pub bandwidth_cost: f64,
    /// Relative weight for pairwise paths.
    pub pairwise_weight: f64,
    /// Relative weight for global fallback paths.
    pub fallback_weight: f64,
}

impl Default for CommunicationCostModel {
    fn default() -> Self {
        Self {
            latency_cost: 1.0,
            bandwidth_cost: 0.2,
            pairwise_weight: 1.0,
            fallback_weight: 1.6,
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct SchedulingPlan {
    local_batch_limit: usize,
    global_fallback_batch_limit: usize,
    estimated_comm_cost: f64,
}

#[derive(Clone, Debug, Default)]
pub struct DistributedMetalMetrics {
    pub local_gates: usize,
    pub remote_gates: usize,
    pub remote_gates_executed: usize,
    pub remote_gates_no_exchange: usize,
    pub remote_gates_skipped: usize,
    pub remote_gates_exchange_required: usize,
    pub remote_gates_pairwise_fast_path: usize,
    pub remote_gates_global_fallback: usize,
    pub remote_gates_global_fallback_batches: usize,
    pub communication_events: usize,
    pub communication_partners: usize,
    pub local_batches: usize,
    pub adaptive_local_batch_limit: usize,
    pub adaptive_global_fallback_batch_limit: usize,
    pub scheduler_estimated_comm_cost: f64,
    pub shard_remote_world_emulation_used: bool,
    pub elapsed_ms: f64,
}

#[derive(Clone, Debug)]
pub struct DistributedMetalRunResult {
    pub local_probabilities: Vec<f64>,
    pub metrics: DistributedMetalMetrics,
    pub layout: ShardLayout,
}

#[derive(Clone, Debug)]
pub struct DistributedMetalWorldRunResult {
    pub global_probabilities: Vec<f64>,
    pub per_rank_probabilities: Vec<Vec<f64>>,
    pub metrics: DistributedMetalMetrics,
    pub layouts: Vec<ShardLayout>,
}

/// Single-process distributed executor that simulates all ranks together.
///
/// It executes local gates shard-wise (Metal if available), and executes
/// remote gates via no-exchange reductions, pairwise shard fast paths, and
/// batched gather/apply/scatter global fallback.
pub struct DistributedMetalWorldExecutor {
    global_num_qubits: usize,
    layouts: Vec<ShardLayout>,
    backends: Vec<LocalBackendState>,
    global_fallback_gpu: Option<MetalSimulator>,
    scheduler_model: CommunicationCostModel,
    config: DistributedMetalConfig,
}

impl DistributedMetalWorldExecutor {
    pub fn new(
        global_num_qubits: usize,
        world_size: usize,
        config: DistributedMetalConfig,
    ) -> Result<Self, String> {
        let mut layouts = Vec::with_capacity(world_size);
        let mut backends = Vec::with_capacity(world_size);
        for rank in 0..world_size {
            let layout = ShardLayout::for_rank(global_num_qubits, world_size, rank)?;
            let backend = LocalBackendState::new(layout.local_num_qubits, config.strict_gpu_only)?;
            layouts.push(layout);
            backends.push(backend);
        }

        let mut executor = Self {
            global_num_qubits,
            layouts,
            backends,
            global_fallback_gpu: if config.strict_gpu_only {
                Some(MetalSimulator::new(global_num_qubits).map_err(|e| {
                    format!("strict_gpu_only global fallback GPU init failed: {}", e)
                })?)
            } else {
                None
            },
            scheduler_model: CommunicationCostModel::default(),
            config,
        };
        executor.initialize_ground_state()?;
        Ok(executor)
    }

    pub fn set_cost_model(&mut self, model: CommunicationCostModel) {
        self.scheduler_model = model;
    }

    pub fn layouts(&self) -> &[ShardLayout] {
        &self.layouts
    }

    pub fn execute_partitioned(
        &mut self,
        gates: &[Gate],
    ) -> Result<DistributedMetalWorldRunResult, String> {
        self.initialize_ground_state()?;
        let mut metrics = DistributedMetalMetrics::default();
        let t0 = Instant::now();

        let mut local_batch: Vec<Gate> = Vec::new();
        let mut global_fallback_batch: Vec<Gate> = Vec::new();
        let mut all_partners = BTreeSet::new();

        let local_layout = self.layouts[0].clone();
        let scheduling_plan =
            build_scheduling_plan(&local_layout, gates, &self.config, &self.scheduler_model);
        let local_batch_limit = scheduling_plan.local_batch_limit;
        let global_fallback_batch_limit = scheduling_plan.global_fallback_batch_limit;
        metrics.adaptive_local_batch_limit = local_batch_limit;
        metrics.adaptive_global_fallback_batch_limit = global_fallback_batch_limit;
        metrics.scheduler_estimated_comm_cost = scheduling_plan.estimated_comm_cost;

        for gate in gates {
            if local_layout.gate_is_local(gate) {
                self.flush_global_fallback_batch(&mut global_fallback_batch, &mut metrics)?;
                metrics.local_gates += 1;
                local_batch.push(local_layout.localize_gate(gate)?);
                if local_batch.len() >= local_batch_limit {
                    self.flush_local_batch_all(&mut local_batch, &mut metrics)?;
                }
                continue;
            }

            self.flush_local_batch_all(&mut local_batch, &mut metrics)?;
            metrics.remote_gates += 1;

            for layout in &self.layouts {
                for partner in layout.communication_partners_for_gate(gate) {
                    all_partners.insert(partner);
                }
            }

            match classify_remote_execution(&local_layout, gate) {
                RemoteExecutionKind::NoExchange => {
                    self.flush_global_fallback_batch(&mut global_fallback_batch, &mut metrics)?;
                    for idx in 0..self.layouts.len() {
                        if !apply_remote_gate_without_exchange_for_layout(
                            &self.layouts[idx],
                            &mut self.backends[idx],
                            gate,
                        )? {
                            return Err(format!(
                                "remote no-exchange classification mismatch for gate {:?}",
                                gate
                            ));
                        }
                    }
                    metrics.remote_gates_executed += 1;
                    metrics.remote_gates_no_exchange += 1;
                }
                RemoteExecutionKind::Pairwise => {
                    self.flush_global_fallback_batch(&mut global_fallback_batch, &mut metrics)?;
                    metrics.remote_gates_exchange_required += 1;
                    if self.try_apply_pairwise_remote_gate(gate, &mut metrics)? {
                        metrics.remote_gates_pairwise_fast_path += 1;
                        metrics.remote_gates_executed += 1;
                    } else {
                        if self.config.strict_gpu_only && self.global_fallback_gpu.is_none() {
                            return Err(format!(
                                "strict_gpu_only cannot execute remote gate without pairwise kernel (global GPU fallback unavailable): {:?}",
                                gate
                            ));
                        }
                        global_fallback_batch.push(gate.clone());
                        metrics.remote_gates_global_fallback += 1;
                        if global_fallback_batch.len() >= global_fallback_batch_limit {
                            self.flush_global_fallback_batch(
                                &mut global_fallback_batch,
                                &mut metrics,
                            )?;
                        }
                    }
                }
                RemoteExecutionKind::GlobalFallback => {
                    if self.config.strict_gpu_only && self.global_fallback_gpu.is_none() {
                        return Err(format!(
                            "strict_gpu_only cannot execute remote gate (global GPU fallback unavailable): {:?}",
                            gate
                        ));
                    }
                    metrics.remote_gates_exchange_required += 1;
                    global_fallback_batch.push(gate.clone());
                    metrics.remote_gates_global_fallback += 1;
                    if global_fallback_batch.len() >= global_fallback_batch_limit {
                        self.flush_global_fallback_batch(&mut global_fallback_batch, &mut metrics)?;
                    }
                }
            }
        }

        self.flush_local_batch_all(&mut local_batch, &mut metrics)?;
        self.flush_global_fallback_batch(&mut global_fallback_batch, &mut metrics)?;
        metrics.communication_partners = all_partners.len();
        metrics.elapsed_ms = t0.elapsed().as_secs_f64() * 1000.0;

        let global_state = self.gather_global_state()?;
        let global_probabilities: Vec<f64> = global_state.iter().map(|a| a.norm_sqr()).collect();
        let per_rank_probabilities = self.backends.iter().map(|b| b.probabilities()).collect();

        Ok(DistributedMetalWorldRunResult {
            global_probabilities,
            per_rank_probabilities,
            metrics,
            layouts: self.layouts.clone(),
        })
    }

    fn initialize_ground_state(&mut self) -> Result<(), String> {
        let mut global = vec![C64::new(0.0, 0.0); 1usize << self.global_num_qubits];
        global[0] = C64::new(1.0, 0.0);
        self.scatter_global_state(&global)
    }

    fn flush_local_batch_all(
        &mut self,
        local_batch: &mut Vec<Gate>,
        metrics: &mut DistributedMetalMetrics,
    ) -> Result<(), String> {
        if local_batch.is_empty() {
            return Ok(());
        }
        for backend in &mut self.backends {
            backend.run_batch(local_batch)?;
        }
        metrics.local_batches += 1;
        local_batch.clear();
        Ok(())
    }

    fn gather_global_state(&self) -> Result<Vec<C64>, String> {
        let mut global = vec![C64::new(0.0, 0.0); 1usize << self.global_num_qubits];
        for (layout, backend) in self.layouts.iter().zip(self.backends.iter()) {
            let local = backend.read_state()?;
            if local.len() != layout.local_amplitudes {
                return Err(format!(
                    "local state length mismatch on rank {}: got {}, expected {}",
                    layout.rank,
                    local.len(),
                    layout.local_amplitudes
                ));
            }
            global[layout.global_start..layout.global_end].copy_from_slice(&local);
        }
        Ok(global)
    }

    fn scatter_global_state(&mut self, global: &[C64]) -> Result<(), String> {
        let expected = 1usize << self.global_num_qubits;
        if global.len() != expected {
            return Err(format!(
                "global state length mismatch: got {}, expected {}",
                global.len(),
                expected
            ));
        }
        for (layout, backend) in self.layouts.iter().zip(self.backends.iter_mut()) {
            backend.set_state(&global[layout.global_start..layout.global_end])?;
        }
        Ok(())
    }

    fn flush_global_fallback_batch(
        &mut self,
        global_fallback_batch: &mut Vec<Gate>,
        metrics: &mut DistributedMetalMetrics,
    ) -> Result<(), String> {
        if global_fallback_batch.is_empty() {
            return Ok(());
        }
        let gathered = self.gather_global_state()?;
        if let Some(global_gpu) = self.global_fallback_gpu.as_mut() {
            global_gpu.write_state(&gathered)?;
            global_gpu.run_circuit(global_fallback_batch);
            let evolved = global_gpu.read_state();
            self.scatter_global_state(&evolved)?;
        } else {
            let mut global_state = QuantumState::new(self.global_num_qubits);
            global_state.amplitudes_mut().copy_from_slice(&gathered);
            for gate in global_fallback_batch.iter() {
                apply_gate_cpu(&mut global_state, gate)?;
            }
            self.scatter_global_state(global_state.amplitudes_ref())?;
        }
        metrics.communication_events += 1;
        metrics.remote_gates_global_fallback_batches += 1;
        metrics.remote_gates_executed += global_fallback_batch.len();
        global_fallback_batch.clear();
        Ok(())
    }

    fn try_apply_pairwise_remote_gate(
        &mut self,
        gate: &Gate,
        metrics: &mut DistributedMetalMetrics,
    ) -> Result<bool, String> {
        if self.layouts.is_empty() {
            return Ok(false);
        }

        let layout = &self.layouts[0];
        let shard_controls: Vec<usize> = gate
            .controls
            .iter()
            .copied()
            .filter(|&q| !layout.is_local_qubit(q))
            .collect();
        let shard_targets: Vec<usize> = gate
            .targets
            .iter()
            .copied()
            .filter(|&q| !layout.is_local_qubit(q))
            .collect();
        let local_controls: Vec<usize> = gate
            .controls
            .iter()
            .copied()
            .filter(|&q| layout.is_local_qubit(q))
            .collect();
        let local_targets: Vec<usize> = gate
            .targets
            .iter()
            .copied()
            .filter(|&q| layout.is_local_qubit(q))
            .collect();

        // Two shard-target pairwise fast paths (no controls): SWAP / iSWAP.
        if shard_controls.is_empty()
            && gate.controls.is_empty()
            && local_controls.is_empty()
            && local_targets.is_empty()
            && shard_targets.len() == 2
            && gate.targets.len() == 2
        {
            let bit_a = shard_targets[0].saturating_sub(layout.local_num_qubits);
            let bit_b = shard_targets[1].saturating_sub(layout.local_num_qubits);
            if bit_a >= layout.shard_bits || bit_b >= layout.shard_bits || bit_a == bit_b {
                return Ok(false);
            }

            let mask_a = 1usize << bit_a;
            let mask_b = 1usize << bit_b;
            let group_mask = mask_a | mask_b;
            let mut pair_events = 0usize;

            match &gate.gate_type {
                GateType::SWAP => {
                    for base in 0..layout.world_size {
                        if (base & group_mask) != 0 {
                            continue;
                        }
                        let rank01 = base | mask_b;
                        let rank10 = base | mask_a;
                        let (left, right) = mutable_pair(&mut self.backends, rank01, rank10)?;
                        let state01 = left.read_state()?;
                        let state10 = right.read_state()?;
                        left.set_state(&state10)?;
                        right.set_state(&state01)?;
                        pair_events += 1;
                    }
                }
                GateType::ISWAP => {
                    let i_phase = C64::new(0.0, 1.0);
                    for base in 0..layout.world_size {
                        if (base & group_mask) != 0 {
                            continue;
                        }
                        let rank01 = base | mask_b;
                        let rank10 = base | mask_a;
                        let (left, right) = mutable_pair(&mut self.backends, rank01, rank10)?;
                        let old01 = left.read_state()?;
                        let old10 = right.read_state()?;
                        if old01.len() != old10.len() {
                            return Err(format!(
                                "pairwise iSWAP shard length mismatch: {} vs {}",
                                old01.len(),
                                old10.len()
                            ));
                        }
                        let mut new01 = vec![C64::new(0.0, 0.0); old01.len()];
                        let mut new10 = vec![C64::new(0.0, 0.0); old10.len()];
                        for idx in 0..old01.len() {
                            new01[idx] = i_phase * old10[idx];
                            new10[idx] = i_phase * old01[idx];
                        }
                        left.set_state(&new01)?;
                        right.set_state(&new10)?;
                        pair_events += 1;
                    }
                }
                _ => {}
            }

            if pair_events > 0 {
                metrics.communication_events += pair_events;
                return Ok(true);
            }
        }

        if !shard_controls.is_empty() || shard_targets.len() != 1 {
            // Additional fast path: shard-control -> shard-target CNOT is a pure
            // permutation over rank ownership and can be handled with pair swaps.
            if !matches!(&gate.gate_type, GateType::CNOT)
                || gate.controls.len() != 1
                || gate.targets.len() != 1
                || shard_controls.len() != 1
                || shard_targets.len() != 1
                || !local_controls.is_empty()
                || !local_targets.is_empty()
            {
                return Ok(false);
            }

            let control_bit = shard_controls[0].saturating_sub(layout.local_num_qubits);
            let target_bit = shard_targets[0].saturating_sub(layout.local_num_qubits);
            if control_bit >= layout.shard_bits
                || target_bit >= layout.shard_bits
                || control_bit == target_bit
            {
                return Ok(false);
            }

            let control_mask = 1usize << control_bit;
            let target_mask = 1usize << target_bit;
            let mut pair_events = 0usize;

            for rank0 in 0..layout.world_size {
                if (rank0 & control_mask) == 0 || (rank0 & target_mask) != 0 {
                    continue;
                }
                let rank1 = rank0 ^ target_mask;
                if rank1 >= layout.world_size {
                    return Ok(false);
                }

                let (left, right) = mutable_pair(&mut self.backends, rank0, rank1)?;
                let state0 = left.read_state()?;
                let state1 = right.read_state()?;
                left.set_state(&state1)?;
                right.set_state(&state0)?;
                pair_events += 1;
            }

            if pair_events > 0 {
                metrics.communication_events += pair_events;
                return Ok(true);
            }
            return Ok(false);
        }

        // Extended pairwise gates: SWAP/iSWAP with one local + one shard target,
        // or Toffoli with two local controls and one shard target.
        // These are dispatched through apply_pairwise_shard_gate which handles
        // multi-target and multi-control layouts.
        let is_extended_pairwise = shard_targets.len() == 1
            && shard_controls.is_empty()
            && match &gate.gate_type {
                GateType::SWAP | GateType::ISWAP => {
                    gate.targets.len() == 2
                        && gate.controls.is_empty()
                        && local_targets.len() == 1
                }
                GateType::Toffoli => {
                    gate.targets.len() == 1
                        && gate.controls.len() == 2
                        && gate.controls.iter().all(|&c| c < layout.local_num_qubits)
                }
                _ => false,
            };

        if is_extended_pairwise {
            let shard_bit = shard_targets[0].saturating_sub(layout.local_num_qubits);
            if shard_bit < layout.shard_bits {
                let rank_mask = 1usize << shard_bit;
                let mut pair_events = 0usize;
                for rank0 in 0..layout.world_size {
                    if (rank0 & rank_mask) != 0 {
                        continue;
                    }
                    let rank1 = rank0 ^ rank_mask;
                    if rank1 >= layout.world_size {
                        return Ok(false);
                    }
                    let (left, right) = mutable_pair(&mut self.backends, rank0, rank1)?;
                    let mut state0 = left.read_state()?;
                    let mut state1 = right.read_state()?;
                    if !apply_pairwise_shard_gate(
                        gate,
                        layout.local_num_qubits,
                        &mut state0,
                        &mut state1,
                    )? {
                        return Ok(false);
                    }
                    left.set_state(&state0)?;
                    right.set_state(&state1)?;
                    pair_events += 1;
                }
                if pair_events > 0 {
                    metrics.communication_events += pair_events;
                    return Ok(true);
                }
            }
        }

        let shard_target = shard_targets[0];
        if gate.targets.len() != 1 || gate.targets[0] != shard_target {
            return Ok(false);
        }

        let shard_bit = shard_target.saturating_sub(layout.local_num_qubits);
        if shard_bit >= layout.shard_bits {
            return Ok(false);
        }
        let rank_mask = 1usize << shard_bit;

        let mut pair_events = 0usize;
        for rank0 in 0..layout.world_size {
            if (rank0 & rank_mask) != 0 {
                continue;
            }
            let rank1 = rank0 ^ rank_mask;
            if rank1 >= layout.world_size {
                return Ok(false);
            }

            let (left, right) = mutable_pair(&mut self.backends, rank0, rank1)?;
            let mut state0 = left.read_state()?;
            let mut state1 = right.read_state()?;
            if !apply_pairwise_shard_gate(gate, layout.local_num_qubits, &mut state0, &mut state1)?
            {
                return Ok(false);
            }
            left.set_state(&state0)?;
            right.set_state(&state1)?;
            pair_events += 1;
        }

        if pair_events > 0 {
            metrics.communication_events += pair_events;
            return Ok(true);
        }
        Ok(false)
    }
}

pub struct DistributedMetalShardExecutor {
    mpi: MPICommunicator,
    layout: ShardLayout,
    config: DistributedMetalConfig,
}

impl DistributedMetalShardExecutor {
    pub fn new(
        global_num_qubits: usize,
        mpi: MPICommunicator,
        config: DistributedMetalConfig,
    ) -> Result<Self, String> {
        let layout = ShardLayout::for_rank(global_num_qubits, mpi.size(), mpi.rank())?;
        Ok(Self {
            mpi,
            layout,
            config,
        })
    }

    pub fn new_single_process(
        global_num_qubits: usize,
        config: DistributedMetalConfig,
    ) -> Result<Self, String> {
        Self::new(
            global_num_qubits,
            MPICommunicator { rank: 0, size: 1 },
            config,
        )
    }

    pub fn layout(&self) -> &ShardLayout {
        &self.layout
    }

    pub fn execute_partitioned(&self, gates: &[Gate]) -> Result<DistributedMetalRunResult, String> {
        if matches!(
            self.config.remote_execution_mode,
            ShardRemoteExecutionMode::EmulatedWorldExact
        ) && gates.iter().any(|g| !self.layout.gate_is_local(g))
        {
            let mut world = DistributedMetalWorldExecutor::new(
                self.layout.global_num_qubits,
                self.layout.world_size,
                DistributedMetalConfig {
                    // World path always executes exact remote semantics.
                    remote_execution_mode: ShardRemoteExecutionMode::Skip,
                    ..self.config.clone()
                },
            )?;
            let world_run = world.execute_partitioned(gates)?;
            let mut metrics = world_run.metrics;
            metrics.shard_remote_world_emulation_used = true;
            let local_probabilities = world_run
                .per_rank_probabilities
                .get(self.layout.rank)
                .cloned()
                .ok_or_else(|| {
                    format!(
                        "world emulation missing rank {} probabilities",
                        self.layout.rank
                    )
                })?;
            return Ok(DistributedMetalRunResult {
                local_probabilities,
                metrics,
                layout: self.layout.clone(),
            });
        }

        let mut backend =
            LocalBackendState::new(self.layout.local_num_qubits, self.config.strict_gpu_only)?;
        let mut metrics = DistributedMetalMetrics::default();
        let t0 = Instant::now();

        let mut local_batch: Vec<Gate> = Vec::new();
        let mut all_partners = BTreeSet::new();

        for gate in gates {
            if self.layout.gate_is_local(gate) {
                metrics.local_gates += 1;
                local_batch.push(self.layout.localize_gate(gate)?);
                if local_batch.len() >= self.config.max_local_batch.max(1) {
                    flush_local_batch(&mut backend, &mut local_batch, &mut metrics)?;
                }
            } else {
                flush_local_batch(&mut backend, &mut local_batch, &mut metrics)?;
                metrics.remote_gates += 1;

                let partners = self.layout.communication_partners_for_gate(gate);
                for &p in &partners {
                    all_partners.insert(p);
                }
                if self.apply_remote_gate_without_exchange(&mut backend, gate)? {
                    metrics.remote_gates_executed += 1;
                    metrics.remote_gates_no_exchange += 1;
                } else {
                    metrics.remote_gates_exchange_required += 1;
                    let applied = self.exchange_remote_gate_state(
                        gate,
                        &partners,
                        &mut metrics,
                        &mut backend,
                    );

                    if applied {
                        metrics.remote_gates_executed += 1;
                    } else if self.config.fail_on_remote_gates {
                        return Err(format!(
                            "remote gate requires exchange but could not be applied: {:?}",
                            gate
                        ));
                    } else {
                        metrics.remote_gates_skipped += 1;
                    }
                }
            }
        }
        flush_local_batch(&mut backend, &mut local_batch, &mut metrics)?;

        metrics.communication_partners = all_partners.len();
        metrics.elapsed_ms = t0.elapsed().as_secs_f64() * 1000.0;

        Ok(DistributedMetalRunResult {
            local_probabilities: backend.probabilities(),
            metrics,
            layout: self.layout.clone(),
        })
    }

    /// Exchange shard boundary amplitudes with peer ranks for a remote gate.
    ///
    /// For each communication partner identified by `ShardLayout::communication_partners_for_gate`:
    /// 1. Identify the shard-domain qubit(s) that connect this rank to the partner.
    /// 2. Pack the real local amplitudes into a send buffer.
    /// 3. Exchange via MPI send/recv (or barrier when real MPI is unavailable).
    /// 4. Unpack received amplitudes and apply the gate transformation across
    ///    the shard boundary using `apply_pairwise_shard_gate`.
    /// 5. Write the modified local state back to the backend and send modified
    ///    partner amplitudes back so the partner can update its state.
    ///
    /// When the `distributed` feature is not enabled, this falls back to a
    /// barrier synchronization (the existing behavior for single-process shards).
    ///
    /// Returns `true` if the gate was successfully applied via exchange.
    fn exchange_remote_gate_state(
        &self,
        gate: &Gate,
        partners: &[usize],
        metrics: &mut DistributedMetalMetrics,
        backend: &mut LocalBackendState,
    ) -> bool {
        if partners.is_empty() {
            return false;
        }

        #[cfg(feature = "distributed")]
        {
            use mpi::traits::*;

            let mut any_applied = false;

            for &partner in partners {
                let diff_bits = self.layout.rank ^ partner;
                if diff_bits == 0 {
                    continue;
                }

                let shard_bit = diff_bits.trailing_zeros() as usize;
                if shard_bit >= self.layout.shard_bits {
                    continue;
                }

                let local_size = self.layout.local_amplitudes;

                // Pack REAL local amplitudes into f64 pairs (re, im).
                let local_state = match backend.read_state() {
                    Ok(s) => s,
                    Err(_) => continue,
                };
                let mut send_buf = Vec::with_capacity(local_size * 2);
                for amp in local_state.iter().take(local_size) {
                    send_buf.push(amp.re);
                    send_buf.push(amp.im);
                }

                let mut recv_buf = vec![0.0f64; local_size * 2];

                // Exchange: lower rank sends first to avoid deadlock.
                if self.layout.rank < partner {
                    self.mpi.send(&send_buf, partner);
                    self.mpi.recv(&mut recv_buf, partner);
                } else {
                    self.mpi.recv(&mut recv_buf, partner);
                    self.mpi.send(&send_buf, partner);
                }

                // Reconstruct partner amplitudes from received buffer.
                let mut partner_amps: Vec<C64> = recv_buf
                    .chunks_exact(2)
                    .map(|c| C64::new(c[0], c[1]))
                    .collect();

                // Apply the gate across the shard boundary using the same
                // pairwise logic as the world executor. The lower rank holds
                // shard0 (shard bit = 0) and the higher rank holds shard1
                // (shard bit = 1).
                let mut my_amps = local_state;
                let applied = if self.layout.rank < partner {
                    apply_pairwise_shard_gate(
                        gate,
                        self.layout.local_num_qubits,
                        &mut my_amps,
                        &mut partner_amps,
                    )
                } else {
                    apply_pairwise_shard_gate(
                        gate,
                        self.layout.local_num_qubits,
                        &mut partner_amps,
                        &mut my_amps,
                    )
                };

                match applied {
                    Ok(true) => {
                        // Write modified local state back to the backend.
                        let _ = backend.set_state(&my_amps);

                        // Send modified partner amplitudes back so the partner
                        // can update its local state.
                        let mut modified_buf = Vec::with_capacity(local_size * 2);
                        for amp in partner_amps.iter() {
                            modified_buf.push(amp.re);
                            modified_buf.push(amp.im);
                        }

                        let mut return_buf = vec![0.0f64; local_size * 2];
                        if self.layout.rank < partner {
                            self.mpi.send(&modified_buf, partner);
                            self.mpi.recv(&mut return_buf, partner);
                        } else {
                            self.mpi.recv(&mut return_buf, partner);
                            self.mpi.send(&modified_buf, partner);
                        }

                        // Apply the returned modified state from the partner.
                        let returned_amps: Vec<C64> = return_buf
                            .chunks_exact(2)
                            .map(|c| C64::new(c[0], c[1]))
                            .collect();
                        let _ = backend.set_state(&returned_amps);

                        any_applied = true;
                    }
                    _ => {
                        // Gate type not supported by pairwise path; send
                        // zero-modified buffers back to keep the protocol
                        // symmetric (partner expects a return exchange).
                        let zero_buf = vec![0.0f64; local_size * 2];
                        let mut _discard = vec![0.0f64; local_size * 2];
                        if self.layout.rank < partner {
                            self.mpi.send(&zero_buf, partner);
                            self.mpi.recv(&mut _discard, partner);
                        } else {
                            self.mpi.recv(&mut _discard, partner);
                            self.mpi.send(&zero_buf, partner);
                        }
                    }
                }

                metrics.communication_events += 1;
            }

            return any_applied;
        }

        #[cfg(not(feature = "distributed"))]
        {
            // Without real MPI, fall back to barrier synchronization.
            // Real remote gate execution for single-process multi-rank
            // simulation is handled by EmulatedWorldExact mode.
            let _ = gate;
            let _ = backend;
            self.mpi.barrier();
            metrics.communication_events += 1;
            false
        }
    }

    fn apply_remote_gate_without_exchange(
        &self,
        backend: &mut LocalBackendState,
        gate: &Gate,
    ) -> Result<bool, String> {
        apply_remote_gate_without_exchange_for_layout(&self.layout, backend, gate)
    }
}

fn flush_local_batch(
    backend: &mut LocalBackendState,
    local_batch: &mut Vec<Gate>,
    metrics: &mut DistributedMetalMetrics,
) -> Result<(), String> {
    if local_batch.is_empty() {
        return Ok(());
    }
    backend.run_batch(local_batch)?;
    metrics.local_batches += 1;
    local_batch.clear();
    Ok(())
}

fn apply_remote_gate_without_exchange_for_layout(
    layout: &ShardLayout,
    backend: &mut LocalBackendState,
    gate: &Gate,
) -> Result<bool, String> {
    let shard_controls: Vec<usize> = gate
        .controls
        .iter()
        .copied()
        .filter(|&q| !layout.is_local_qubit(q))
        .collect();
    let local_controls: Vec<usize> = gate
        .controls
        .iter()
        .copied()
        .filter(|&q| layout.is_local_qubit(q))
        .collect();
    let local_targets: Vec<usize> = gate
        .targets
        .iter()
        .copied()
        .filter(|&q| layout.is_local_qubit(q))
        .collect();
    let shard_targets: Vec<usize> = gate
        .targets
        .iter()
        .copied()
        .filter(|&q| !layout.is_local_qubit(q))
        .collect();

    // Case A: rank-conditioned local gate (all targets local, some controls shard).
    if shard_targets.is_empty() && !shard_controls.is_empty() {
        if !shard_controls_are_active(layout, &shard_controls) {
            // Condition false -> gate acts as identity on this shard.
            return Ok(true);
        }

        if let Some(reduced) = reduce_remote_controlled_gate(gate, &local_controls, &local_targets)?
        {
            backend.run_single_gate(&reduced)?;
            return Ok(true);
        }
    }

    // Case B: single-qubit diagonal gate on a shard qubit.
    // This can be applied without exchange because each shard corresponds to a fixed
    // computational-basis value for the shard-domain qubit.
    if gate.controls.is_empty() && shard_targets.len() == 1 && local_targets.is_empty() {
        if let Some(phase) = shard_single_qubit_diagonal_phase(
            &gate.gate_type,
            rank_shard_bit(layout, shard_targets[0]),
        ) {
            backend.scale_all(phase)?;
            return Ok(true);
        }
    }

    // Case C: local-control with shard-target for diagonal controlled gates.
    // With a fixed shard bit per rank, this reduces to a rank-conditioned
    // diagonal local gate on the local control, requiring no exchange.
    if shard_controls.is_empty()
        && shard_targets.len() == 1
        && local_targets.is_empty()
        && local_controls.len() == 1
        && gate.controls.len() == 1
    {
        let local_control = local_controls[0];
        let target_shard_active = rank_shard_bit(layout, shard_targets[0]);

        match &gate.gate_type {
            GateType::CZ => {
                if target_shard_active {
                    backend.run_single_gate(&Gate::z(local_control))?;
                }
                return Ok(true);
            }
            GateType::CRz(theta) => {
                let phase = if target_shard_active {
                    *theta / 2.0
                } else {
                    -*theta / 2.0
                };
                backend.run_single_gate(&Gate::phase(local_control, phase))?;
                return Ok(true);
            }
            GateType::CR(theta) => {
                if target_shard_active {
                    backend.run_single_gate(&Gate::phase(local_control, *theta))?;
                }
                return Ok(true);
            }
            _ => {}
        }
    }

    // Case D: shard-control with shard-target for diagonal controlled gates.
    // Both qubit values are fixed per rank, so this reduces to a global phase.
    if shard_controls.len() == 1
        && shard_targets.len() == 1
        && local_controls.is_empty()
        && local_targets.is_empty()
        && gate.controls.len() == 1
    {
        if let Some(phase) = shard_controlled_diagonal_phase(
            &gate.gate_type,
            rank_shard_bit(layout, shard_controls[0]),
            rank_shard_bit(layout, shard_targets[0]),
        ) {
            backend.scale_all(phase)?;
            return Ok(true);
        }
    }

    Ok(false)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum RemoteExecutionKind {
    NoExchange,
    Pairwise,
    GlobalFallback,
}

fn ceil_div(a: usize, b: usize) -> usize {
    if a == 0 {
        0
    } else {
        (a + b - 1) / b
    }
}

fn build_scheduling_plan(
    layout: &ShardLayout,
    gates: &[Gate],
    config: &DistributedMetalConfig,
    model: &CommunicationCostModel,
) -> SchedulingPlan {
    let mut local_count = 0usize;
    let mut pairwise_count = 0usize;
    let mut fallback_count = 0usize;
    let mut no_exchange_count = 0usize;

    for gate in gates {
        if layout.gate_is_local(gate) {
            local_count += 1;
            continue;
        }
        match classify_remote_execution(layout, gate) {
            RemoteExecutionKind::NoExchange => no_exchange_count += 1,
            RemoteExecutionKind::Pairwise => pairwise_count += 1,
            RemoteExecutionKind::GlobalFallback => fallback_count += 1,
        }
    }

    let total = gates.len().max(1);
    let remote_count = pairwise_count + fallback_count + no_exchange_count;
    let remote_density = remote_count as f64 / total as f64;

    let local_cap = config.max_local_batch.max(1);
    let mut local_candidates = vec![16usize, 32, 64, 128, 256, 512];
    local_candidates.retain(|c| *c <= local_cap);
    if !local_candidates.contains(&local_cap) {
        local_candidates.push(local_cap);
    }
    local_candidates.sort_unstable();
    local_candidates.dedup();

    let mut best_local = local_cap;
    let mut best_local_score = f64::INFINITY;
    for c in &local_candidates {
        let batches = ceil_div(local_count, *c).max(1) as f64;
        let score = batches * 0.06
            + remote_density * (local_count as f64 / *c as f64) * model.latency_cost * 0.35;
        if score < best_local_score {
            best_local_score = score;
            best_local = *c;
        }
    }

    let mut fallback_candidates = vec![8usize, 16, 32, 64, 128];
    if config.strict_gpu_only {
        // Prefer larger batches when strict mode routes fallback through global GPU.
        fallback_candidates.retain(|c| *c >= 16);
    }

    let mut best_fallback = 32usize;
    let mut best_comm_score = f64::INFINITY;
    if fallback_count == 0 {
        best_comm_score = pairwise_count as f64 * model.bandwidth_cost * model.pairwise_weight;
    }
    for c in fallback_candidates {
        if fallback_count == 0 {
            break;
        }
        let fallback_batches = ceil_div(fallback_count, c) as f64;
        let pairwise_events = pairwise_count as f64; // pairwise paths communicate per partner group
        let score = fallback_batches * model.latency_cost * model.fallback_weight
            + fallback_count as f64 * model.bandwidth_cost * model.fallback_weight
            + pairwise_events * model.bandwidth_cost * model.pairwise_weight
            + remote_density * model.latency_cost * (1.0 / c as f64);
        if score < best_comm_score {
            best_comm_score = score;
            best_fallback = c;
        }
    }

    SchedulingPlan {
        local_batch_limit: best_local.max(1),
        global_fallback_batch_limit: best_fallback.max(1),
        estimated_comm_cost: if best_comm_score.is_finite() {
            best_comm_score
        } else {
            0.0
        },
    }
}

fn classify_remote_execution(layout: &ShardLayout, gate: &Gate) -> RemoteExecutionKind {
    if remote_gate_no_exchange_supported(layout, gate) {
        return RemoteExecutionKind::NoExchange;
    }
    if pairwise_remote_gate_supported(layout, gate) {
        return RemoteExecutionKind::Pairwise;
    }
    RemoteExecutionKind::GlobalFallback
}

fn remote_gate_no_exchange_supported(layout: &ShardLayout, gate: &Gate) -> bool {
    let shard_controls: Vec<usize> = gate
        .controls
        .iter()
        .copied()
        .filter(|&q| !layout.is_local_qubit(q))
        .collect();
    let local_controls: Vec<usize> = gate
        .controls
        .iter()
        .copied()
        .filter(|&q| layout.is_local_qubit(q))
        .collect();
    let local_targets: Vec<usize> = gate
        .targets
        .iter()
        .copied()
        .filter(|&q| layout.is_local_qubit(q))
        .collect();
    let shard_targets: Vec<usize> = gate
        .targets
        .iter()
        .copied()
        .filter(|&q| !layout.is_local_qubit(q))
        .collect();

    if shard_targets.is_empty() && !shard_controls.is_empty() {
        if matches!(
            reduce_remote_controlled_gate(gate, &local_controls, &local_targets),
            Ok(Some(_))
        ) {
            return true;
        }
    }

    if gate.controls.is_empty() && shard_targets.len() == 1 && local_targets.is_empty() {
        if shard_single_qubit_diagonal_phase(&gate.gate_type, false).is_some() {
            return true;
        }
    }

    if shard_controls.is_empty()
        && shard_targets.len() == 1
        && local_targets.is_empty()
        && local_controls.len() == 1
        && gate.controls.len() == 1
    {
        if matches!(
            &gate.gate_type,
            GateType::CZ | GateType::CRz(_) | GateType::CR(_)
        ) {
            return true;
        }
    }

    if shard_controls.len() == 1
        && shard_targets.len() == 1
        && local_controls.is_empty()
        && local_targets.is_empty()
        && gate.controls.len() == 1
    {
        if matches!(
            &gate.gate_type,
            GateType::CZ | GateType::CR(_) | GateType::CRz(_)
        ) {
            return true;
        }
    }

    false
}

fn pairwise_remote_gate_supported(layout: &ShardLayout, gate: &Gate) -> bool {
    let shard_controls: Vec<usize> = gate
        .controls
        .iter()
        .copied()
        .filter(|&q| !layout.is_local_qubit(q))
        .collect();
    let shard_targets: Vec<usize> = gate
        .targets
        .iter()
        .copied()
        .filter(|&q| !layout.is_local_qubit(q))
        .collect();
    let local_controls: Vec<usize> = gate
        .controls
        .iter()
        .copied()
        .filter(|&q| layout.is_local_qubit(q))
        .collect();
    let local_targets: Vec<usize> = gate
        .targets
        .iter()
        .copied()
        .filter(|&q| layout.is_local_qubit(q))
        .collect();

    if shard_controls.is_empty()
        && gate.controls.is_empty()
        && local_controls.is_empty()
        && local_targets.is_empty()
        && shard_targets.len() == 2
        && gate.targets.len() == 2
    {
        let bit_a = shard_targets[0].saturating_sub(layout.local_num_qubits);
        let bit_b = shard_targets[1].saturating_sub(layout.local_num_qubits);
        if bit_a >= layout.shard_bits || bit_b >= layout.shard_bits || bit_a == bit_b {
            return false;
        }
        return matches!(&gate.gate_type, GateType::SWAP | GateType::ISWAP);
    }

    // Mixed local+shard target: SWAP/iSWAP with one local target and one shard target.
    if shard_controls.is_empty()
        && local_controls.is_empty()
        && shard_targets.len() == 1
        && local_targets.len() == 1
        && gate.targets.len() == 2
        && gate.controls.is_empty()
    {
        let shard_bit = shard_targets[0].saturating_sub(layout.local_num_qubits);
        if shard_bit < layout.shard_bits {
            return matches!(&gate.gate_type, GateType::SWAP | GateType::ISWAP);
        }
    }

    // Toffoli with two local controls and one shard target.
    if shard_controls.is_empty()
        && shard_targets.len() == 1
        && local_controls.len() == 2
        && local_targets.is_empty()
        && gate.targets.len() == 1
        && gate.controls.len() == 2
        && gate.controls.iter().all(|&c| c < layout.local_num_qubits)
    {
        let shard_bit = shard_targets[0].saturating_sub(layout.local_num_qubits);
        if shard_bit < layout.shard_bits {
            return matches!(&gate.gate_type, GateType::Toffoli);
        }
    }

    // Single shard-target fast path.
    if shard_controls.is_empty() && shard_targets.len() == 1 {
        if gate.targets.len() != 1
            || gate.targets[0] != shard_targets[0]
            || !local_targets.is_empty()
            || gate.controls.iter().any(|&q| q >= layout.local_num_qubits)
        {
            return false;
        }

        let shard_bit = shard_targets[0].saturating_sub(layout.local_num_qubits);
        if shard_bit >= layout.shard_bits {
            return false;
        }

        return match &gate.gate_type {
            GateType::H
            | GateType::X
            | GateType::Y
            | GateType::Rx(_)
            | GateType::Ry(_)
            | GateType::Rz(_) => local_controls.is_empty(),
            GateType::CNOT
            | GateType::CZ
            | GateType::CR(_)
            | GateType::CRx(_)
            | GateType::CRy(_)
            | GateType::CRz(_) => local_controls.len() == 1,
            _ => false,
        };
    }

    // Shard-control -> shard-target CNOT permutation fast path.
    if matches!(&gate.gate_type, GateType::CNOT)
        && gate.controls.len() == 1
        && gate.targets.len() == 1
        && shard_controls.len() == 1
        && shard_targets.len() == 1
        && local_controls.is_empty()
        && local_targets.is_empty()
    {
        let control_bit = shard_controls[0].saturating_sub(layout.local_num_qubits);
        let target_bit = shard_targets[0].saturating_sub(layout.local_num_qubits);
        if control_bit < layout.shard_bits
            && target_bit < layout.shard_bits
            && control_bit != target_bit
        {
            return true;
        }
    }

    false
}

fn shard_controls_are_active(layout: &ShardLayout, shard_controls: &[usize]) -> bool {
    shard_controls.iter().all(|&q| rank_shard_bit(layout, q))
}

fn rank_shard_bit(layout: &ShardLayout, global_qubit: usize) -> bool {
    if layout.is_local_qubit(global_qubit) {
        return false;
    }
    let shard_bit = global_qubit - layout.local_num_qubits;
    if shard_bit >= layout.shard_bits {
        return false;
    }
    ((layout.rank >> shard_bit) & 1) == 1
}

fn mutable_pair<T>(items: &mut [T], i: usize, j: usize) -> Result<(&mut T, &mut T), String> {
    if i == j || i >= items.len() || j >= items.len() {
        return Err(format!(
            "invalid mutable pair indices ({}, {}) for len {}",
            i,
            j,
            items.len()
        ));
    }
    let (lo, hi, swap) = if i < j { (i, j, false) } else { (j, i, true) };
    let (left, right) = items.split_at_mut(hi);
    let a = &mut left[lo];
    let b = &mut right[0];
    if swap {
        Ok((b, a))
    } else {
        Ok((a, b))
    }
}

fn apply_pairwise_shard_gate(
    gate: &Gate,
    local_num_qubits: usize,
    shard0: &mut [C64],
    shard1: &mut [C64],
) -> Result<bool, String> {
    if shard0.len() != shard1.len() {
        return Err(format!(
            "pairwise shard length mismatch: {} vs {}",
            shard0.len(),
            shard1.len()
        ));
    }

    // ---------------------------------------------------------------
    // Extended pairwise gates: SWAP, iSWAP (one local + one shard target),
    // and Toffoli (two local controls + one shard target).
    // These have different target/control layouts than the single-target
    // path below, so we handle them first.
    // ---------------------------------------------------------------

    // SWAP(q_local, q_shard): exchange the bit value of q_local with the shard bit.
    // For each index where q_local is set, swap shard0[idx] <-> shard1[idx ^ (1<<q_local)].
    if matches!(&gate.gate_type, GateType::SWAP)
        && gate.targets.len() == 2
        && gate.controls.is_empty()
    {
        let local_targets: Vec<usize> = gate
            .targets
            .iter()
            .copied()
            .filter(|&q| q < local_num_qubits)
            .collect();
        let shard_targets: Vec<usize> = gate
            .targets
            .iter()
            .copied()
            .filter(|&q| q >= local_num_qubits)
            .collect();
        if local_targets.len() == 1 && shard_targets.len() == 1 {
            let local_bit = local_targets[0];
            let mask = 1usize << local_bit;
            for idx in 0..shard0.len() {
                if (idx & mask) != 0 {
                    // idx has local bit set; partner = idx with local bit cleared
                    let partner = idx ^ mask;
                    // Swap: |shard=0, local=1> <-> |shard=1, local=0>
                    std::mem::swap(&mut shard0[idx], &mut shard1[partner]);
                }
            }
            return Ok(true);
        }
    }

    // iSWAP(q_local, q_shard): like SWAP but with i*phase on off-diagonal blocks.
    // |01> -> i|10>, |10> -> i|01>.
    if matches!(&gate.gate_type, GateType::ISWAP)
        && gate.targets.len() == 2
        && gate.controls.is_empty()
    {
        let local_targets: Vec<usize> = gate
            .targets
            .iter()
            .copied()
            .filter(|&q| q < local_num_qubits)
            .collect();
        let shard_targets: Vec<usize> = gate
            .targets
            .iter()
            .copied()
            .filter(|&q| q >= local_num_qubits)
            .collect();
        if local_targets.len() == 1 && shard_targets.len() == 1 {
            let local_bit = local_targets[0];
            let mask = 1usize << local_bit;
            let i_phase = C64::new(0.0, 1.0);
            for idx in 0..shard0.len() {
                if (idx & mask) != 0 {
                    let partner = idx ^ mask;
                    // iSWAP: |shard=0, local=1> -> i*|shard=1, local=0>
                    //         |shard=1, local=0> -> i*|shard=0, local=1>
                    let a = shard0[idx];
                    let b = shard1[partner];
                    shard0[idx] = i_phase * b;
                    shard1[partner] = i_phase * a;
                }
            }
            return Ok(true);
        }
    }

    // Toffoli(ctrl1_local, ctrl2_local, shard_target): 2 local controls, 1 shard target.
    if matches!(&gate.gate_type, GateType::Toffoli)
        && gate.targets.len() == 1
        && gate.controls.len() == 2
        && gate.targets[0] >= local_num_qubits
        && gate.controls.iter().all(|&c| c < local_num_qubits)
    {
        let controls_active =
            |idx: usize| gate.controls.iter().all(|&c| ((idx >> c) & 1) == 1);
        for idx in 0..shard0.len() {
            if controls_active(idx) {
                std::mem::swap(&mut shard0[idx], &mut shard1[idx]);
            }
        }
        return Ok(true);
    }

    // ---------------------------------------------------------------
    // Original single shard-target path (H, X, Y, rotations, controlled gates).
    // ---------------------------------------------------------------

    if gate.targets.len() != 1 || gate.targets[0] < local_num_qubits {
        return Ok(false);
    }
    if gate.controls.iter().any(|&q| q >= local_num_qubits) {
        return Ok(false);
    }

    let local_controls = &gate.controls;
    let controls_active = |idx: usize| local_controls.iter().all(|&c| ((idx >> c) & 1) == 1);
    let half = 0.5f64;
    let inv_sqrt2 = (half).sqrt();

    match &gate.gate_type {
        GateType::H => {
            if !local_controls.is_empty() {
                return Ok(false);
            }
            for idx in 0..shard0.len() {
                let a = shard0[idx];
                let b = shard1[idx];
                shard0[idx] = C64::new((a.re + b.re) * inv_sqrt2, (a.im + b.im) * inv_sqrt2);
                shard1[idx] = C64::new((a.re - b.re) * inv_sqrt2, (a.im - b.im) * inv_sqrt2);
            }
            Ok(true)
        }
        GateType::X => {
            if !local_controls.is_empty() {
                return Ok(false);
            }
            for idx in 0..shard0.len() {
                std::mem::swap(&mut shard0[idx], &mut shard1[idx]);
            }
            Ok(true)
        }
        GateType::Y => {
            if !local_controls.is_empty() {
                return Ok(false);
            }
            for idx in 0..shard0.len() {
                let a = shard0[idx];
                let b = shard1[idx];
                shard0[idx] = C64::new(b.im, -b.re); // -i * b
                shard1[idx] = C64::new(-a.im, a.re); // i * a
            }
            Ok(true)
        }
        GateType::Rx(theta) => {
            if !local_controls.is_empty() {
                return Ok(false);
            }
            let c = (theta / 2.0).cos();
            let s = (theta / 2.0).sin();
            let minus_i_s = C64::new(0.0, -s);
            for idx in 0..shard0.len() {
                let a = shard0[idx];
                let b = shard1[idx];
                shard0[idx] = C64::new(c, 0.0) * a + minus_i_s * b;
                shard1[idx] = minus_i_s * a + C64::new(c, 0.0) * b;
            }
            Ok(true)
        }
        GateType::Ry(theta) => {
            if !local_controls.is_empty() {
                return Ok(false);
            }
            let c = (theta / 2.0).cos();
            let s = (theta / 2.0).sin();
            for idx in 0..shard0.len() {
                let a = shard0[idx];
                let b = shard1[idx];
                shard0[idx] = C64::new(c, 0.0) * a + C64::new(-s, 0.0) * b;
                shard1[idx] = C64::new(s, 0.0) * a + C64::new(c, 0.0) * b;
            }
            Ok(true)
        }
        GateType::Rz(theta) => {
            if !local_controls.is_empty() {
                return Ok(false);
            }
            let e0 = C64::new((-theta / 2.0).cos(), (-theta / 2.0).sin());
            let e1 = C64::new((theta / 2.0).cos(), (theta / 2.0).sin());
            for idx in 0..shard0.len() {
                shard0[idx] *= e0;
                shard1[idx] *= e1;
            }
            Ok(true)
        }
        GateType::CNOT => {
            if local_controls.len() != 1 {
                return Ok(false);
            }
            for idx in 0..shard0.len() {
                if controls_active(idx) {
                    std::mem::swap(&mut shard0[idx], &mut shard1[idx]);
                }
            }
            Ok(true)
        }
        GateType::CZ => {
            if local_controls.len() != 1 {
                return Ok(false);
            }
            for idx in 0..shard1.len() {
                if controls_active(idx) {
                    shard1[idx] = -shard1[idx];
                }
            }
            Ok(true)
        }
        GateType::CR(theta) => {
            if local_controls.len() != 1 {
                return Ok(false);
            }
            let phase = C64::new(theta.cos(), theta.sin());
            for idx in 0..shard1.len() {
                if controls_active(idx) {
                    shard1[idx] *= phase;
                }
            }
            Ok(true)
        }
        GateType::CRx(theta) => {
            if local_controls.len() != 1 {
                return Ok(false);
            }
            let c = (theta / 2.0).cos();
            let s = (theta / 2.0).sin();
            let minus_i_s = C64::new(0.0, -s);
            for idx in 0..shard0.len() {
                if controls_active(idx) {
                    let a = shard0[idx];
                    let b = shard1[idx];
                    shard0[idx] = C64::new(c, 0.0) * a + minus_i_s * b;
                    shard1[idx] = minus_i_s * a + C64::new(c, 0.0) * b;
                }
            }
            Ok(true)
        }
        GateType::CRy(theta) => {
            if local_controls.len() != 1 {
                return Ok(false);
            }
            let c = (theta / 2.0).cos();
            let s = (theta / 2.0).sin();
            for idx in 0..shard0.len() {
                if controls_active(idx) {
                    let a = shard0[idx];
                    let b = shard1[idx];
                    shard0[idx] = C64::new(c, 0.0) * a + C64::new(-s, 0.0) * b;
                    shard1[idx] = C64::new(s, 0.0) * a + C64::new(c, 0.0) * b;
                }
            }
            Ok(true)
        }
        GateType::CRz(theta) => {
            if local_controls.len() != 1 {
                return Ok(false);
            }
            let e0 = C64::new((-theta / 2.0).cos(), (-theta / 2.0).sin());
            let e1 = C64::new((theta / 2.0).cos(), (theta / 2.0).sin());
            for idx in 0..shard0.len() {
                if controls_active(idx) {
                    shard0[idx] *= e0;
                    shard1[idx] *= e1;
                }
            }
            Ok(true)
        }
        _ => Ok(false),
    }
}

enum LocalBackendState {
    Metal(MetalSimulator),
    Cpu(QuantumState),
}

impl LocalBackendState {
    fn new(local_num_qubits: usize, strict_gpu_only: bool) -> Result<Self, String> {
        match MetalSimulator::new(local_num_qubits) {
            Ok(sim) => Ok(Self::Metal(sim)),
            Err(e) if strict_gpu_only => Err(format!("Metal backend unavailable: {}", e)),
            Err(_) => Ok(Self::Cpu(QuantumState::new(local_num_qubits))),
        }
    }

    fn run_batch(&mut self, gates: &[Gate]) -> Result<(), String> {
        match self {
            Self::Metal(sim) => {
                sim.run_circuit(gates);
                Ok(())
            }
            Self::Cpu(state) => {
                for gate in gates {
                    apply_gate_cpu(state, gate)?;
                }
                Ok(())
            }
        }
    }

    fn run_single_gate(&mut self, gate: &Gate) -> Result<(), String> {
        match self {
            Self::Metal(sim) => {
                sim.run_circuit(std::slice::from_ref(gate));
                Ok(())
            }
            Self::Cpu(state) => apply_gate_cpu(state, gate),
        }
    }

    fn probabilities(&self) -> Vec<f64> {
        match self {
            Self::Metal(sim) => sim.probabilities().into_iter().map(|p| p as f64).collect(),
            Self::Cpu(state) => state.probabilities(),
        }
    }

    fn read_state(&self) -> Result<Vec<C64>, String> {
        match self {
            Self::Metal(sim) => Ok(sim.read_state()),
            Self::Cpu(state) => Ok(state.amplitudes_ref().to_vec()),
        }
    }

    fn set_state(&mut self, amplitudes: &[C64]) -> Result<(), String> {
        match self {
            Self::Metal(sim) => sim.write_state(amplitudes),
            Self::Cpu(state) => {
                if amplitudes.len() != state.dim {
                    return Err(format!(
                        "state length mismatch: got {}, expected {}",
                        amplitudes.len(),
                        state.dim
                    ));
                }
                state.amplitudes_mut().copy_from_slice(amplitudes);
                Ok(())
            }
        }
    }

    fn scale_all(&mut self, factor: C64) -> Result<(), String> {
        match self {
            Self::Metal(sim) => {
                let mut amplitudes = sim.read_state();
                for amp in &mut amplitudes {
                    *amp *= factor;
                }
                sim.write_state(&amplitudes)
            }
            Self::Cpu(state) => {
                for amp in state.amplitudes_mut() {
                    *amp *= factor;
                }
                Ok(())
            }
        }
    }
}

fn reduce_remote_controlled_gate(
    gate: &Gate,
    local_controls: &[usize],
    local_targets: &[usize],
) -> Result<Option<Gate>, String> {
    // All handled cases require fully local targets.
    if local_targets.len() != gate.targets.len() {
        return Ok(None);
    }

    match &gate.gate_type {
        GateType::CNOT => {
            // shard-control CNOT(local target) -> X(local target)
            if gate.controls.len() == 1 && local_controls.is_empty() && local_targets.len() == 1 {
                return Ok(Some(Gate::x(local_targets[0])));
            }
            // two-control decomposition not applicable for CNOT.
            Ok(None)
        }
        GateType::CZ => {
            // shard-control CZ(local target) -> Z(local target)
            if gate.controls.len() == 1 && local_controls.is_empty() && local_targets.len() == 1 {
                return Ok(Some(Gate::z(local_targets[0])));
            }
            Ok(None)
        }
        GateType::CRz(theta) => {
            if gate.controls.len() == 1 && local_controls.is_empty() && local_targets.len() == 1 {
                return Ok(Some(Gate::rz(local_targets[0], *theta)));
            }
            Ok(None)
        }
        GateType::CRx(theta) => {
            if gate.controls.len() == 1 && local_controls.is_empty() && local_targets.len() == 1 {
                return Ok(Some(Gate::rx(local_targets[0], *theta)));
            }
            Ok(None)
        }
        GateType::CRy(theta) => {
            if gate.controls.len() == 1 && local_controls.is_empty() && local_targets.len() == 1 {
                return Ok(Some(Gate::ry(local_targets[0], *theta)));
            }
            Ok(None)
        }
        GateType::Toffoli => {
            // One local + one shard control -> local CNOT.
            if gate.controls.len() == 2 && local_controls.len() == 1 && local_targets.len() == 1 {
                return Ok(Some(Gate::cnot(local_controls[0], local_targets[0])));
            }
            // Two shard controls -> local X.
            if gate.controls.len() == 2 && local_controls.is_empty() && local_targets.len() == 1 {
                return Ok(Some(Gate::x(local_targets[0])));
            }
            Ok(None)
        }
        GateType::CCZ => {
            // One local + one shard control -> local CZ.
            if gate.controls.len() == 2 && local_controls.len() == 1 && local_targets.len() == 1 {
                return Ok(Some(Gate::cz(local_controls[0], local_targets[0])));
            }
            // Two shard controls -> local Z.
            if gate.controls.len() == 2 && local_controls.is_empty() && local_targets.len() == 1 {
                return Ok(Some(Gate::z(local_targets[0])));
            }
            Ok(None)
        }
        _ => Ok(None),
    }
}

fn shard_single_qubit_diagonal_phase(gate_type: &GateType, target_active: bool) -> Option<C64> {
    match gate_type {
        GateType::Z => {
            if target_active {
                Some(C64::new(-1.0, 0.0))
            } else {
                Some(C64::new(1.0, 0.0))
            }
        }
        GateType::S => {
            if target_active {
                Some(C64::new(0.0, 1.0))
            } else {
                Some(C64::new(1.0, 0.0))
            }
        }
        GateType::T => {
            if target_active {
                let angle = std::f64::consts::FRAC_PI_4;
                Some(C64::new(angle.cos(), angle.sin()))
            } else {
                Some(C64::new(1.0, 0.0))
            }
        }
        GateType::Phase(theta) => {
            if target_active {
                Some(C64::new(theta.cos(), theta.sin()))
            } else {
                Some(C64::new(1.0, 0.0))
            }
        }
        GateType::Rz(theta) => {
            let angle = if target_active {
                *theta / 2.0
            } else {
                -*theta / 2.0
            };
            Some(C64::new(angle.cos(), angle.sin()))
        }
        _ => None,
    }
}

fn shard_controlled_diagonal_phase(
    gate_type: &GateType,
    control_active: bool,
    target_active: bool,
) -> Option<C64> {
    match gate_type {
        GateType::CZ => {
            if control_active && target_active {
                Some(C64::new(-1.0, 0.0))
            } else {
                Some(C64::new(1.0, 0.0))
            }
        }
        GateType::CR(theta) => {
            if control_active && target_active {
                Some(C64::new(theta.cos(), theta.sin()))
            } else {
                Some(C64::new(1.0, 0.0))
            }
        }
        GateType::CRz(theta) => {
            if !control_active {
                return Some(C64::new(1.0, 0.0));
            }
            let angle = if target_active {
                *theta / 2.0
            } else {
                -*theta / 2.0
            };
            Some(C64::new(angle.cos(), angle.sin()))
        }
        _ => None,
    }
}

fn apply_gate_cpu(state: &mut QuantumState, gate: &Gate) -> Result<(), String> {
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
        GateType::CR(theta) => {
            GateOperations::cphase(state, gate.controls[0], gate.targets[0], *theta)
        }
        GateType::SX => GateOperations::sx(state, gate.targets[0]),
        GateType::Phase(theta) => GateOperations::phase(state, gate.targets[0], *theta),
        GateType::ISWAP => GateOperations::iswap(state, gate.targets[0], gate.targets[1]),
        GateType::CCZ => {
            GateOperations::ccz(state, gate.controls[0], gate.controls[1], gate.targets[0])
        }
        _ => {
            return Err(format!(
                "unsupported CPU gate in distributed skeleton: {:?}",
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
    fn test_layout_basic() {
        let layout = ShardLayout::for_rank(10, 4, 1).expect("layout");
        assert_eq!(layout.shard_bits, 2);
        assert_eq!(layout.local_num_qubits, 8);
        assert_eq!(layout.local_amplitudes, 256);
        assert_eq!(layout.global_start, 256);
        assert_eq!(layout.global_end, 512);
    }

    #[test]
    fn test_gate_locality() {
        let layout = ShardLayout::for_rank(6, 4, 0).expect("layout"); // local qubits = 4
        assert!(layout.gate_is_local(&Gate::h(0)));
        assert!(layout.gate_is_local(&Gate::cnot(1, 2)));
        assert!(!layout.gate_is_local(&Gate::h(5)));
        assert!(!layout.gate_is_local(&Gate::cnot(1, 5)));
    }

    #[test]
    fn test_partner_derivation() {
        let layout = ShardLayout::for_rank(6, 4, 0).expect("layout"); // shard bits: q4,q5
        let partners = layout.communication_partners_for_gate(&Gate::h(5));
        assert_eq!(partners, vec![2]);
        let partners2 = layout.communication_partners_for_gate(&Gate::cnot(4, 5));
        assert_eq!(partners2, vec![1, 2]);
    }

    #[test]
    fn test_single_rank_smoke() {
        let exec = DistributedMetalShardExecutor::new(
            3,
            MPICommunicator { rank: 0, size: 1 },
            DistributedMetalConfig::default(),
        )
        .expect("executor");

        let gates = vec![Gate::h(0), Gate::cnot(0, 1), Gate::rz(2, 0.2)];
        let result = exec.execute_partitioned(&gates).expect("run");
        let total: f64 = result.local_probabilities.iter().sum();
        assert!(total > 0.9 && total < 1.1);
        assert_eq!(result.metrics.remote_gates, 0);
    }

    #[test]
    fn test_remote_gate_is_counted_when_not_failing() {
        let exec = DistributedMetalShardExecutor::new(
            4,
            MPICommunicator { rank: 0, size: 4 },
            DistributedMetalConfig {
                strict_gpu_only: false,
                fail_on_remote_gates: false,
                remote_execution_mode: ShardRemoteExecutionMode::Skip,
                max_local_batch: 32,
            },
        )
        .expect("executor");

        let gates = vec![Gate::h(3)];
        let result = exec.execute_partitioned(&gates).expect("run");
        assert_eq!(result.metrics.remote_gates, 1);
        assert_eq!(result.metrics.remote_gates_skipped, 1);
        assert_eq!(result.metrics.remote_gates_exchange_required, 1);
        assert!(result.metrics.communication_events >= 1);
    }

    #[test]
    fn test_remote_control_can_reduce_to_local_gate() {
        // global=4, world=2 -> local qubits = 3, shard qubit = q3
        let exec = DistributedMetalShardExecutor::new(
            4,
            MPICommunicator { rank: 1, size: 2 }, // shard bit is 1
            DistributedMetalConfig::default(),
        )
        .expect("executor");

        // Control on shard qubit q3, target local q0: should reduce to local X(q0) on rank=1.
        let gates = vec![Gate::cnot(3, 0)];
        let result = exec.execute_partitioned(&gates).expect("run");
        assert_eq!(result.metrics.remote_gates, 1);
        assert_eq!(result.metrics.remote_gates_executed, 1);
        assert_eq!(result.metrics.remote_gates_no_exchange, 1);
        assert_eq!(result.metrics.remote_gates_skipped, 0);
    }

    #[test]
    fn test_remote_diagonal_shard_gate_no_exchange() {
        // global=4, world=4 -> local qubits = 2, shard qubits = q2,q3.
        let exec = DistributedMetalShardExecutor::new(
            4,
            MPICommunicator { rank: 0, size: 4 },
            DistributedMetalConfig::default(),
        )
        .expect("executor");

        let gates = vec![Gate::z(3)];
        let result = exec.execute_partitioned(&gates).expect("run");
        assert_eq!(result.metrics.remote_gates, 1);
        assert_eq!(result.metrics.remote_gates_executed, 1);
        assert_eq!(result.metrics.remote_gates_no_exchange, 1);
        assert_eq!(result.metrics.remote_gates_exchange_required, 0);
        assert_eq!(result.metrics.remote_gates_skipped, 0);
    }

    #[test]
    fn test_remote_local_control_shard_target_cz_no_exchange() {
        // global=4, world=4 -> local qubits q0,q1 and shard qubits q2,q3.
        let exec = DistributedMetalShardExecutor::new(
            4,
            MPICommunicator { rank: 2, size: 4 }, // q3=1, q2=0
            DistributedMetalConfig::default(),
        )
        .expect("executor");

        let gates = vec![Gate::cz(1, 3)];
        let result = exec.execute_partitioned(&gates).expect("run");
        assert_eq!(result.metrics.remote_gates, 1);
        assert_eq!(result.metrics.remote_gates_no_exchange, 1);
        assert_eq!(result.metrics.remote_gates_exchange_required, 0);
        assert_eq!(result.metrics.remote_gates_skipped, 0);
    }

    #[test]
    fn test_world_executor_initialization_is_globally_normalized() {
        let mut exec = DistributedMetalWorldExecutor::new(4, 4, DistributedMetalConfig::default())
            .expect("executor");
        let result = exec.execute_partitioned(&[]).expect("run");

        let global_sum: f64 = result.global_probabilities.iter().sum();
        assert!((global_sum - 1.0).abs() < 1e-9);

        for (rank, probs) in result.per_rank_probabilities.iter().enumerate() {
            let rank_sum: f64 = probs.iter().sum();
            if rank == 0 {
                assert!((rank_sum - 1.0).abs() < 1e-9);
            } else {
                assert!(rank_sum.abs() < 1e-9);
            }
        }
    }

    #[test]
    fn test_world_executor_executes_remote_gate_via_exchange() {
        let mut exec = DistributedMetalWorldExecutor::new(4, 4, DistributedMetalConfig::default())
            .expect("executor");
        let gates = vec![Gate::h(3)];
        let result = exec.execute_partitioned(&gates).expect("run");

        assert_eq!(result.metrics.remote_gates, 1);
        assert_eq!(result.metrics.remote_gates_exchange_required, 1);
        assert_eq!(result.metrics.remote_gates_executed, 1);
        assert_eq!(result.metrics.remote_gates_skipped, 0);
        assert!(result.metrics.communication_events >= 1);
        assert_eq!(result.metrics.remote_gates_pairwise_fast_path, 1);
        assert_eq!(result.metrics.remote_gates_global_fallback, 0);

        let expected = {
            let mut state = QuantumState::new(4);
            for gate in &gates {
                apply_gate_cpu(&mut state, gate).expect("reference apply");
            }
            state.probabilities()
        };
        assert_eq!(result.global_probabilities.len(), expected.len());
        for (a, b) in result.global_probabilities.iter().zip(expected.iter()) {
            assert!((a - b).abs() < 1e-6);
        }
    }

    #[test]
    fn test_world_executor_matches_reference_for_mixed_local_remote() {
        let mut exec = DistributedMetalWorldExecutor::new(5, 4, DistributedMetalConfig::default())
            .expect("executor");
        let gates = vec![
            Gate::h(0),
            Gate::cnot(0, 1),
            Gate::h(4),       // shard-domain qubit -> exchange path
            Gate::cnot(4, 2), // reducible remote-control path
            Gate::phase(3, 0.3),
            Gate::cz(1, 4), // exchange path
        ];

        let result = exec.execute_partitioned(&gates).expect("run");
        let expected = {
            let mut state = QuantumState::new(5);
            for gate in &gates {
                apply_gate_cpu(&mut state, gate).expect("reference apply");
            }
            state.probabilities()
        };

        assert_eq!(result.global_probabilities.len(), expected.len());
        for (a, b) in result.global_probabilities.iter().zip(expected.iter()) {
            assert!((a - b).abs() < 1e-6);
        }
        assert_eq!(result.metrics.remote_gates, 4);
        assert!(result.metrics.remote_gates_executed >= 4);
        assert_eq!(result.metrics.remote_gates_skipped, 0);
        assert!(result.metrics.remote_gates_pairwise_fast_path >= 1);
        assert!(result.metrics.remote_gates_no_exchange >= 1);
    }

    #[test]
    fn test_world_executor_cz_local_to_shard_becomes_no_exchange() {
        let mut exec = DistributedMetalWorldExecutor::new(5, 4, DistributedMetalConfig::default())
            .expect("executor");
        let gates = vec![Gate::h(1), Gate::cz(1, 4)];
        let result = exec.execute_partitioned(&gates).expect("run");

        let expected = {
            let mut state = QuantumState::new(5);
            for gate in &gates {
                apply_gate_cpu(&mut state, gate).expect("reference apply");
            }
            state.probabilities()
        };

        for (a, b) in result.global_probabilities.iter().zip(expected.iter()) {
            assert!((a - b).abs() < 1e-6);
        }
        assert_eq!(result.metrics.remote_gates, 1);
        assert_eq!(result.metrics.remote_gates_no_exchange, 1);
        assert_eq!(result.metrics.remote_gates_exchange_required, 0);
    }

    #[test]
    fn test_world_executor_remote_diagonal_shard_gate_affects_future_interference() {
        let mut exec = DistributedMetalWorldExecutor::new(4, 4, DistributedMetalConfig::default())
            .expect("executor");
        let gates = vec![Gate::h(3), Gate::rz(3, 0.7), Gate::h(3)];

        let result = exec.execute_partitioned(&gates).expect("run");
        let expected = {
            let mut state = QuantumState::new(4);
            for gate in &gates {
                apply_gate_cpu(&mut state, gate).expect("reference apply");
            }
            state.probabilities()
        };

        for (a, b) in result.global_probabilities.iter().zip(expected.iter()) {
            assert!((a - b).abs() < 1e-6);
        }
        assert!(result.metrics.remote_gates_no_exchange >= 1);
    }

    #[test]
    fn test_world_executor_shard_to_shard_cnot_uses_pairwise_fast_path() {
        let mut exec = DistributedMetalWorldExecutor::new(4, 4, DistributedMetalConfig::default())
            .expect("executor");
        // q2,q3 are shard-domain qubits for global=4, world=4.
        let gates = vec![Gate::h(2), Gate::cnot(2, 3)];

        let result = exec.execute_partitioned(&gates).expect("run");
        let expected = {
            let mut state = QuantumState::new(4);
            for gate in &gates {
                apply_gate_cpu(&mut state, gate).expect("reference apply");
            }
            state.probabilities()
        };

        for (a, b) in result.global_probabilities.iter().zip(expected.iter()) {
            assert!((a - b).abs() < 1e-6);
        }
        assert_eq!(result.metrics.remote_gates, 2);
        assert_eq!(result.metrics.remote_gates_global_fallback, 0);
        assert!(result.metrics.remote_gates_pairwise_fast_path >= 2);
    }

    #[test]
    fn test_world_executor_shard_to_shard_swap_uses_pairwise_fast_path() {
        let mut exec = DistributedMetalWorldExecutor::new(4, 4, DistributedMetalConfig::default())
            .expect("executor");
        // q2,q3 are shard-domain qubits for global=4, world=4.
        let gates = vec![Gate::h(2), Gate::x(3), Gate::swap(2, 3)];

        let result = exec.execute_partitioned(&gates).expect("run");
        let expected = {
            let mut state = QuantumState::new(4);
            for gate in &gates {
                apply_gate_cpu(&mut state, gate).expect("reference apply");
            }
            state.probabilities()
        };

        for (a, b) in result.global_probabilities.iter().zip(expected.iter()) {
            assert!((a - b).abs() < 1e-6);
        }
        assert_eq!(result.metrics.remote_gates, 3);
        assert_eq!(result.metrics.remote_gates_global_fallback, 0);
        assert!(result.metrics.remote_gates_pairwise_fast_path >= 3);
    }

    #[test]
    fn test_world_executor_shard_to_shard_iswap_uses_pairwise_fast_path() {
        let mut exec = DistributedMetalWorldExecutor::new(4, 4, DistributedMetalConfig::default())
            .expect("executor");
        // Include post-iSWAP interference so phase handling affects probabilities.
        let gates = vec![
            Gate::h(2),
            Gate::h(3),
            Gate::iswap(2, 3),
            Gate::h(2),
            Gate::h(3),
        ];

        let result = exec.execute_partitioned(&gates).expect("run");
        let expected = {
            let mut state = QuantumState::new(4);
            for gate in &gates {
                apply_gate_cpu(&mut state, gate).expect("reference apply");
            }
            state.probabilities()
        };

        for (a, b) in result.global_probabilities.iter().zip(expected.iter()) {
            assert!((a - b).abs() < 1e-6);
        }
        assert_eq!(result.metrics.remote_gates, 5);
        assert_eq!(result.metrics.remote_gates_global_fallback, 0);
        assert!(result.metrics.remote_gates_pairwise_fast_path >= 5);
    }

    #[test]
    fn test_world_executor_batches_consecutive_global_fallback_remote_gates() {
        let mut exec = DistributedMetalWorldExecutor::new(4, 4, DistributedMetalConfig::default())
            .expect("executor");
        // CCZ between local controls and shard target uses global fallback (not pairwise-handled).
        let gates = vec![Gate::ccz(0, 1, 2), Gate::ccz(0, 1, 2), Gate::h(0)];

        let result = exec.execute_partitioned(&gates).expect("run");
        let expected = {
            let mut state = QuantumState::new(4);
            for gate in &gates {
                apply_gate_cpu(&mut state, gate).expect("reference apply");
            }
            state.probabilities()
        };

        for (a, b) in result.global_probabilities.iter().zip(expected.iter()) {
            assert!((a - b).abs() < 1e-6);
        }
        assert_eq!(result.metrics.remote_gates, 2);
        assert_eq!(result.metrics.remote_gates_global_fallback, 2);
        assert_eq!(result.metrics.remote_gates_global_fallback_batches, 1);
        assert_eq!(result.metrics.remote_gates_executed, 2);
        assert_eq!(result.metrics.communication_events, 1);
    }

    #[test]
    fn test_world_executor_flushes_global_fallback_batch_on_local_boundary() {
        let mut exec = DistributedMetalWorldExecutor::new(4, 4, DistributedMetalConfig::default())
            .expect("executor");
        // Local gate between two fallback gates should force two fallback batches.
        let gates = vec![Gate::ccz(0, 1, 2), Gate::h(0), Gate::ccz(0, 1, 2)];

        let result = exec.execute_partitioned(&gates).expect("run");
        let expected = {
            let mut state = QuantumState::new(4);
            for gate in &gates {
                apply_gate_cpu(&mut state, gate).expect("reference apply");
            }
            state.probabilities()
        };

        for (a, b) in result.global_probabilities.iter().zip(expected.iter()) {
            assert!((a - b).abs() < 1e-6);
        }
        assert_eq!(result.metrics.remote_gates, 2);
        assert_eq!(result.metrics.remote_gates_global_fallback, 2);
        assert_eq!(result.metrics.remote_gates_global_fallback_batches, 2);
        assert_eq!(result.metrics.remote_gates_executed, 2);
        assert_eq!(result.metrics.communication_events, 2);
    }

    #[test]
    fn test_scheduler_prefers_large_fallback_batch_for_fallback_heavy_workload() {
        let mut exec = DistributedMetalWorldExecutor::new(4, 4, DistributedMetalConfig::default())
            .expect("executor");
        let gates = vec![Gate::ccz(0, 1, 2); 12];
        let result = exec.execute_partitioned(&gates).expect("run");

        assert_eq!(result.metrics.remote_gates, 12);
        assert_eq!(result.metrics.remote_gates_global_fallback, 12);
        assert!(
            result.metrics.adaptive_global_fallback_batch_limit >= 64,
            "expected large fallback batch, got {}",
            result.metrics.adaptive_global_fallback_batch_limit
        );
    }

    #[test]
    fn test_scheduler_uses_default_fallback_limit_when_no_fallback_needed() {
        let mut exec = DistributedMetalWorldExecutor::new(4, 4, DistributedMetalConfig::default())
            .expect("executor");
        let gates = vec![Gate::h(3), Gate::h(3)];
        let result = exec.execute_partitioned(&gates).expect("run");

        assert_eq!(result.metrics.remote_gates_global_fallback, 0);
        assert_eq!(result.metrics.adaptive_global_fallback_batch_limit, 32);
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn test_world_executor_strict_gpu_only_executes_global_fallback_via_gpu() {
        let mut exec = DistributedMetalWorldExecutor::new(
            4,
            4,
            DistributedMetalConfig {
                strict_gpu_only: true,
                fail_on_remote_gates: false,
                remote_execution_mode: ShardRemoteExecutionMode::Skip,
                max_local_batch: 32,
            },
        )
        .expect("executor");
        let gates = vec![Gate::ccz(0, 1, 2)];
        let result = exec
            .execute_partitioned(&gates)
            .expect("strict GPU-only fallback should execute via global GPU path");
        assert_eq!(result.metrics.remote_gates, 1);
        assert_eq!(result.metrics.remote_gates_global_fallback, 1);
        assert_eq!(result.metrics.remote_gates_global_fallback_batches, 1);
        assert_eq!(result.metrics.remote_gates_executed, 1);
        assert!(result.metrics.communication_events >= 1);
    }

    #[test]
    fn test_shard_executor_emulated_world_exact_executes_remote_gate() {
        let cfg = DistributedMetalConfig {
            remote_execution_mode: ShardRemoteExecutionMode::EmulatedWorldExact,
            ..DistributedMetalConfig::default()
        };
        let exec =
            DistributedMetalShardExecutor::new(4, MPICommunicator { rank: 2, size: 4 }, cfg)
                .expect("executor");

        let gates = vec![Gate::h(3), Gate::phase(0, 0.2)];
        let out = exec.execute_partitioned(&gates).expect("run");

        let mut world = DistributedMetalWorldExecutor::new(4, 4, DistributedMetalConfig::default())
            .expect("world");
        let world_out = world.execute_partitioned(&gates).expect("world run");
        let expected = &world_out.per_rank_probabilities[2];

        assert!(out.metrics.shard_remote_world_emulation_used);
        assert_eq!(out.metrics.remote_gates, 1);
        assert_eq!(out.metrics.remote_gates_skipped, 0);
        assert_eq!(out.metrics.remote_gates_executed, 1);
        assert_eq!(out.local_probabilities.len(), expected.len());
        for (a, b) in out.local_probabilities.iter().zip(expected.iter()) {
            assert!((a - b).abs() < 1e-6);
        }
    }

    // =================================================================
    // AA2c: Real state exchange + gate application tests
    // =================================================================

    #[test]
    fn test_exchange_real_state_two_ranks() {
        // Verify that exchange_remote_gate_state with real backend state
        // is correctly wired through execute_partitioned on the shard executor.
        // With world_size=2, q2 is the shard qubit, q0-q1 are local.
        // Without the distributed feature, exchange falls back to barrier,
        // so we test via the world executor which exercises the same
        // apply_pairwise_shard_gate logic used by the exchange path.
        let mut exec = DistributedMetalWorldExecutor::new(3, 2, DistributedMetalConfig::default())
            .expect("executor");
        // H on shard-domain qubit q2 requires exchange between the two ranks.
        let gates = vec![Gate::h(2)];
        let result = exec.execute_partitioned(&gates).expect("run");

        // Reference: single-process simulation
        let expected = {
            let mut state = QuantumState::new(3);
            apply_gate_cpu(&mut state, &Gate::h(2)).expect("ref");
            state.probabilities()
        };

        for (a, b) in result.global_probabilities.iter().zip(expected.iter()) {
            assert!(
                (a - b).abs() < 1e-6,
                "probability mismatch: {} vs {}",
                a,
                b
            );
        }
        assert_eq!(result.metrics.remote_gates, 1);
        assert_eq!(result.metrics.remote_gates_executed, 1);
        assert_eq!(result.metrics.remote_gates_skipped, 0);
    }

    #[test]
    fn test_remote_cnot_via_exchange() {
        // CNOT with control on shard qubit, target on shard qubit.
        // This exercises the pairwise shard gate CNOT path.
        let mut exec = DistributedMetalWorldExecutor::new(4, 4, DistributedMetalConfig::default())
            .expect("executor");
        // Prepare: H on q2 (shard), then CNOT(q2, q3) both shard qubits.
        let gates = vec![Gate::h(2), Gate::cnot(2, 3)];
        let result = exec.execute_partitioned(&gates).expect("run");

        let expected = {
            let mut state = QuantumState::new(4);
            for gate in &gates {
                apply_gate_cpu(&mut state, gate).expect("ref");
            }
            state.probabilities()
        };

        for (a, b) in result.global_probabilities.iter().zip(expected.iter()) {
            assert!(
                (a - b).abs() < 1e-6,
                "CNOT exchange probability mismatch: {} vs {}",
                a,
                b
            );
        }
        // Both gates are remote (shard-domain qubits in 4-rank layout).
        assert!(result.metrics.remote_gates >= 2);
        assert_eq!(result.metrics.remote_gates_skipped, 0);
    }

    #[test]
    fn test_exchange_preserves_norm() {
        // After applying remote gates via exchange, the total probability
        // must remain 1.0 (unitarity preservation).
        let mut exec = DistributedMetalWorldExecutor::new(4, 4, DistributedMetalConfig::default())
            .expect("executor");
        let gates = vec![
            Gate::h(2),
            Gate::h(3),
            Gate::cnot(2, 3),
            Gate::rx(2, 0.7),
            Gate::ry(3, 1.3),
        ];
        let result = exec.execute_partitioned(&gates).expect("run");

        let total: f64 = result.global_probabilities.iter().sum();
        assert!(
            (total - 1.0).abs() < 1e-6,
            "norm not preserved after exchange: total = {}",
            total
        );
    }

    #[test]
    fn test_pairwise_shard_gate_cnot_correctness() {
        // Directly test apply_pairwise_shard_gate for a CNOT with local
        // control and shard-domain target.
        // Setup: 2 local qubits, shard qubit = q2.
        // shard0 = amplitudes where shard bit = 0
        // shard1 = amplitudes where shard bit = 1
        // Start with |100> = shard0=[0,0,0,0], shard1=[1,0,0,0]
        // CNOT(q0_local_control, q2_shard_target) with control active
        // should swap shard0[idx] <-> shard1[idx] when control bit is set.
        let local_num_qubits = 2;
        let size = 1 << local_num_qubits; // 4
        let mut shard0 = vec![C64::new(0.0, 0.0); size];
        let mut shard1 = vec![C64::new(0.0, 0.0); size];
        // State: |q2=1, q1=0, q0=0> -> shard1[0] = 1
        shard1[0] = C64::new(1.0, 0.0);

        // CNOT with local control q0, shard target q2.
        let gate = Gate::cnot(0, 2);
        let applied =
            apply_pairwise_shard_gate(&gate, local_num_qubits, &mut shard0, &mut shard1)
                .expect("apply");
        assert!(applied, "CNOT should be supported by pairwise path");

        // After CNOT(control=q0, target=q2):
        // Only indices where q0=1 (i.e., idx & 1 == 1) get swapped.
        // Since our state had shard1[0]=1 and q0=0 at idx=0, nothing swaps.
        // shard1[0] should still be 1.
        assert!(
            (shard1[0].re - 1.0).abs() < 1e-10,
            "CNOT should not swap when control bit is 0"
        );

        // Now test with control active: state |q2=1, q1=0, q0=1> -> shard1[1] = 1
        let mut shard0b = vec![C64::new(0.0, 0.0); size];
        let mut shard1b = vec![C64::new(0.0, 0.0); size];
        shard1b[1] = C64::new(1.0, 0.0); // idx=1 means q0=1

        let applied2 =
            apply_pairwise_shard_gate(&gate, local_num_qubits, &mut shard0b, &mut shard1b)
                .expect("apply2");
        assert!(applied2);

        // CNOT swaps shard0[1] <-> shard1[1] since q0=1.
        // Before: shard0b[1]=0, shard1b[1]=1
        // After:  shard0b[1]=1, shard1b[1]=0
        assert!(
            (shard0b[1].re - 1.0).abs() < 1e-10,
            "CNOT should move amplitude to shard0 when control active"
        );
        assert!(
            shard1b[1].re.abs() < 1e-10,
            "shard1 should be empty after CNOT swap"
        );
    }

    // =================================================================
    // SWAP, Toffoli, iSWAP pairwise shard gate tests
    // =================================================================

    #[test]
    fn test_pairwise_shard_gate_swap_local_shard() {
        // SWAP(q_local=0, q_shard=2) on a 2-local-qubit layout.
        // For each idx where local bit 0 is set, swap shard0[idx] <-> shard1[idx ^ 1].
        let local_num_qubits = 2;
        let size = 1 << local_num_qubits; // 4 amplitudes per shard

        // State: |q2=0, q1=0, q0=1> -> shard0[1] = 1 (shard bit=0, local q0=1)
        let mut shard0 = vec![C64::new(0.0, 0.0); size];
        let mut shard1 = vec![C64::new(0.0, 0.0); size];
        shard0[1] = C64::new(1.0, 0.0);

        let gate = Gate::swap(0, 2);
        let applied =
            apply_pairwise_shard_gate(&gate, local_num_qubits, &mut shard0, &mut shard1)
                .expect("apply");
        assert!(applied, "SWAP should be supported by pairwise path");

        // After SWAP(q0, q2): |shard=0, q0=1> -> |shard=1, q0=0>
        // shard0[idx=1] (q0=1, shard=0) swaps with shard1[idx=0] (q0=0, shard=1)
        assert!(
            shard0[1].re.abs() < 1e-10,
            "shard0[1] should be zero after SWAP"
        );
        assert!(
            (shard1[0].re - 1.0).abs() < 1e-10,
            "shard1[0] should have the amplitude after SWAP"
        );
    }

    #[test]
    fn test_pairwise_shard_gate_swap_matches_reference() {
        // Full end-to-end: SWAP(local, shard) via world executor matches reference.
        let mut exec = DistributedMetalWorldExecutor::new(4, 4, DistributedMetalConfig::default())
            .expect("executor");
        // q2, q3 are shard qubits. SWAP(q0_local, q2_shard)
        let gates = vec![Gate::h(0), Gate::swap(0, 2)];
        let result = exec.execute_partitioned(&gates).expect("run");

        let expected = {
            let mut state = QuantumState::new(4);
            for gate in &gates {
                apply_gate_cpu(&mut state, gate).expect("ref");
            }
            state.probabilities()
        };

        for (a, b) in result.global_probabilities.iter().zip(expected.iter()) {
            assert!(
                (a - b).abs() < 1e-6,
                "SWAP pairwise mismatch: {} vs {}",
                a,
                b
            );
        }
        assert!(result.metrics.remote_gates >= 1);
        assert_eq!(result.metrics.remote_gates_skipped, 0);
    }

    #[test]
    fn test_pairwise_shard_gate_iswap_local_shard() {
        // iSWAP(q_local=0, q_shard=2): for each idx where q0 is set,
        // iSWAP shard0[idx] and shard1[idx ^ 1] with i*phase.
        let local_num_qubits = 2;
        let size = 1 << local_num_qubits;

        let mut shard0 = vec![C64::new(0.0, 0.0); size];
        let mut shard1 = vec![C64::new(0.0, 0.0); size];
        // State: |shard=0, q0=1> -> shard0[1] = 1
        shard0[1] = C64::new(1.0, 0.0);

        let gate = Gate::iswap(0, 2);
        let applied =
            apply_pairwise_shard_gate(&gate, local_num_qubits, &mut shard0, &mut shard1)
                .expect("apply");
        assert!(applied, "iSWAP should be supported by pairwise path");

        // After iSWAP: |shard=0, q0=1> -> i*|shard=1, q0=0>
        // shard0[1] -> i * shard1_old[0], shard1[0] -> i * shard0_old[1]
        assert!(
            shard0[1].norm_sqr() < 1e-10,
            "shard0[1] should be zero after iSWAP"
        );
        assert!(
            (shard1[0] - C64::new(0.0, 1.0)).norm_sqr() < 1e-10,
            "shard1[0] should be i after iSWAP, got {:?}",
            shard1[0]
        );
    }

    #[test]
    fn test_pairwise_shard_gate_iswap_matches_reference() {
        let mut exec = DistributedMetalWorldExecutor::new(4, 4, DistributedMetalConfig::default())
            .expect("executor");
        // iSWAP between local and shard qubit, with interference
        let gates = vec![
            Gate::h(0),
            Gate::h(2),
            Gate::iswap(0, 2),
            Gate::h(0),
            Gate::h(2),
        ];
        let result = exec.execute_partitioned(&gates).expect("run");

        let expected = {
            let mut state = QuantumState::new(4);
            for gate in &gates {
                apply_gate_cpu(&mut state, gate).expect("ref");
            }
            state.probabilities()
        };

        for (a, b) in result.global_probabilities.iter().zip(expected.iter()) {
            assert!(
                (a - b).abs() < 1e-6,
                "iSWAP pairwise mismatch: {} vs {}",
                a,
                b
            );
        }
        assert!(result.metrics.remote_gates >= 1);
        assert_eq!(result.metrics.remote_gates_skipped, 0);
    }

    #[test]
    fn test_pairwise_shard_gate_toffoli_local_controls() {
        // Toffoli(ctrl0_local, ctrl1_local, target_shard_q2)
        // When both controls are active, flip the shard qubit.
        let local_num_qubits = 2;
        let size = 1 << local_num_qubits;

        // Test 1: Both controls active (q0=1, q1=1 -> idx=3)
        let mut shard0 = vec![C64::new(0.0, 0.0); size];
        let mut shard1 = vec![C64::new(0.0, 0.0); size];
        // State: |q2=0, q1=1, q0=1> -> shard0[3] = 1 (shard bit=0)
        shard0[3] = C64::new(1.0, 0.0);

        let gate = Gate::toffoli(0, 1, 2);
        let applied =
            apply_pairwise_shard_gate(&gate, local_num_qubits, &mut shard0, &mut shard1)
                .expect("apply");
        assert!(applied, "Toffoli should be supported by pairwise path");

        // After Toffoli: idx=3 (both controls active) -> shard0[3] <-> shard1[3]
        assert!(
            shard0[3].re.abs() < 1e-10,
            "shard0[3] should be zero after Toffoli flip"
        );
        assert!(
            (shard1[3].re - 1.0).abs() < 1e-10,
            "shard1[3] should have the amplitude"
        );

        // Test 2: Only one control active (q0=1, q1=0 -> idx=1): no flip
        let mut shard0b = vec![C64::new(0.0, 0.0); size];
        let mut shard1b = vec![C64::new(0.0, 0.0); size];
        shard0b[1] = C64::new(1.0, 0.0); // idx=1 means q0=1, q1=0

        let applied2 =
            apply_pairwise_shard_gate(&gate, local_num_qubits, &mut shard0b, &mut shard1b)
                .expect("apply2");
        assert!(applied2);

        // Should NOT flip because only q0 is active
        assert!(
            (shard0b[1].re - 1.0).abs() < 1e-10,
            "shard0[1] should be unchanged (only one control active)"
        );
        assert!(
            shard1b[1].re.abs() < 1e-10,
            "shard1[1] should remain zero"
        );
    }

    #[test]
    fn test_pairwise_shard_gate_toffoli_matches_reference() {
        let mut exec = DistributedMetalWorldExecutor::new(4, 4, DistributedMetalConfig::default())
            .expect("executor");
        // q2, q3 are shard. Toffoli(q0_local, q1_local, q2_shard)
        let gates = vec![Gate::h(0), Gate::h(1), Gate::toffoli(0, 1, 2)];
        let result = exec.execute_partitioned(&gates).expect("run");

        let expected = {
            let mut state = QuantumState::new(4);
            for gate in &gates {
                apply_gate_cpu(&mut state, gate).expect("ref");
            }
            state.probabilities()
        };

        for (a, b) in result.global_probabilities.iter().zip(expected.iter()) {
            assert!(
                (a - b).abs() < 1e-6,
                "Toffoli pairwise mismatch: {} vs {}",
                a,
                b
            );
        }
        assert!(result.metrics.remote_gates >= 1);
        assert_eq!(result.metrics.remote_gates_skipped, 0);
    }

    #[test]
    fn test_pairwise_shard_gate_swap_preserves_norm() {
        // Verify unitarity: norm is preserved after SWAP across shards.
        let mut exec = DistributedMetalWorldExecutor::new(4, 4, DistributedMetalConfig::default())
            .expect("executor");
        let gates = vec![
            Gate::h(0),
            Gate::h(1),
            Gate::swap(0, 2),
            Gate::cnot(0, 1),
            Gate::swap(1, 3),
        ];
        let result = exec.execute_partitioned(&gates).expect("run");
        let total: f64 = result.global_probabilities.iter().sum();
        assert!(
            (total - 1.0).abs() < 1e-6,
            "norm not preserved after SWAP exchange: {}",
            total
        );
    }
}
