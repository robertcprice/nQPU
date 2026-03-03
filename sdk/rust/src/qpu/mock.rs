//! Mock QPU provider for testing without real hardware credentials.

use async_trait::async_trait;
use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use std::time::Duration;

use crate::qpu::error::QPUError;
use crate::qpu::job::*;
use crate::qpu::provider::{QPUJob, QPUProvider};
use crate::qpu::validation::CircuitValidator;
use crate::qpu::QPUCircuit;

/// Mock quantum provider for testing.
///
/// Simulates QPU behavior without making any network calls. Useful for:
/// - Unit testing circuit submission workflows
/// - Testing error handling paths
/// - CI/CD pipelines without credentials
pub struct MockProvider {
    backends: Vec<BackendInfo>,
    simulated_latency: Duration,
    failure_rate: f64,
}

impl MockProvider {
    /// Create a mock provider with default backends.
    pub fn new() -> Self {
        Self {
            backends: vec![
                BackendInfo {
                    name: "mock_5q".into(),
                    provider: "Mock".into(),
                    num_qubits: 5,
                    is_simulator: true,
                    status: BackendStatus::Online,
                    queue_length: 0,
                    basis_gates: vec![
                        "h".into(), "x".into(), "y".into(), "z".into(),
                        "cx".into(), "cz".into(), "rz".into(), "sx".into(),
                    ],
                    coupling_map: Some(vec![
                        (0, 1), (1, 0), (1, 2), (2, 1),
                        (2, 3), (3, 2), (3, 4), (4, 3),
                    ]),
                    max_shots: 100_000,
                    max_depth: Some(1000),
                    avg_gate_error_1q: Some(0.001),
                    avg_gate_error_2q: Some(0.01),
                    avg_readout_error: Some(0.02),
                    avg_t1_us: Some(100.0),
                    avg_t2_us: Some(80.0),
                    metadata: HashMap::new(),
                },
                BackendInfo {
                    name: "mock_27q".into(),
                    provider: "Mock".into(),
                    num_qubits: 27,
                    is_simulator: true,
                    status: BackendStatus::Online,
                    queue_length: 3,
                    basis_gates: vec![
                        "h".into(), "x".into(), "cx".into(), "rz".into(), "sx".into(),
                    ],
                    coupling_map: None, // All-to-all for simplicity
                    max_shots: 100_000,
                    max_depth: Some(500),
                    avg_gate_error_1q: Some(0.0005),
                    avg_gate_error_2q: Some(0.008),
                    avg_readout_error: Some(0.015),
                    avg_t1_us: Some(200.0),
                    avg_t2_us: Some(150.0),
                    metadata: HashMap::new(),
                },
            ],
            simulated_latency: Duration::from_millis(10),
            failure_rate: 0.0,
        }
    }

    /// Create with custom backends.
    pub fn with_backends(backends: Vec<BackendInfo>) -> Self {
        Self {
            backends,
            simulated_latency: Duration::from_millis(10),
            failure_rate: 0.0,
        }
    }

    /// Set simulated latency for API calls.
    pub fn set_latency(&mut self, latency: Duration) {
        self.simulated_latency = latency;
    }

    /// Set failure rate (0.0 - 1.0) for simulating errors.
    pub fn set_failure_rate(&mut self, rate: f64) {
        self.failure_rate = rate.clamp(0.0, 1.0);
    }

    /// Simulate measurement counts for a circuit.
    fn simulate_counts(circuit: &QPUCircuit, shots: usize) -> HashMap<String, usize> {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        let mut counts = HashMap::new();
        let n_bits = circuit.measurements.len().max(1);

        for _ in 0..shots {
            let mut bitstring = String::with_capacity(n_bits);
            for _ in 0..n_bits {
                if rng.gen_bool(0.5) {
                    bitstring.push('1');
                } else {
                    bitstring.push('0');
                }
            }
            *counts.entry(bitstring).or_insert(0) += 1;
        }
        counts
    }
}

impl Default for MockProvider {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl QPUProvider for MockProvider {
    fn name(&self) -> &str {
        "Mock Provider"
    }

    async fn list_backends(&self) -> Result<Vec<BackendInfo>, QPUError> {
        tokio::time::sleep(self.simulated_latency).await;
        Ok(self.backends.clone())
    }

    async fn get_backend(&self, name: &str) -> Result<BackendInfo, QPUError> {
        tokio::time::sleep(self.simulated_latency).await;
        self.backends
            .iter()
            .find(|b| b.name == name)
            .cloned()
            .ok_or_else(|| QPUError::BackendUnavailable(name.into()))
    }

    async fn submit_circuit(
        &self,
        circuit: &QPUCircuit,
        backend: &str,
        config: &JobConfig,
    ) -> Result<Box<dyn QPUJob>, QPUError> {
        tokio::time::sleep(self.simulated_latency).await;

        // Check failure rate
        if self.failure_rate > 0.0 {
            use rand::Rng;
            if rand::thread_rng().gen_bool(self.failure_rate) {
                return Err(QPUError::ExecutionError("Simulated failure".into()));
            }
        }

        let backend_info = self
            .backends
            .iter()
            .find(|b| b.name == backend)
            .ok_or_else(|| QPUError::BackendUnavailable(backend.into()))?;

        // Validate
        let report = self.validate_circuit(circuit, backend_info)?;
        if !report.is_valid {
            return Err(QPUError::ValidationError(
                crate::qpu::error::ValidationError::EmptyCircuit,
            ));
        }

        let counts = Self::simulate_counts(circuit, config.shots);

        Ok(Box::new(MockJob {
            job_id: format!("mock-{}", rand::random::<u32>()),
            backend: backend.into(),
            counts,
            shots: config.shots,
        }))
    }

    async fn estimate_cost(
        &self,
        _circuit: &QPUCircuit,
        backend: &str,
        shots: usize,
    ) -> Result<Option<CostEstimate>, QPUError> {
        Ok(Some(CostEstimate {
            provider: "Mock".into(),
            backend: backend.into(),
            shots,
            estimated_cost_usd: 0.0,
            estimated_queue_time: Some(Duration::from_secs(0)),
            is_free_tier: true,
            details: Some("Mock provider — no real cost".into()),
        }))
    }

    fn validate_circuit(
        &self,
        circuit: &QPUCircuit,
        backend: &BackendInfo,
    ) -> Result<ValidationReport, QPUError> {
        let validator = CircuitValidator::new(backend);
        Ok(validator.validate(circuit))
    }
}

/// Mock job that completes immediately.
struct MockJob {
    job_id: String,
    backend: String,
    counts: HashMap<String, usize>,
    shots: usize,
}

#[async_trait]
impl QPUJob for MockJob {
    fn id(&self) -> &str {
        &self.job_id
    }

    fn backend(&self) -> &str {
        &self.backend
    }

    fn provider(&self) -> &str {
        "Mock"
    }

    async fn status(&self) -> Result<JobStatus, QPUError> {
        Ok(JobStatus::Completed)
    }

    async fn wait_for_completion(
        &self,
        _timeout: Duration,
        _poll_interval: Duration,
    ) -> Result<JobResult, QPUError> {
        Ok(JobResult {
            job_id: self.job_id.clone(),
            backend: self.backend.clone(),
            counts: self.counts.clone(),
            shots: self.shots,
            is_simulator: true,
            execution_time: Some(Duration::from_millis(1)),
            queue_time: Some(Duration::from_millis(0)),
            metadata: HashMap::new(),
        })
    }

    async fn result(&self) -> Result<Option<JobResult>, QPUError> {
        Ok(Some(JobResult {
            job_id: self.job_id.clone(),
            backend: self.backend.clone(),
            counts: self.counts.clone(),
            shots: self.shots,
            is_simulator: true,
            execution_time: Some(Duration::from_millis(1)),
            queue_time: Some(Duration::from_millis(0)),
            metadata: HashMap::new(),
        }))
    }

    async fn cancel(&self) -> Result<(), QPUError> {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // =========================================================================
    // 1. Provider creation and configuration
    // =========================================================================

    #[tokio::test]
    async fn test_mock_provider_name() {
        let provider = MockProvider::new();
        assert_eq!(provider.name(), "Mock Provider");
    }

    #[tokio::test]
    async fn test_mock_provider_default_backends() {
        let provider = MockProvider::new();
        let backends = provider.list_backends().await.unwrap();
        assert_eq!(backends.len(), 2);
        assert_eq!(backends[0].name, "mock_5q");
        assert_eq!(backends[0].num_qubits, 5);
        assert!(backends[0].is_simulator);
        assert_eq!(backends[0].status, BackendStatus::Online);
        assert_eq!(backends[1].name, "mock_27q");
        assert_eq!(backends[1].num_qubits, 27);
    }

    #[tokio::test]
    async fn test_mock_provider_custom_backends() {
        let custom = vec![BackendInfo {
            name: "custom_3q".into(),
            provider: "Custom".into(),
            num_qubits: 3,
            is_simulator: true,
            status: BackendStatus::Online,
            queue_length: 0,
            basis_gates: vec!["h".into(), "cx".into()],
            coupling_map: None,
            max_shots: 8192,
            max_depth: Some(200),
            avg_gate_error_1q: None,
            avg_gate_error_2q: None,
            avg_readout_error: None,
            avg_t1_us: None,
            avg_t2_us: None,
            metadata: HashMap::new(),
        }];
        let provider = MockProvider::with_backends(custom);
        let backends = provider.list_backends().await.unwrap();
        assert_eq!(backends.len(), 1);
        assert_eq!(backends[0].name, "custom_3q");
        assert_eq!(backends[0].num_qubits, 3);
    }

    #[tokio::test]
    async fn test_mock_provider_default_trait() {
        // MockProvider implements Default
        let provider = MockProvider::default();
        let backends = provider.list_backends().await.unwrap();
        assert_eq!(backends.len(), 2);
    }

    // =========================================================================
    // 2. Circuit submission — Bell state end-to-end
    // =========================================================================

    #[tokio::test]
    async fn test_submit_bell_state_full_pipeline() {
        let provider = MockProvider::new();
        let circuit = QPUCircuit::bell_state();
        let config = JobConfig::default();

        // Submit
        let job = provider
            .submit_circuit(&circuit, "mock_5q", &config)
            .await
            .unwrap();

        // Job metadata
        assert!(!job.id().is_empty());
        assert!(job.id().starts_with("mock-"));
        assert_eq!(job.backend(), "mock_5q");
        assert_eq!(job.provider(), "Mock");

        // Status should be Completed immediately for MockJob
        let status = job.status().await.unwrap();
        assert_eq!(status, JobStatus::Completed);
        assert!(status.is_terminal());
        assert!(status.is_success());

        // Wait for completion (should return immediately)
        let result = job
            .wait_for_completion(Duration::from_secs(10), Duration::from_millis(100))
            .await
            .unwrap();

        // Verify result structure
        assert_eq!(result.shots, 1024);
        assert!(result.is_simulator);
        assert!(!result.counts.is_empty());
        assert_eq!(result.backend, "mock_5q");
        assert!(result.execution_time.is_some());
        assert!(result.queue_time.is_some());

        // Verify shot counts sum to total shots
        let total: usize = result.counts.values().sum();
        assert_eq!(total, 1024);

        // All bitstrings should be 2 bits long (2 measurements)
        for bitstring in result.counts.keys() {
            assert_eq!(bitstring.len(), 2, "Bitstring '{}' should be 2 bits", bitstring);
            assert!(
                bitstring.chars().all(|c| c == '0' || c == '1'),
                "Bitstring '{}' should only contain 0s and 1s",
                bitstring
            );
        }
    }

    // =========================================================================
    // 3. Job status polling
    // =========================================================================

    #[tokio::test]
    async fn test_mock_job_status_is_completed() {
        let provider = MockProvider::new();
        let circuit = QPUCircuit::bell_state();
        let config = JobConfig::default();
        let job = provider
            .submit_circuit(&circuit, "mock_5q", &config)
            .await
            .unwrap();

        // MockJob always reports Completed
        let status = job.status().await.unwrap();
        assert_eq!(status, JobStatus::Completed);
        assert!(status.is_terminal());
        assert!(status.is_success());
    }

    // =========================================================================
    // 4. Result retrieval with measurement counts
    // =========================================================================

    #[tokio::test]
    async fn test_result_retrieval_via_result_method() {
        let provider = MockProvider::new();
        let circuit = QPUCircuit::bell_state();
        let config = JobConfig::default();
        let job = provider
            .submit_circuit(&circuit, "mock_5q", &config)
            .await
            .unwrap();

        // Use result() instead of wait_for_completion()
        let result = job.result().await.unwrap();
        assert!(result.is_some());
        let result = result.unwrap();
        assert_eq!(result.shots, 1024);
        assert!(!result.counts.is_empty());

        // Probabilities should sum to ~1.0
        let probs = result.probabilities();
        let prob_sum: f64 = probs.values().sum();
        assert!(
            (prob_sum - 1.0).abs() < 1e-10,
            "Probabilities should sum to 1.0, got {}",
            prob_sum
        );

        // most_frequent should return a valid entry
        let (top_bitstring, top_count) = result.most_frequent().unwrap();
        assert!(!top_bitstring.is_empty());
        assert!(top_count > 0);
        assert!(top_count <= 1024);
    }

    #[tokio::test]
    async fn test_result_custom_shot_count() {
        let provider = MockProvider::new();
        let circuit = QPUCircuit::bell_state();
        let config = JobConfig {
            shots: 4096,
            ..Default::default()
        };
        let job = provider
            .submit_circuit(&circuit, "mock_5q", &config)
            .await
            .unwrap();
        let result = job
            .wait_for_completion(Duration::from_secs(5), Duration::from_millis(50))
            .await
            .unwrap();
        assert_eq!(result.shots, 4096);
        let total: usize = result.counts.values().sum();
        assert_eq!(total, 4096);
    }

    // =========================================================================
    // 5. Job cancellation
    // =========================================================================

    #[tokio::test]
    async fn test_job_cancellation() {
        let provider = MockProvider::new();
        let circuit = QPUCircuit::bell_state();
        let config = JobConfig::default();
        let job = provider
            .submit_circuit(&circuit, "mock_5q", &config)
            .await
            .unwrap();

        // cancel() should succeed without error
        let cancel_result = job.cancel().await;
        assert!(cancel_result.is_ok());
    }

    // =========================================================================
    // 6. Multiple concurrent job submissions
    // =========================================================================

    #[tokio::test]
    async fn test_multiple_concurrent_submissions() {
        let provider = MockProvider::new();
        let circuit = QPUCircuit::bell_state();
        let config = JobConfig::default();

        // Submit 5 jobs concurrently
        let mut handles = Vec::new();
        for i in 0..5 {
            let backend = if i % 2 == 0 { "mock_5q" } else { "mock_27q" };
            let job = provider
                .submit_circuit(&circuit, backend, &config)
                .await
                .unwrap();
            handles.push(job);
        }

        // All should have unique IDs
        let ids: Vec<String> = handles.iter().map(|j| j.id().to_string()).collect();
        let unique_ids: std::collections::HashSet<&String> = ids.iter().collect();
        assert_eq!(
            unique_ids.len(),
            5,
            "All 5 jobs should have unique IDs, got {:?}",
            ids
        );

        // All should complete successfully
        for (i, job) in handles.iter().enumerate() {
            let result = job.result().await.unwrap();
            assert!(result.is_some(), "Job {} should have a result", i);
            let result = result.unwrap();
            assert_eq!(result.shots, 1024);
            let total: usize = result.counts.values().sum();
            assert_eq!(total, 1024);
        }
    }

    // =========================================================================
    // 7. Invalid circuit rejection
    // =========================================================================

    #[tokio::test]
    async fn test_reject_empty_circuit() {
        let provider = MockProvider::new();
        // Circuit with no gates and no measurements
        let circuit = QPUCircuit::new(2, 2);
        let config = JobConfig::default();
        let result = provider
            .submit_circuit(&circuit, "mock_5q", &config)
            .await;
        assert!(result.is_err(), "Empty circuit should be rejected");
    }

    #[tokio::test]
    async fn test_reject_circuit_too_many_qubits() {
        let provider = MockProvider::new();
        // mock_5q only has 5 qubits; circuit needs 10
        let mut circuit = QPUCircuit::new(10, 10);
        circuit.h(0);
        circuit.cx(0, 9);
        circuit.measure_all();
        let config = JobConfig::default();
        let result = provider
            .submit_circuit(&circuit, "mock_5q", &config)
            .await;
        assert!(
            result.is_err(),
            "Circuit requiring 10 qubits should fail on 5-qubit backend"
        );
    }

    #[tokio::test]
    async fn test_circuit_fits_on_larger_backend() {
        let provider = MockProvider::new();
        // Same 10-qubit circuit should fit on mock_27q
        let mut circuit = QPUCircuit::new(10, 10);
        circuit.h(0);
        circuit.cx(0, 1);
        circuit.measure_all();
        let config = JobConfig::default();
        let result = provider
            .submit_circuit(&circuit, "mock_27q", &config)
            .await;
        assert!(
            result.is_ok(),
            "10-qubit circuit should succeed on 27-qubit backend"
        );
    }

    // =========================================================================
    // 8. Provider capabilities / backend info query
    // =========================================================================

    #[tokio::test]
    async fn test_get_backend_by_name() {
        let provider = MockProvider::new();

        let backend = provider.get_backend("mock_5q").await.unwrap();
        assert_eq!(backend.name, "mock_5q");
        assert_eq!(backend.num_qubits, 5);
        assert_eq!(backend.max_shots, 100_000);
        assert!(backend.coupling_map.is_some());
        assert!(!backend.basis_gates.is_empty());
        assert!(backend.basis_gates.contains(&"h".to_string()));
        assert!(backend.basis_gates.contains(&"cx".to_string()));

        let backend27 = provider.get_backend("mock_27q").await.unwrap();
        assert_eq!(backend27.name, "mock_27q");
        assert_eq!(backend27.num_qubits, 27);
        assert_eq!(backend27.queue_length, 3);
        // mock_27q has no coupling map (all-to-all)
        assert!(backend27.coupling_map.is_none());
    }

    #[tokio::test]
    async fn test_get_backend_nonexistent() {
        let provider = MockProvider::new();
        let result = provider.get_backend("does_not_exist").await;
        assert!(result.is_err());
        match result.unwrap_err() {
            QPUError::BackendUnavailable(name) => {
                assert_eq!(name, "does_not_exist");
            }
            other => panic!("Expected BackendUnavailable, got {:?}", other),
        }
    }

    #[tokio::test]
    async fn test_backend_error_rates() {
        let provider = MockProvider::new();
        let backend = provider.get_backend("mock_5q").await.unwrap();
        assert_eq!(backend.avg_gate_error_1q, Some(0.001));
        assert_eq!(backend.avg_gate_error_2q, Some(0.01));
        assert_eq!(backend.avg_readout_error, Some(0.02));
        assert_eq!(backend.avg_t1_us, Some(100.0));
        assert_eq!(backend.avg_t2_us, Some(80.0));
    }

    // =========================================================================
    // 9. Error handling for nonexistent job IDs / invalid backends
    // =========================================================================

    #[tokio::test]
    async fn test_submit_to_nonexistent_backend() {
        let provider = MockProvider::new();
        let circuit = QPUCircuit::bell_state();
        let config = JobConfig::default();
        let result = provider
            .submit_circuit(&circuit, "nonexistent_backend", &config)
            .await;
        assert!(result.is_err());
        match result.err().unwrap() {
            QPUError::BackendUnavailable(name) => {
                assert_eq!(name, "nonexistent_backend");
            }
            other => panic!("Expected BackendUnavailable, got {:?}", other),
        }
    }

    // =========================================================================
    // 10. Cost estimation
    // =========================================================================

    #[tokio::test]
    async fn test_cost_estimate_is_free() {
        let provider = MockProvider::new();
        let circuit = QPUCircuit::bell_state();
        let cost = provider
            .estimate_cost(&circuit, "mock_5q", 1024)
            .await
            .unwrap();
        assert!(cost.is_some());
        let cost = cost.unwrap();
        assert_eq!(cost.estimated_cost_usd, 0.0);
        assert!(cost.is_free_tier);
        assert_eq!(cost.provider, "Mock");
        assert_eq!(cost.backend, "mock_5q");
        assert_eq!(cost.shots, 1024);
        assert!(cost.details.is_some());
    }

    #[tokio::test]
    async fn test_cost_estimate_different_shots() {
        let provider = MockProvider::new();
        let circuit = QPUCircuit::bell_state();
        let cost = provider
            .estimate_cost(&circuit, "mock_27q", 8192)
            .await
            .unwrap()
            .unwrap();
        assert_eq!(cost.shots, 8192);
        assert_eq!(cost.backend, "mock_27q");
    }

    // =========================================================================
    // 11. Validation reports
    // =========================================================================

    #[tokio::test]
    async fn test_validate_valid_circuit() {
        let provider = MockProvider::new();
        let circuit = QPUCircuit::bell_state();
        let backend = provider.get_backend("mock_5q").await.unwrap();
        let report = provider.validate_circuit(&circuit, &backend).unwrap();
        assert!(report.is_valid, "Bell state should be valid on mock_5q");
        assert!(report.errors.is_empty());
        assert!(report.estimated_fidelity.is_some());
        let fidelity = report.estimated_fidelity.unwrap();
        assert!(
            fidelity > 0.0 && fidelity <= 1.0,
            "Fidelity should be in (0, 1], got {}",
            fidelity
        );
    }

    #[tokio::test]
    async fn test_validate_empty_circuit_has_errors() {
        let provider = MockProvider::new();
        let circuit = QPUCircuit::new(2, 2); // no gates, no measurements
        let backend = provider.get_backend("mock_5q").await.unwrap();
        let report = provider.validate_circuit(&circuit, &backend).unwrap();
        assert!(!report.is_valid, "Empty circuit should fail validation");
        assert!(!report.errors.is_empty());
    }

    #[tokio::test]
    async fn test_validate_too_many_qubits_report() {
        let provider = MockProvider::new();
        let mut circuit = QPUCircuit::new(10, 10);
        circuit.h(0);
        circuit.measure_all();
        let backend = provider.get_backend("mock_5q").await.unwrap();
        let report = provider.validate_circuit(&circuit, &backend).unwrap();
        assert!(
            !report.is_valid,
            "10-qubit circuit should fail on 5-qubit backend"
        );
        let has_qubit_error = report
            .errors
            .iter()
            .any(|e| e.contains("qubits"));
        assert!(
            has_qubit_error,
            "Errors should mention qubit count: {:?}",
            report.errors
        );
    }

    #[tokio::test]
    async fn test_validate_unsupported_gate_warning() {
        let provider = MockProvider::new();
        // mock_5q basis: h, x, y, z, cx, cz, rz, sx
        // U3 is NOT in the basis set
        let mut circuit = QPUCircuit::new(2, 2);
        circuit.u3(0, 1.0, 2.0, 3.0);
        circuit.measure_all();
        let backend = provider.get_backend("mock_5q").await.unwrap();
        let report = provider.validate_circuit(&circuit, &backend).unwrap();
        // Should still be valid (unsupported gates produce warnings, not errors)
        assert!(report.is_valid);
        let has_gate_warning = report
            .warnings
            .iter()
            .any(|w| w.contains("transpiled") || w.contains("basis"));
        assert!(
            has_gate_warning,
            "Should warn about out-of-basis gates: {:?}",
            report.warnings
        );
    }

    // =========================================================================
    // 12. GHZ state circuit submission
    // =========================================================================

    #[tokio::test]
    async fn test_submit_ghz_state() {
        let provider = MockProvider::new();
        let circuit = QPUCircuit::ghz_state(4);
        let config = JobConfig {
            shots: 2048,
            ..Default::default()
        };
        let job = provider
            .submit_circuit(&circuit, "mock_5q", &config)
            .await
            .unwrap();
        let result = job
            .wait_for_completion(Duration::from_secs(5), Duration::from_millis(50))
            .await
            .unwrap();

        assert_eq!(result.shots, 2048);
        let total: usize = result.counts.values().sum();
        assert_eq!(total, 2048);

        // All bitstrings should be 4 bits long (4-qubit GHZ)
        for bs in result.counts.keys() {
            assert_eq!(bs.len(), 4);
        }
    }

    // =========================================================================
    // 13. Latency configuration
    // =========================================================================

    #[tokio::test]
    async fn test_simulated_latency() {
        let mut provider = MockProvider::new();
        provider.set_latency(Duration::from_millis(1));
        let start = std::time::Instant::now();
        let _ = provider.list_backends().await.unwrap();
        let elapsed = start.elapsed();
        // Should take at least 1ms due to simulated latency
        assert!(
            elapsed >= Duration::from_millis(1),
            "Expected >= 1ms latency, got {:?}",
            elapsed
        );
    }

    // =========================================================================
    // 14. Connectivity violation warnings
    // =========================================================================

    #[tokio::test]
    async fn test_connectivity_violation_produces_warning() {
        let provider = MockProvider::new();
        // mock_5q coupling: (0,1),(1,0),(1,2),(2,1),(2,3),(3,2),(3,4),(4,3)
        // A cx(0,4) is NOT in the coupling map
        let mut circuit = QPUCircuit::new(5, 5);
        circuit.h(0);
        circuit.cx(0, 4); // violates connectivity
        circuit.measure_all();

        let backend = provider.get_backend("mock_5q").await.unwrap();
        let report = provider.validate_circuit(&circuit, &backend).unwrap();
        // Circuit is still valid (routing will fix it), but should have a warning
        assert!(report.is_valid);
        let has_connectivity_warning = report
            .warnings
            .iter()
            .any(|w| w.contains("connectivity") || w.contains("routing"));
        assert!(
            has_connectivity_warning,
            "Should warn about connectivity violations: {:?}",
            report.warnings
        );
    }

    // =========================================================================
    // 15. JobResult helper methods
    // =========================================================================

    #[tokio::test]
    async fn test_job_result_probabilities() {
        let provider = MockProvider::new();
        let circuit = QPUCircuit::bell_state();
        let config = JobConfig { shots: 10000, ..Default::default() };
        let job = provider
            .submit_circuit(&circuit, "mock_5q", &config)
            .await
            .unwrap();
        let result = job.result().await.unwrap().unwrap();
        let probs = result.probabilities();

        // Each probability should be in [0, 1]
        for (bs, p) in &probs {
            assert!(
                *p >= 0.0 && *p <= 1.0,
                "Probability for '{}' = {} out of range",
                bs,
                p
            );
        }

        // Sum to 1.0
        let sum: f64 = probs.values().sum();
        assert!(
            (sum - 1.0).abs() < 1e-10,
            "Probabilities sum to {}, expected 1.0",
            sum
        );
    }

    #[tokio::test]
    async fn test_job_result_most_frequent() {
        let provider = MockProvider::new();
        let circuit = QPUCircuit::bell_state();
        let config = JobConfig { shots: 100, ..Default::default() };
        let job = provider
            .submit_circuit(&circuit, "mock_5q", &config)
            .await
            .unwrap();
        let result = job.result().await.unwrap().unwrap();
        let (top_bs, top_count) = result.most_frequent().unwrap();
        // The most frequent bitstring should have the highest count
        for (_, count) in &result.counts {
            assert!(top_count >= *count);
        }
        assert!(!top_bs.is_empty());
    }

    // =========================================================================
    // 16. Submit on both backends
    // =========================================================================

    #[tokio::test]
    async fn test_submit_on_both_default_backends() {
        let provider = MockProvider::new();
        let circuit = QPUCircuit::bell_state();
        let config = JobConfig::default();

        let job_5q = provider
            .submit_circuit(&circuit, "mock_5q", &config)
            .await
            .unwrap();
        assert_eq!(job_5q.backend(), "mock_5q");

        let job_27q = provider
            .submit_circuit(&circuit, "mock_27q", &config)
            .await
            .unwrap();
        assert_eq!(job_27q.backend(), "mock_27q");

        // Both should produce valid results
        let r1 = job_5q.result().await.unwrap().unwrap();
        let r2 = job_27q.result().await.unwrap().unwrap();
        assert_eq!(r1.shots, 1024);
        assert_eq!(r2.shots, 1024);
    }
}
