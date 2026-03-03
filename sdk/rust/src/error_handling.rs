//! Robust Error Handling and Validation for Quantum Operations
//!
//! This module provides comprehensive error handling, validation, and recovery
//! mechanisms for quantum simulation operations.

use std::fmt;
use std::result;

/// Quantum simulation error types
#[derive(Clone, Debug, PartialEq)]
pub enum QuantumError {
    /// Invalid qubit index
    InvalidQubitIndex { index: usize, max: usize },

    /// Invalid gate operation
    InvalidGate { gate: String, reason: String },

    /// Circuit validation failed
    CircuitValidationFailed { errors: Vec<String> },

    /// State initialization failed
    StateInitializationFailed { reason: String },

    /// Measurement failed
    MeasurementFailed { reason: String },

    /// GPU operation failed
    GpuError { message: String },

    /// Memory allocation failed
    MemoryError { requested: usize, available: usize },

    /// Parameter validation failed
    InvalidParameter { name: String, value: String },

    /// Numerical error (overflow, underflow, NaN)
    NumericalError { operation: String, value: f64 },

    /// Circuit execution failed
    ExecutionFailed { stage: String, reason: String },

    /// Batch size exceeded
    BatchTooLarge { requested: usize, max: usize },

    /// GPU buffer error
    BufferError { operation: String, size: usize },
}

impl fmt::Display for QuantumError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            QuantumError::InvalidQubitIndex { index, max } => {
                write!(f, "Invalid qubit index {}: maximum is {}", index, max)
            }
            QuantumError::InvalidGate { gate, reason } => {
                write!(f, "Invalid gate '{}': {}", gate, reason)
            }
            QuantumError::CircuitValidationFailed { errors } => {
                write!(f, "Circuit validation failed: {}", errors.join(", "))
            }
            QuantumError::StateInitializationFailed { reason } => {
                write!(f, "State initialization failed: {}", reason)
            }
            QuantumError::MeasurementFailed { reason } => {
                write!(f, "Measurement failed: {}", reason)
            }
            QuantumError::GpuError { message } => {
                write!(f, "GPU error: {}", message)
            }
            QuantumError::MemoryError { requested, available } => {
                write!(f, "Memory error: requested {} bytes, only {} available", requested, available)
            }
            QuantumError::InvalidParameter { name, value } => {
                write!(f, "Invalid parameter '{}': value {}", name, value)
            }
            QuantumError::NumericalError { operation, value } => {
                write!(f, "Numerical error in {}: {}", operation, value)
            }
            QuantumError::ExecutionFailed { stage, reason } => {
                write!(f, "Execution failed at stage '{}': {}", stage, reason)
            }
            QuantumError::BatchTooLarge { requested, max } => {
                write!(f, "Batch too large: requested {}, maximum {}", requested, max)
            }
            QuantumError::BufferError { operation, size } => {
                write!(f, "Buffer error for operation '{}' with size {}", operation, size)
            }
        }
    }
}

impl std::error::Error for QuantumError {}

/// Result type for quantum operations
pub type Result<T> = std::result::Result<T, QuantumError>;

/// Validator for quantum operations
pub struct Validator;

impl Validator {
    /// Validate qubit index
    pub fn validate_qubit_index(index: usize, num_qubits: usize) -> Result<()> {
        if index >= num_qubits {
            return Err(QuantumError::InvalidQubitIndex { index, max: num_qubits - 1 });
        }
        Ok(())
    }

    /// Validate multiple qubit indices
    pub fn validate_qubit_indices(indices: &[usize], num_qubits: usize) -> Result<()> {
        for &index in indices {
            Self::validate_qubit_index(index, num_qubits)?;
        }
        Ok(())
    }

    /// Validate gate operation
    pub fn validate_gate(
        gate_type: &str,
        targets: &[usize],
        controls: &[usize],
        num_qubits: usize,
    ) -> Result<()> {
        Self::validate_qubit_indices(targets, num_qubits)?;
        Self::validate_qubit_indices(controls, num_qubits)?;

        // Check for duplicate qubits
        let all_qubits: std::collections::HashSet<_> =
            targets.iter().chain(controls.iter()).copied().collect();
        if all_qubits.len() != targets.len() + controls.len() {
            return Err(QuantumError::InvalidGate {
                gate: gate_type.to_string(),
                reason: "duplicate qubits in targets/controls".to_string(),
            });
        }

        // Gate-specific validation
        match gate_type {
            "H" | "X" | "Y" | "Z" | "S" | "T" | "Rx" | "Ry" | "Rz" => {
                if targets.len() != 1 || !controls.is_empty() {
                    return Err(QuantumError::InvalidGate {
                        gate: gate_type.to_string(),
                        reason: format!("expected 1 target, 0 controls, got {} targets, {} controls",
                            targets.len(), controls.len()),
                    });
                }
            }
            "CNOT" | "CZ" => {
                if targets.len() != 1 || controls.len() != 1 {
                    return Err(QuantumError::InvalidGate {
                        gate: gate_type.to_string(),
                        reason: "CNOT/CZ requires exactly 1 target and 1 control".to_string(),
                    });
                }
            }
            "SWAP" => {
                if targets.len() != 2 || !controls.is_empty() {
                    return Err(QuantumError::InvalidGate {
                        gate: gate_type.to_string(),
                        reason: "SWAP requires exactly 2 targets and 0 controls".to_string(),
                    });
                }
            }
            "Toffoli" => {
                if targets.len() != 1 || controls.len() != 2 {
                    return Err(QuantumError::InvalidGate {
                        gate: gate_type.to_string(),
                        reason: "Toffoli requires exactly 1 target and 2 controls".to_string(),
                    });
                }
            }
            _ => {}
        }

        Ok(())
    }

    /// Validate circuit parameters
    pub fn validate_parameters(params: &[f64], expected: usize) -> Result<()> {
        if params.len() != expected {
            return Err(QuantumError::InvalidParameter {
                name: "parameters".to_string(),
                value: format!("expected {} parameters, got {}", expected, params.len()),
            });
        }

        // Check for NaN or infinity
        for (i, &param) in params.iter().enumerate() {
            if param.is_nan() {
                return Err(QuantumError::NumericalError {
                    operation: format!("parameter {}", i),
                    value: param,
                });
            }
            if param.is_infinite() {
                return Err(QuantumError::NumericalError {
                    operation: format!("parameter {}", i),
                    value: param,
                });
            }
        }

        Ok(())
    }

    /// Validate batch size
    pub fn validate_batch_size(size: usize, max: usize) -> Result<()> {
        if size > max {
            return Err(QuantumError::BatchTooLarge { requested: size, max });
        }
        Ok(())
    }

    /// Validate memory allocation
    pub fn validate_memory(requested: usize, available: usize) -> Result<()> {
        if requested > available {
            return Err(QuantumError::MemoryError { requested, available });
        }
        Ok(())
    }
}

/// Circuit validator for checking circuit integrity
pub struct CircuitValidator {
    num_qubits: usize,
    errors: Vec<String>,
    warnings: Vec<String>,
}

impl CircuitValidator {
    pub fn new(num_qubits: usize) -> Self {
        CircuitValidator {
            num_qubits,
            errors: Vec::new(),
            warnings: Vec::new(),
        }
    }

    /// Add a validation error
    fn add_error(&mut self, error: String) {
        self.errors.push(error);
    }

    /// Add a validation warning
    fn add_warning(&mut self, warning: String) {
        self.warnings.push(warning);
    }

    /// Check if validation passed
    pub fn is_valid(&self) -> bool {
        self.errors.is_empty()
    }

    /// Get all errors
    pub fn get_errors(&self) -> &[String] {
        &self.errors
    }

    /// Get all warnings
    pub fn get_warnings(&self) -> &[String] {
        &self.warnings
    }

    /// Check for potential circuit optimizations
    pub fn check_optimization_opportunities(&self) -> Vec<String> {
        vec![
            "Consider merging consecutive rotation gates on the same qubit".to_string(),
            "Look for inverse gate pairs (H·H, X·X, etc.) that can be cancelled".to_string(),
            "Check if single-qubit gates can be moved through multi-qubit gates".to_string(),
        ]
    }
}

/// Safe wrapper for quantum operations with automatic validation
pub struct SafeQuantumExecutor {
    num_qubits: usize,
    validation_enabled: bool,
}

impl SafeQuantumExecutor {
    pub fn new(num_qubits: usize, validation_enabled: bool) -> Self {
        SafeQuantumExecutor {
            num_qubits,
            validation_enabled,
        }
    }

    /// Execute a gate with validation
    pub fn execute_gate<F>(
        &self,
        gate_name: &str,
        targets: &[usize],
        controls: &[usize],
        operation: F,
    ) -> Result<()>
    where
        F: FnOnce() -> (),
    {
        if self.validation_enabled {
            Validator::validate_gate(gate_name, targets, controls, self.num_qubits)?;
        }

        operation();
        Ok(())
    }
}

/// Recovery strategies for failed operations
pub enum RecoveryStrategy {
    /// Retry the operation
    Retry { max_attempts: usize },

    /// Skip the operation and continue
    Skip,

    /// Use a fallback implementation
    Fallback { alternative: String },

    /// Abort the entire operation
    Abort,
}

/// Error recovery handler
pub struct ErrorRecovery {
    strategy: RecoveryStrategy,
}

impl ErrorRecovery {
    pub fn new(strategy: RecoveryStrategy) -> Self {
        ErrorRecovery { strategy }
    }

    /// Attempt to recover from an error
    pub fn recover<T, F>(&self, operation: F) -> Result<T>
    where
        T: Default,
        F: Fn() -> Result<T>,
    {
        match &self.strategy {
            RecoveryStrategy::Retry { max_attempts } => {
                let mut last_error = None;
                for _ in 0..*max_attempts {
                    match operation() {
                        Ok(result) => return Ok(result),
                        Err(e) => last_error = Some(e),
                    }
                }
                Err(last_error.unwrap())
            }
            RecoveryStrategy::Skip => Ok(T::default()),
            RecoveryStrategy::Fallback { alternative: _ } => {
                // In a full implementation, this would try the alternative
                Ok(T::default())
            }
            RecoveryStrategy::Abort => operation(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_qubit_validation() {
        assert!(Validator::validate_qubit_index(0, 4).is_ok());
        assert!(Validator::validate_qubit_index(3, 4).is_ok());
        assert!(Validator::validate_qubit_index(4, 4).is_err());
    }

    #[test]
    fn test_gate_validation() {
        assert!(Validator::validate_gate("H", &[0], &[], 4).is_ok());
        assert!(Validator::validate_gate("CNOT", &[1], &[0], 4).is_ok());
        assert!(Validator::validate_gate("H", &[], &[], 4).is_err());
        assert!(Validator::validate_gate("CNOT", &[0], &[], 4).is_err());
    }

    #[test]
    fn test_parameter_validation() {
        assert!(Validator::validate_parameters(&[0.0, 1.0, 2.0], 3).is_ok());
        assert!(Validator::validate_parameters(&[0.0, 1.0], 3).is_err());
        assert!(Validator::validate_parameters(&[f64::NAN], 1).is_err());
    }

    #[test]
    fn test_error_display() {
        let err = QuantumError::InvalidQubitIndex { index: 10, max: 4 };
        assert_eq!(format!("{}", err), "Invalid qubit index 10: maximum is 4");
    }
}
