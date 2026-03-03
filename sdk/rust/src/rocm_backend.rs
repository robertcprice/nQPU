//! ROCm Backend for AMD GPU Acceleration
//!
//! This module provides AMD GPU acceleration using ROCm (Radeon Open Compute).
//! It is only available when the "rocm" feature is enabled and on systems
//! with AMD GPUs and ROCm toolkit installed.

/// Error type for ROCm operations
#[derive(Clone, Debug)]
pub enum RocmError {
    InitializationFailed(String),
    DeviceNotFound,
    MemoryAllocationFailed { requested: usize, available: usize },
    KernelLaunchFailed(String),
    InvalidArgument(String),
    InternalError(String),
}

impl std::fmt::Display for RocmError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RocmError::InitializationFailed(msg) => {
                write!(f, "ROCm initialization failed: {}", msg)
            }
            RocmError::DeviceNotFound => write!(f, "No ROCm device found"),
            RocmError::MemoryAllocationFailed {
                requested,
                available,
            } => {
                write!(
                    f,
                    "Memory allocation failed: requested {} bytes, {} available",
                    requested, available
                )
            }
            RocmError::KernelLaunchFailed(msg) => write!(f, "Kernel launch failed: {}", msg),
            RocmError::InvalidArgument(msg) => write!(f, "Invalid argument: {}", msg),
            RocmError::InternalError(msg) => write!(f, "Internal error: {}", msg),
        }
    }
}

impl std::error::Error for RocmError {}

pub type RocmResult<T> = std::result::Result<T, RocmError>;

/// Complex number for ROCm (matches HIP complex structure)
#[derive(Clone, Copy, Debug, PartialEq)]
#[repr(C)]
pub struct RocmComplex {
    pub x: f64, // Real part
    pub y: f64, // Imaginary part
}

impl RocmComplex {
    pub fn new(re: f64, im: f64) -> Self {
        RocmComplex { x: re, y: im }
    }

    pub fn zero() -> Self {
        RocmComplex { x: 0.0, y: 0.0 }
    }

    pub fn one() -> Self {
        RocmComplex { x: 1.0, y: 0.0 }
    }
}

/// ROCm quantum simulator backend
pub struct RocmQuantumSimulator {
    device: RocmDevice,
    num_qubits: usize,
    state_buffer: Vec<RocmComplex>,
}

/// ROCm device information
#[derive(Clone, Debug)]
pub struct RocmDevice {
    pub name: String,
    pub total_memory: usize,
    pub compute_capability: String,
}

impl RocmDevice {
    /// Attempt to initialize an AMD ROCm GPU device.
    ///
    /// **Note**: This requires linking against the AMD HIP runtime (libamdhip64).
    /// Without the HIP SDK installed, this will always return an error.
    /// Build with `--features rocm` and the HIP SDK to enable AMD GPU support.
    pub fn new(_device_index: usize) -> RocmResult<Self> {
        Err(RocmError::InitializationFailed(
            "ROCm/HIP runtime not available. Install the AMD ROCm SDK and rebuild \
             with `--features rocm` to enable AMD GPU support.".to_string()
        ))
    }
}

impl RocmQuantumSimulator {
    /// Create a new ROCm quantum simulator
    pub fn new(num_qubits: usize) -> RocmResult<Self> {
        let device = RocmDevice::new(0)?;
        let dim = 1_usize << num_qubits;
        let mut state_buffer = vec![RocmComplex::zero(); dim];
        state_buffer[0] = RocmComplex::one(); // Initialize to |0...0⟩

        Ok(RocmQuantumSimulator {
            device,
            num_qubits,
            state_buffer,
        })
    }

    /// Get the number of qubits
    pub fn num_qubits(&self) -> usize {
        self.num_qubits
    }

    /// Get the device name
    pub fn device_name(&self) -> &str {
        &self.device.name
    }

    /// Reset the state to |0...0⟩
    pub fn reset(&mut self) {
        self.state_buffer = vec![RocmComplex::zero(); 1 << self.num_qubits];
        self.state_buffer[0] = RocmComplex::one();
    }

    /// Get the current state vector
    pub fn get_state(&self) -> Vec<RocmComplex> {
        self.state_buffer.clone()
    }

    /// Set the state vector
    pub fn set_state(&mut self, state: Vec<RocmComplex>) -> RocmResult<()> {
        if state.len() != (1 << self.num_qubits) {
            return Err(RocmError::InvalidArgument(
                "State size mismatch".to_string(),
            ));
        }
        self.state_buffer = state;
        Ok(())
    }

    /// Apply Hadamard gate
    pub fn h(&mut self, qubit: usize) -> RocmResult<()> {
        if qubit >= self.num_qubits {
            return Err(RocmError::InvalidArgument(format!(
                "Qubit index {} out of range",
                qubit
            )));
        }

        let stride = 1 << qubit;
        for i in 0..self.state_buffer.len() {
            let j = i ^ stride;
            if i < j {
                let a = self.state_buffer[i];
                let b = self.state_buffer[j];
                let inv_sqrt2 = 0.7071067811865475;
                self.state_buffer[i] =
                    RocmComplex::new((a.x + b.x) * inv_sqrt2, (a.y + b.y) * inv_sqrt2);
                self.state_buffer[j] =
                    RocmComplex::new((a.x - b.x) * inv_sqrt2, (a.y - b.y) * inv_sqrt2);
            }
        }
        Ok(())
    }

    /// Apply Pauli-X gate
    pub fn x(&mut self, qubit: usize) -> RocmResult<()> {
        if qubit >= self.num_qubits {
            return Err(RocmError::InvalidArgument(format!(
                "Qubit index {} out of range",
                qubit
            )));
        }

        let stride = 1 << qubit;
        for i in 0..self.state_buffer.len() {
            let j = i ^ stride;
            if i < j {
                self.state_buffer.swap(i, j);
            }
        }
        Ok(())
    }

    /// Apply Pauli-Y gate
    pub fn y(&mut self, qubit: usize) -> RocmResult<()> {
        if qubit >= self.num_qubits {
            return Err(RocmError::InvalidArgument(format!(
                "Qubit index {} out of range",
                qubit
            )));
        }

        let stride = 1 << qubit;
        for i in 0..self.state_buffer.len() {
            let j = i ^ stride;
            if i < j {
                let a = self.state_buffer[i];
                let b = self.state_buffer[j];
                // Y: |0⟩ → i|1⟩, |1⟩ → -i|0⟩
                self.state_buffer[i] = RocmComplex::new(b.y, -b.x);
                self.state_buffer[j] = RocmComplex::new(-a.y, a.x);
            }
        }
        Ok(())
    }

    /// Apply Pauli-Z gate
    pub fn z(&mut self, qubit: usize) -> RocmResult<()> {
        if qubit >= self.num_qubits {
            return Err(RocmError::InvalidArgument(format!(
                "Qubit index {} out of range",
                qubit
            )));
        }

        let mask = 1 << qubit;
        for i in 0..self.state_buffer.len() {
            if (i & mask) != 0 {
                self.state_buffer[i].x = -self.state_buffer[i].x;
                self.state_buffer[i].y = -self.state_buffer[i].y;
            }
        }
        Ok(())
    }

    /// Apply S gate (phase gate, π/2 rotation)
    pub fn s(&mut self, qubit: usize) -> RocmResult<()> {
        if qubit >= self.num_qubits {
            return Err(RocmError::InvalidArgument(format!(
                "Qubit index {} out of range",
                qubit
            )));
        }

        let mask = 1 << qubit;
        for i in 0..self.state_buffer.len() {
            if (i & mask) != 0 {
                // Multiply by i: (x + iy) * i = -y + ix
                let orig_x = self.state_buffer[i].x;
                self.state_buffer[i].x = -self.state_buffer[i].y;
                self.state_buffer[i].y = orig_x;
            }
        }
        Ok(())
    }

    /// Apply T gate (π/8 gate, π/4 rotation)
    pub fn t(&mut self, qubit: usize) -> RocmResult<()> {
        if qubit >= self.num_qubits {
            return Err(RocmError::InvalidArgument(format!(
                "Qubit index {} out of range",
                qubit
            )));
        }

        let mask = 1 << qubit;
        let cos_pi4 = (std::f64::consts::PI / 4.0).cos();
        let sin_pi4 = (std::f64::consts::PI / 4.0).sin();
        for i in 0..self.state_buffer.len() {
            if (i & mask) != 0 {
                let orig_x = self.state_buffer[i].x;
                self.state_buffer[i].x = orig_x * cos_pi4 - self.state_buffer[i].y * sin_pi4;
                self.state_buffer[i].y = orig_x * sin_pi4 + self.state_buffer[i].y * cos_pi4;
            }
        }
        Ok(())
    }

    /// Apply RX rotation gate
    pub fn rx(&mut self, qubit: usize, theta: f64) -> RocmResult<()> {
        if qubit >= self.num_qubits {
            return Err(RocmError::InvalidArgument(format!(
                "Qubit index {} out of range",
                qubit
            )));
        }

        let stride = 1 << qubit;
        let cos_half = (theta / 2.0).cos();
        let sin_half = (theta / 2.0).sin();
        for i in 0..self.state_buffer.len() {
            let j = i ^ stride;
            if i < j {
                let a = self.state_buffer[i];
                let b = self.state_buffer[j];
                // Rx(θ) = [[cos(θ/2), -i*sin(θ/2)], [-i*sin(θ/2), cos(θ/2)]]
                self.state_buffer[i] = RocmComplex::new(
                    a.x * cos_half + b.y * sin_half,
                    a.y * cos_half - b.x * sin_half,
                );
                self.state_buffer[j] = RocmComplex::new(
                    b.x * cos_half + a.y * sin_half,
                    b.y * cos_half - a.x * sin_half,
                );
            }
        }
        Ok(())
    }

    /// Apply RY rotation gate
    pub fn ry(&mut self, qubit: usize, theta: f64) -> RocmResult<()> {
        if qubit >= self.num_qubits {
            return Err(RocmError::InvalidArgument(format!(
                "Qubit index {} out of range",
                qubit
            )));
        }

        let stride = 1 << qubit;
        let cos_half = (theta / 2.0).cos();
        let sin_half = (theta / 2.0).sin();
        for i in 0..self.state_buffer.len() {
            let j = i ^ stride;
            if i < j {
                let a = self.state_buffer[i];
                let b = self.state_buffer[j];
                self.state_buffer[i] = RocmComplex::new(
                    a.x * cos_half - b.x * sin_half,
                    a.y * cos_half - b.y * sin_half,
                );
                self.state_buffer[j] = RocmComplex::new(
                    a.x * sin_half + b.x * cos_half,
                    a.y * sin_half + b.y * cos_half,
                );
            }
        }
        Ok(())
    }

    /// Apply RZ rotation gate
    pub fn rz(&mut self, qubit: usize, theta: f64) -> RocmResult<()> {
        if qubit >= self.num_qubits {
            return Err(RocmError::InvalidArgument(format!(
                "Qubit index {} out of range",
                qubit
            )));
        }

        let mask = 1 << qubit;
        let cos_half = (theta / 2.0).cos();
        let sin_half = (theta / 2.0).sin();
        for i in 0..self.state_buffer.len() {
            let orig_x = self.state_buffer[i].x;
            if (i & mask) != 0 {
                // |1⟩: multiply by exp(iθ/2)
                self.state_buffer[i].x = orig_x * cos_half - self.state_buffer[i].y * sin_half;
                self.state_buffer[i].y = orig_x * sin_half + self.state_buffer[i].y * cos_half;
            } else {
                // |0⟩: multiply by exp(-iθ/2)
                self.state_buffer[i].x = orig_x * cos_half + self.state_buffer[i].y * sin_half;
                self.state_buffer[i].y = -orig_x * sin_half + self.state_buffer[i].y * cos_half;
            }
        }
        Ok(())
    }

    /// Apply CNOT gate
    pub fn cx(&mut self, control: usize, target: usize) -> RocmResult<()> {
        if control >= self.num_qubits || target >= self.num_qubits {
            return Err(RocmError::InvalidArgument(
                "Qubit index out of range".to_string(),
            ));
        }

        let control_mask = 1 << control;
        let target_mask = 1 << target;

        for i in 0..self.state_buffer.len() {
            if (i & control_mask) != 0 {
                let j = i ^ target_mask;
                if i < j {
                    self.state_buffer.swap(i, j);
                }
            }
        }
        Ok(())
    }

    /// Apply CZ gate
    pub fn cz(&mut self, control: usize, target: usize) -> RocmResult<()> {
        if control >= self.num_qubits || target >= self.num_qubits {
            return Err(RocmError::InvalidArgument(
                "Qubit index out of range".to_string(),
            ));
        }

        let control_mask = 1 << control;
        let target_mask = 1 << target;

        for i in 0..self.state_buffer.len() {
            if (i & control_mask) != 0 && (i & target_mask) != 0 {
                self.state_buffer[i].x = -self.state_buffer[i].x;
                self.state_buffer[i].y = -self.state_buffer[i].y;
            }
        }
        Ok(())
    }

    /// Apply SWAP gate
    pub fn swap(&mut self, qubit1: usize, qubit2: usize) -> RocmResult<()> {
        if qubit1 >= self.num_qubits || qubit2 >= self.num_qubits {
            return Err(RocmError::InvalidArgument(
                "Qubit index out of range".to_string(),
            ));
        }
        if qubit1 == qubit2 {
            return Ok(());
        }

        let mask1 = 1 << qubit1;
        let mask2 = 1 << qubit2;

        for i in 0..self.state_buffer.len() {
            let bit1 = (i & mask1) != 0;
            let bit2 = (i & mask2) != 0;
            if bit1 != bit2 {
                let mut j = i;
                if bit1 {
                    j &= !mask1;
                } else {
                    j |= mask1;
                }
                if bit2 {
                    j &= !mask2;
                } else {
                    j |= mask2;
                }
                if i < j {
                    self.state_buffer.swap(i, j);
                }
            }
        }
        Ok(())
    }

    /// Measure a single qubit
    pub fn measure(&mut self, qubit: usize) -> RocmResult<bool> {
        if qubit >= self.num_qubits {
            return Err(RocmError::InvalidArgument(format!(
                "Qubit index {} out of range",
                qubit
            )));
        }

        // Calculate probabilities
        let mut p0 = 0.0;
        let stride = 1 << qubit;
        for i in 0..self.state_buffer.len() {
            if (i & stride) == 0 {
                let amp = &self.state_buffer[i];
                p0 += amp.x * amp.x + amp.y * amp.y;
            }
        }

        // Random measurement
        let outcome = rand::random::<f64>() > p0;

        // Collapse state to measurement outcome
        let mask = 1 << qubit;
        for i in 0..self.state_buffer.len() {
            if ((i & mask) != 0) != outcome {
                self.state_buffer[i] = RocmComplex::zero();
            }
        }

        // Normalize
        let mut norm = 0.0;
        for amp in &self.state_buffer {
            norm += amp.x * amp.x + amp.y * amp.y;
        }
        if norm > 0.0 {
            let scale = 1.0 / norm.sqrt();
            for amp in &mut self.state_buffer {
                amp.x *= scale;
                amp.y *= scale;
            }
        }

        Ok(outcome)
    }
}

/// Check if ROCm is available on this system.
///
/// On Linux, probes for the ROCm installation by checking:
/// 1. Whether `/opt/rocm` exists (standard ROCm install path)
/// 2. Whether the HIP runtime library (`libamdhip64.so`) is present
/// 3. Whether any AMD GPU devices are exposed via `/sys/class/kfd/kfd/topology/nodes/`
///
/// On non-Linux platforms, ROCm is never available and this returns false
/// immediately.
pub fn is_rocm_available() -> bool {
    #[cfg(target_os = "linux")]
    {
        use std::path::Path;

        // Check 1: ROCm SDK installation directory
        let rocm_dir_exists = Path::new("/opt/rocm").is_dir();
        if !rocm_dir_exists {
            // Also check ROCM_PATH environment variable
            if let Ok(rocm_path) = std::env::var("ROCM_PATH") {
                if !Path::new(&rocm_path).is_dir() {
                    return false;
                }
            } else {
                return false;
            }
        }

        // Check 2: HIP runtime library
        let hip_lib_paths = [
            "/opt/rocm/lib/libamdhip64.so",
            "/opt/rocm/hip/lib/libamdhip64.so",
        ];
        let hip_lib_found = hip_lib_paths.iter().any(|p| Path::new(p).exists());
        if !hip_lib_found {
            // Try ROCM_PATH-based path
            if let Ok(rocm_path) = std::env::var("ROCM_PATH") {
                let custom_path = format!("{}/lib/libamdhip64.so", rocm_path);
                if !Path::new(&custom_path).exists() {
                    return false;
                }
            } else {
                return false;
            }
        }

        // Check 3: AMD GPU devices via KFD topology
        // The KFD (Kernel Fusion Driver) exposes GPU nodes under
        // /sys/class/kfd/kfd/topology/nodes/*/properties
        // A node with "gfx_target_version" > 0 indicates a GPU.
        let kfd_nodes = Path::new("/sys/class/kfd/kfd/topology/nodes");
        if kfd_nodes.is_dir() {
            if let Ok(entries) = std::fs::read_dir(kfd_nodes) {
                for entry in entries.flatten() {
                    let props_path = entry.path().join("properties");
                    if let Ok(props) = std::fs::read_to_string(&props_path) {
                        // Look for a non-zero gfx_target_version indicating a GPU
                        for line in props.lines() {
                            if line.starts_with("gfx_target_version") {
                                if let Some(val_str) = line.split_whitespace().nth(1) {
                                    if let Ok(ver) = val_str.parse::<u64>() {
                                        if ver > 0 {
                                            return true;
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        // Fallback: ROCm dir and lib exist but no GPU nodes found
        // This can happen in container environments; still report available
        // since the runtime is present and hipGetDeviceCount would be the
        // definitive check at actual init time.
        true
    }

    #[cfg(not(target_os = "linux"))]
    {
        false
    }
}

/// Get the number of available ROCm devices.
///
/// On Linux with ROCm available, counts GPU nodes exposed by the KFD
/// driver. Returns 0 on non-Linux or if ROCm is not available.
pub fn rocm_device_count() -> usize {
    if !is_rocm_available() {
        return 0;
    }

    #[cfg(target_os = "linux")]
    {
        use std::path::Path;

        let kfd_nodes = Path::new("/sys/class/kfd/kfd/topology/nodes");
        if !kfd_nodes.is_dir() {
            return 1; // ROCm available but can't enumerate; assume 1
        }

        let mut gpu_count = 0usize;
        if let Ok(entries) = std::fs::read_dir(kfd_nodes) {
            for entry in entries.flatten() {
                let props_path = entry.path().join("properties");
                if let Ok(props) = std::fs::read_to_string(&props_path) {
                    for line in props.lines() {
                        if line.starts_with("gfx_target_version") {
                            if let Some(val_str) = line.split_whitespace().nth(1) {
                                if let Ok(ver) = val_str.parse::<u64>() {
                                    if ver > 0 {
                                        gpu_count += 1;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        if gpu_count > 0 { gpu_count } else { 1 }
    }

    #[cfg(not(target_os = "linux"))]
    {
        0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rocm_complex() {
        let c = RocmComplex::new(1.0, 2.0);
        assert_eq!(c.x, 1.0);
        assert_eq!(c.y, 2.0);
    }

    #[test]
    fn test_rocm_simulator() {
        // RocmDevice::new currently always returns Err because the HIP
        // runtime is not linked. Verify the error is descriptive, and
        // that if we ever get a working device the simulator API is sound.
        let sim = RocmQuantumSimulator::new(4);
        if sim.is_ok() {
            let mut sim = sim.unwrap();
            assert_eq!(sim.num_qubits(), 4);

            // Test Hadamard
            sim.h(0).unwrap();

            // Test measurement
            let result = sim.measure(0);
            assert!(result.is_ok());
        } else {
            // Expected path: ROCm/HIP not available
            let err = sim.unwrap_err();
            assert!(
                err.to_string().contains("ROCm") || err.to_string().contains("HIP"),
                "Error should mention ROCm or HIP, got: {}",
                err
            );
        }
    }

    #[test]
    fn test_is_rocm_available() {
        // This test will pass on systems without ROCm
        let available = is_rocm_available();
        println!("ROCm available: {}", available);
    }
}
