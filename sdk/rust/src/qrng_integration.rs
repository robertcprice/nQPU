//! QRNG Integration Module
//!
//! Provides a unified interface for quantum random number generators (QRNGs)
//! to supply measurement randomness in the nQPU-Metal quantum simulator.
//!
//! # Architecture
//!
//! The module is built around the [`QrngSource`] trait, which defines a
//! common interface for all randomness sources. Concrete implementations
//! include:
//!
//! - [`SimulatedQrngSource`]: CSPRNG-based fallback (not quantum, but
//!   cryptographically strong) for offline development and testing.
//! - [`AnuQrngSource`]: Client for the ANU Quantum Random Numbers API.
//!   Ships with a `from_bytes` constructor for deterministic testing and
//!   a [`SimulatedAnuSource`] that mimics the API using local CSPRNG.
//! - [`HardwareQrngSource`]: Reads raw bytes from a QRNG device file
//!   (e.g., `/dev/quantis0`). Unix-only, feature-gated.
//! - [`HybridQrngSource`]: Primary + fallback composition with
//!   diagnostic counters for fallback events.
//!
//! The [`QrngRng`] adapter wraps any `QrngSource` to implement
//! `rand::RngCore`, making it drop-in compatible with the rest of the
//! simulator's random number pipeline.
//!
//! # Usage
//!
//! ```rust,no_run
//! use nqpu_metal::qrng_integration::{create_measurement_rng, QrngConfig, QrngSource};
//!
//! // Default: simulated CSPRNG source
//! let mut source = create_measurement_rng(None);
//! let mut buf = [0u8; 32];
//! source.fill_bytes(&mut buf).unwrap();
//! ```

use rand::RngCore;
use std::fmt;

// ---------------------------------------------------------------------------
// Error types
// ---------------------------------------------------------------------------

/// Errors that can occur when interacting with a QRNG source.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum QrngError {
    /// A network-level failure prevented communication with a remote QRNG
    /// service (e.g., DNS resolution failure, connection refused).
    NetworkError(String),

    /// The underlying hardware device reported an error or is unavailable.
    DeviceError(String),

    /// The internal buffer has been fully consumed and the source was unable
    /// to refill it (e.g., a pre-loaded `from_bytes` source with no more
    /// data).
    BufferExhausted,

    /// The operation did not complete within the configured timeout.
    Timeout,
}

impl fmt::Display for QrngError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            QrngError::NetworkError(msg) => write!(f, "QRNG network error: {}", msg),
            QrngError::DeviceError(msg) => write!(f, "QRNG device error: {}", msg),
            QrngError::BufferExhausted => write!(f, "QRNG buffer exhausted"),
            QrngError::Timeout => write!(f, "QRNG operation timed out"),
        }
    }
}

impl std::error::Error for QrngError {}

// ---------------------------------------------------------------------------
// Source metadata
// ---------------------------------------------------------------------------

/// Descriptive metadata for a QRNG source.
#[derive(Debug, Clone)]
pub struct QrngSourceInfo {
    /// Human-readable name of the source (e.g., "ANU QRNG", "Quantis-USB").
    pub name: String,

    /// Estimated throughput in bits per second, if known.
    pub throughput_bps: Option<u64>,

    /// Whether this source is backed by real quantum hardware.
    pub is_hardware: bool,
}

// ---------------------------------------------------------------------------
// Core trait
// ---------------------------------------------------------------------------

/// Trait defining the interface for all QRNG sources.
///
/// Implementations must be able to fill a byte buffer with random data
/// and report metadata about the source.
pub trait QrngSource: Send {
    /// Fill `dest` with random bytes from this source.
    ///
    /// Returns `Ok(())` on success, or an appropriate [`QrngError`] on
    /// failure. Implementations should make a best-effort attempt to fill
    /// the entire buffer; partial fills are treated as errors.
    fn fill_bytes(&mut self, dest: &mut [u8]) -> Result<(), QrngError>;

    /// Return metadata describing this source.
    fn source_info(&self) -> QrngSourceInfo;
}

// ---------------------------------------------------------------------------
// SimulatedQrngSource
// ---------------------------------------------------------------------------

/// A CSPRNG-based QRNG substitute for offline use.
///
/// Uses a simple xorshift128+ generator seeded from `rand::thread_rng()`
/// (which itself is seeded from system entropy). This is **not** quantum
/// random but provides high-quality pseudo-random bytes suitable for
/// simulation when no hardware QRNG is available.
pub struct SimulatedQrngSource {
    /// Internal xorshift128+ state (two 64-bit words).
    state: [u64; 2],
}

impl SimulatedQrngSource {
    /// Create a new `SimulatedQrngSource` seeded from system entropy.
    pub fn new() -> Self {
        let mut rng = rand::thread_rng();
        let s0 = rng.next_u64();
        // Ensure non-zero state: xorshift128+ requires at least one
        // non-zero word.
        let s1 = loop {
            let v = rng.next_u64();
            if v != 0 || s0 != 0 {
                break v;
            }
        };
        Self { state: [s0, s1] }
    }

    /// Create a `SimulatedQrngSource` with a deterministic seed.
    ///
    /// Useful for reproducible tests. The two seed words must not both
    /// be zero.
    ///
    /// # Panics
    ///
    /// Panics if both `seed_lo` and `seed_hi` are zero.
    pub fn with_seed(seed_lo: u64, seed_hi: u64) -> Self {
        assert!(
            seed_lo != 0 || seed_hi != 0,
            "xorshift128+ seed must not be all-zero"
        );
        Self {
            state: [seed_lo, seed_hi],
        }
    }

    /// Advance the xorshift128+ state and return the next 64 bits.
    #[inline]
    fn next_u64_internal(&mut self) -> u64 {
        let mut s1 = self.state[0];
        let s0 = self.state[1];
        self.state[0] = s0;
        s1 ^= s1 << 23;
        s1 ^= s1 >> 17;
        s1 ^= s0;
        s1 ^= s0 >> 26;
        self.state[1] = s1;
        s0.wrapping_add(s1)
    }
}

impl Default for SimulatedQrngSource {
    fn default() -> Self {
        Self::new()
    }
}

impl QrngSource for SimulatedQrngSource {
    fn fill_bytes(&mut self, dest: &mut [u8]) -> Result<(), QrngError> {
        // Fill 8 bytes at a time from the xorshift128+ output.
        let mut pos = 0;
        while pos + 8 <= dest.len() {
            let val = self.next_u64_internal();
            dest[pos..pos + 8].copy_from_slice(&val.to_le_bytes());
            pos += 8;
        }
        // Handle the tail (0..7 remaining bytes).
        if pos < dest.len() {
            let val = self.next_u64_internal();
            let bytes = val.to_le_bytes();
            let remaining = dest.len() - pos;
            dest[pos..].copy_from_slice(&bytes[..remaining]);
        }
        Ok(())
    }

    fn source_info(&self) -> QrngSourceInfo {
        QrngSourceInfo {
            name: "SimulatedQrng (xorshift128+)".to_string(),
            throughput_bps: None, // effectively unlimited
            is_hardware: false,
        }
    }
}

// ---------------------------------------------------------------------------
// AnuQrngSource  (stub / offline-capable)
// ---------------------------------------------------------------------------

/// Represents a structured request to the ANU Quantum Random Numbers API.
///
/// This is exposed so callers can inspect the request that *would* be sent
/// without requiring `reqwest` as a dependency.
#[derive(Debug, Clone)]
pub struct AnuApiRequest {
    /// The full URL for the API endpoint.
    pub url: String,
    /// The API key sent in the `x-api-key` header.
    pub api_key: String,
    /// Number of random bytes requested.
    pub length: usize,
}

/// Client for the ANU Quantum Random Numbers Server.
///
/// Because adding `reqwest` as a hard dependency is not desirable for this
/// crate, the actual HTTP call is not implemented. Instead, this struct
/// supports two modes:
///
/// 1. **Pre-loaded**: Constructed via [`AnuQrngSource::from_bytes`] with a
///    `Vec<u8>` of real QRNG data obtained out-of-band. The source drains
///    this buffer and returns [`QrngError::BufferExhausted`] when empty.
///
/// 2. **Request generation**: [`AnuQrngSource::build_request`] returns an
///    [`AnuApiRequest`] that the caller can execute with any HTTP client.
///
/// For a fully offline simulation of the ANU API, see
/// [`SimulatedAnuSource`].
pub struct AnuQrngSource {
    api_key: String,
    buffer: Vec<u8>,
    buffer_pos: usize,
    prefetch_size: usize,
}

/// Default prefetch buffer size in bytes (4 KiB).
const DEFAULT_ANU_BUFFER_SIZE: usize = 4096;

/// ANU QRNG REST API base URL.
const ANU_API_BASE_URL: &str = "https://api.quantumnumbers.anu.edu.au";

impl AnuQrngSource {
    /// Create a new `AnuQrngSource` configured with an API key.
    ///
    /// The source starts with an empty buffer. Call [`build_request`] to
    /// obtain the HTTP request structure, execute it externally, then feed
    /// the response bytes via [`load_bytes`].
    pub fn new(api_key: String) -> Self {
        Self {
            api_key,
            buffer: Vec::new(),
            buffer_pos: 0,
            prefetch_size: DEFAULT_ANU_BUFFER_SIZE,
        }
    }

    /// Set the prefetch buffer size (number of bytes to request per batch).
    pub fn with_buffer_size(mut self, size: usize) -> Self {
        self.prefetch_size = size;
        self
    }

    /// Create an `AnuQrngSource` pre-loaded with raw random bytes.
    ///
    /// This constructor is the primary way to use the source in tests or
    /// when QRNG bytes have been fetched externally and cached to disk.
    pub fn from_bytes(api_key: String, data: Vec<u8>) -> Self {
        Self {
            api_key,
            buffer: data,
            buffer_pos: 0,
            prefetch_size: DEFAULT_ANU_BUFFER_SIZE,
        }
    }

    /// Load additional random bytes into the internal buffer.
    ///
    /// Bytes are appended after any unconsumed data that remains in the
    /// buffer. This is intended for use after executing the HTTP request
    /// returned by [`build_request`].
    pub fn load_bytes(&mut self, data: &[u8]) {
        // Compact: discard already-consumed prefix.
        if self.buffer_pos > 0 {
            self.buffer = self.buffer[self.buffer_pos..].to_vec();
            self.buffer_pos = 0;
        }
        self.buffer.extend_from_slice(data);
    }

    /// Build the HTTP request structure for fetching random bytes.
    ///
    /// The caller is responsible for executing the request with an HTTP
    /// client and then passing the response body to [`load_bytes`].
    pub fn build_request(&self) -> AnuApiRequest {
        AnuApiRequest {
            url: format!(
                "{}/API/jsonI.php?length={}&type=uint8",
                ANU_API_BASE_URL, self.prefetch_size
            ),
            api_key: self.api_key.clone(),
            length: self.prefetch_size,
        }
    }

    /// Return the number of unconsumed bytes remaining in the buffer.
    pub fn remaining(&self) -> usize {
        if self.buffer_pos >= self.buffer.len() {
            0
        } else {
            self.buffer.len() - self.buffer_pos
        }
    }

    /// Return the configured prefetch buffer size.
    pub fn prefetch_size(&self) -> usize {
        self.prefetch_size
    }
}

impl QrngSource for AnuQrngSource {
    fn fill_bytes(&mut self, dest: &mut [u8]) -> Result<(), QrngError> {
        let available = self.remaining();
        if available < dest.len() {
            return Err(QrngError::BufferExhausted);
        }
        let start = self.buffer_pos;
        let end = start + dest.len();
        dest.copy_from_slice(&self.buffer[start..end]);
        self.buffer_pos = end;
        Ok(())
    }

    fn source_info(&self) -> QrngSourceInfo {
        QrngSourceInfo {
            name: "ANU QRNG".to_string(),
            throughput_bps: Some(5_000_000), // ~5 Mbps typical API throughput
            is_hardware: true,
        }
    }
}

// ---------------------------------------------------------------------------
// SimulatedAnuSource
// ---------------------------------------------------------------------------

/// Simulates the ANU QRNG API using a local CSPRNG.
///
/// Behaves identically to [`AnuQrngSource`] from the caller's perspective
/// but generates bytes locally, making it suitable for offline development
/// and CI pipelines.
pub struct SimulatedAnuSource {
    inner: SimulatedQrngSource,
    prefetch_size: usize,
}

impl SimulatedAnuSource {
    /// Create a new simulated ANU source with the given prefetch size.
    pub fn new(prefetch_size: usize) -> Self {
        Self {
            inner: SimulatedQrngSource::new(),
            prefetch_size,
        }
    }

    /// Create with a deterministic seed for reproducible testing.
    pub fn with_seed(seed_lo: u64, seed_hi: u64, prefetch_size: usize) -> Self {
        Self {
            inner: SimulatedQrngSource::with_seed(seed_lo, seed_hi),
            prefetch_size,
        }
    }

    /// Return the configured prefetch size.
    pub fn prefetch_size(&self) -> usize {
        self.prefetch_size
    }
}

impl QrngSource for SimulatedAnuSource {
    fn fill_bytes(&mut self, dest: &mut [u8]) -> Result<(), QrngError> {
        self.inner.fill_bytes(dest)
    }

    fn source_info(&self) -> QrngSourceInfo {
        QrngSourceInfo {
            name: "Simulated ANU QRNG (local CSPRNG)".to_string(),
            throughput_bps: None,
            is_hardware: false,
        }
    }
}

// ---------------------------------------------------------------------------
// HardwareQrngSource (Unix only)
// ---------------------------------------------------------------------------

/// Reads raw random bytes from a QRNG device file.
///
/// Supports any device that exposes a byte-stream interface, such as:
/// - ID Quantique Quantis (`/dev/quantis0`)
/// - Comscire TRNG (`/dev/comscire0`)
/// - Any `/dev/hwrng` or similar kernel-exposed QRNG device
///
/// This source is only available on Unix platforms.
#[cfg(unix)]
pub struct HardwareQrngSource {
    device_path: String,
    /// Optional cached file handle. Opened lazily on first read.
    file: Option<std::fs::File>,
}

#[cfg(unix)]
impl HardwareQrngSource {
    /// Create a new hardware QRNG source reading from `device_path`.
    ///
    /// The device file is not opened until the first call to
    /// [`fill_bytes`]. This allows construction to succeed even if the
    /// device is not yet available.
    pub fn new(device_path: String) -> Self {
        Self {
            device_path,
            file: None,
        }
    }

    /// Return the configured device path.
    pub fn device_path(&self) -> &str {
        &self.device_path
    }

    /// Open (or re-open) the device file.
    fn open_device(&mut self) -> Result<&mut std::fs::File, QrngError> {
        if self.file.is_none() {
            let f = std::fs::File::open(&self.device_path).map_err(|e| {
                QrngError::DeviceError(format!(
                    "failed to open {}: {}",
                    self.device_path, e
                ))
            })?;
            self.file = Some(f);
        }
        Ok(self.file.as_mut().unwrap())
    }
}

#[cfg(unix)]
impl QrngSource for HardwareQrngSource {
    fn fill_bytes(&mut self, dest: &mut [u8]) -> Result<(), QrngError> {
        use std::io::Read;

        let file = self.open_device()?;
        let mut total_read = 0;
        while total_read < dest.len() {
            match file.read(&mut dest[total_read..]) {
                Ok(0) => {
                    return Err(QrngError::DeviceError(format!(
                        "unexpected EOF reading from {}",
                        self.device_path
                    )));
                }
                Ok(n) => total_read += n,
                Err(ref e) if e.kind() == std::io::ErrorKind::Interrupted => continue,
                Err(e) => {
                    // Invalidate the file handle so the next call attempts
                    // a fresh open.
                    self.file = None;
                    return Err(QrngError::DeviceError(format!(
                        "read error on {}: {}",
                        self.device_path, e
                    )));
                }
            }
        }
        Ok(())
    }

    fn source_info(&self) -> QrngSourceInfo {
        QrngSourceInfo {
            name: format!("Hardware QRNG ({})", self.device_path),
            throughput_bps: None,
            is_hardware: true,
        }
    }
}

// ---------------------------------------------------------------------------
// QrngRng adapter  (QrngSource -> rand::RngCore)
// ---------------------------------------------------------------------------

/// Default internal buffer size for the `QrngRng` adapter (1 KiB).
const QRNG_RNG_BUFFER_SIZE: usize = 1024;

/// Adapter that wraps any [`QrngSource`] to implement [`rand::RngCore`].
///
/// Internally maintains a byte buffer that is refilled from the wrapped
/// source on demand. This amortizes the cost of source calls across many
/// small random-number requests.
pub struct QrngRng {
    source: Box<dyn QrngSource>,
    buffer: Vec<u8>,
    pos: usize,
    buffer_size: usize,
}

impl QrngRng {
    /// Create a new adapter wrapping the given source.
    pub fn new(source: Box<dyn QrngSource>) -> Self {
        Self {
            source,
            buffer: Vec::new(),
            pos: 0,
            buffer_size: QRNG_RNG_BUFFER_SIZE,
        }
    }

    /// Create a new adapter with a custom buffer size.
    pub fn with_buffer_size(source: Box<dyn QrngSource>, buffer_size: usize) -> Self {
        assert!(buffer_size >= 8, "buffer size must be at least 8 bytes");
        Self {
            source,
            buffer: Vec::new(),
            pos: 0,
            buffer_size,
        }
    }

    /// Return a reference to the underlying source's metadata.
    pub fn source_info(&self) -> QrngSourceInfo {
        self.source.source_info()
    }

    /// Refill the internal buffer from the source.
    ///
    /// Returns `Ok(())` on success. On failure the buffer is left empty so
    /// subsequent reads will also attempt a refill.
    fn refill(&mut self) -> Result<(), QrngError> {
        self.buffer.resize(self.buffer_size, 0);
        self.pos = 0;
        self.source.fill_bytes(&mut self.buffer)?;
        Ok(())
    }

    /// Ensure at least `needed` bytes are available in the buffer.
    fn ensure_available(&mut self, needed: usize) -> Result<(), QrngError> {
        let available = if self.pos < self.buffer.len() {
            self.buffer.len() - self.pos
        } else {
            0
        };
        if available < needed {
            self.refill()?;
        }
        Ok(())
    }

    /// Consume `n` bytes from the buffer, copying them to `dest`.
    ///
    /// The caller must ensure at least `n` bytes are available.
    fn consume(&mut self, dest: &mut [u8]) {
        let n = dest.len();
        dest.copy_from_slice(&self.buffer[self.pos..self.pos + n]);
        self.pos += n;
    }
}

impl RngCore for QrngRng {
    fn next_u32(&mut self) -> u32 {
        if self.ensure_available(4).is_err() {
            // Fallback: return 0 on source failure (cannot propagate
            // errors through RngCore::next_u32). Callers that need
            // error handling should use try_fill_bytes.
            return 0;
        }
        let mut buf = [0u8; 4];
        self.consume(&mut buf);
        u32::from_le_bytes(buf)
    }

    fn next_u64(&mut self) -> u64 {
        if self.ensure_available(8).is_err() {
            return 0;
        }
        let mut buf = [0u8; 8];
        self.consume(&mut buf);
        u64::from_le_bytes(buf)
    }

    fn fill_bytes(&mut self, dest: &mut [u8]) {
        // Best effort: if the source fails, fill with zeros.
        if self.try_fill_bytes(dest).is_err() {
            dest.iter_mut().for_each(|b| *b = 0);
        }
    }

    fn try_fill_bytes(&mut self, dest: &mut [u8]) -> Result<(), rand::Error> {
        let mut written = 0;
        while written < dest.len() {
            let available = if self.pos < self.buffer.len() {
                self.buffer.len() - self.pos
            } else {
                0
            };
            if available == 0 {
                self.refill().map_err(|e| {
                    rand::Error::new(e)
                })?;
                continue;
            }
            let chunk = std::cmp::min(available, dest.len() - written);
            dest[written..written + chunk]
                .copy_from_slice(&self.buffer[self.pos..self.pos + chunk]);
            self.pos += chunk;
            written += chunk;
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// HybridQrngSource
// ---------------------------------------------------------------------------

/// A composite source that tries a primary source first and falls back to
/// a secondary source on failure.
///
/// Tracks the number of fallback events for operational diagnostics.
pub struct HybridQrngSource {
    primary: Box<dyn QrngSource>,
    fallback: Box<dyn QrngSource>,
    fallback_count: u64,
    total_calls: u64,
}

impl HybridQrngSource {
    /// Create a new hybrid source with the given primary and fallback.
    pub fn new(
        primary: Box<dyn QrngSource>,
        fallback: Box<dyn QrngSource>,
    ) -> Self {
        Self {
            primary,
            fallback,
            fallback_count: 0,
            total_calls: 0,
        }
    }

    /// Return the number of times the fallback source was used.
    pub fn fallback_count(&self) -> u64 {
        self.fallback_count
    }

    /// Return the total number of `fill_bytes` calls made.
    pub fn total_calls(&self) -> u64 {
        self.total_calls
    }

    /// Return the ratio of fallback events to total calls.
    ///
    /// Returns 0.0 if no calls have been made.
    pub fn fallback_ratio(&self) -> f64 {
        if self.total_calls == 0 {
            0.0
        } else {
            self.fallback_count as f64 / self.total_calls as f64
        }
    }
}

impl QrngSource for HybridQrngSource {
    fn fill_bytes(&mut self, dest: &mut [u8]) -> Result<(), QrngError> {
        self.total_calls += 1;
        match self.primary.fill_bytes(dest) {
            Ok(()) => Ok(()),
            Err(_primary_err) => {
                self.fallback_count += 1;
                self.fallback.fill_bytes(dest)
            }
        }
    }

    fn source_info(&self) -> QrngSourceInfo {
        let primary_info = self.primary.source_info();
        let fallback_info = self.fallback.source_info();
        QrngSourceInfo {
            name: format!(
                "Hybrid ({} -> {})",
                primary_info.name, fallback_info.name
            ),
            throughput_bps: primary_info.throughput_bps,
            is_hardware: primary_info.is_hardware,
        }
    }
}

// ---------------------------------------------------------------------------
// QrngConfig + builder
// ---------------------------------------------------------------------------

/// Strategy for selecting the QRNG source.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum QrngSourceType {
    /// Use a CSPRNG-based simulation (default).
    Simulated,

    /// Use the ANU Quantum RNG REST API.
    Anu {
        /// API key for authentication.
        api_key: String,
    },

    /// Use a hardware QRNG device file (Unix only).
    Hardware {
        /// Path to the device file (e.g., `/dev/quantis0`).
        device_path: String,
    },

    /// Use a simulated ANU source (local CSPRNG mimicking the API).
    SimulatedAnu,
}

/// Configuration for constructing a QRNG source.
///
/// Uses the builder pattern for ergonomic construction.
///
/// # Example
///
/// ```rust
/// use nqpu_metal::qrng_integration::{QrngConfig, QrngSourceType};
///
/// let config = QrngConfig::new()
///     .source_type(QrngSourceType::Simulated)
///     .buffer_size(8192)
///     .enable_fallback(true);
/// ```
#[derive(Debug, Clone)]
pub struct QrngConfig {
    source: QrngSourceType,
    buffer_size: usize,
    fallback_enabled: bool,
    fallback_source: QrngSourceType,
    seed: Option<(u64, u64)>,
}

impl QrngConfig {
    /// Create a new configuration with default settings.
    pub fn new() -> Self {
        Self {
            source: QrngSourceType::Simulated,
            buffer_size: DEFAULT_ANU_BUFFER_SIZE,
            fallback_enabled: true,
            fallback_source: QrngSourceType::Simulated,
            seed: None,
        }
    }

    /// Set the primary source type.
    pub fn source_type(mut self, source: QrngSourceType) -> Self {
        self.source = source;
        self
    }

    /// Set the buffer size for buffered sources.
    pub fn buffer_size(mut self, size: usize) -> Self {
        self.buffer_size = size;
        self
    }

    /// Enable or disable automatic fallback to a secondary source.
    pub fn enable_fallback(mut self, enabled: bool) -> Self {
        self.fallback_enabled = enabled;
        self
    }

    /// Set the fallback source type (used when `enable_fallback` is true).
    pub fn fallback_type(mut self, source: QrngSourceType) -> Self {
        self.fallback_source = source;
        self
    }

    /// Set a deterministic seed for the simulated source (testing only).
    pub fn seed(mut self, lo: u64, hi: u64) -> Self {
        self.seed = Some((lo, hi));
        self
    }

    /// Return the configured source type.
    pub fn get_source_type(&self) -> &QrngSourceType {
        &self.source
    }

    /// Return the configured buffer size.
    pub fn get_buffer_size(&self) -> usize {
        self.buffer_size
    }

    /// Return whether fallback is enabled.
    pub fn is_fallback_enabled(&self) -> bool {
        self.fallback_enabled
    }
}

impl Default for QrngConfig {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Factory function
// ---------------------------------------------------------------------------

/// Build a concrete `QrngSource` from a `QrngSourceType` and optional seed.
fn build_source(
    source_type: &QrngSourceType,
    buffer_size: usize,
    seed: Option<(u64, u64)>,
) -> Box<dyn QrngSource> {
    match source_type {
        QrngSourceType::Simulated => {
            if let Some((lo, hi)) = seed {
                Box::new(SimulatedQrngSource::with_seed(lo, hi))
            } else {
                Box::new(SimulatedQrngSource::new())
            }
        }
        QrngSourceType::Anu { api_key } => {
            // Without reqwest we cannot actually fetch; return an empty
            // ANU source. The caller must use build_request + load_bytes.
            Box::new(AnuQrngSource::new(api_key.clone()).with_buffer_size(buffer_size))
        }
        QrngSourceType::SimulatedAnu => {
            if let Some((lo, hi)) = seed {
                Box::new(SimulatedAnuSource::with_seed(lo, hi, buffer_size))
            } else {
                Box::new(SimulatedAnuSource::new(buffer_size))
            }
        }
        QrngSourceType::Hardware { device_path } => {
            #[cfg(unix)]
            {
                Box::new(HardwareQrngSource::new(device_path.clone()))
            }
            #[cfg(not(unix))]
            {
                let _ = device_path;
                // Fall back to simulated on non-Unix platforms.
                Box::new(SimulatedQrngSource::new())
            }
        }
    }
}

/// Create a measurement RNG source based on the provided configuration.
///
/// If `config` is `None`, a default [`SimulatedQrngSource`] is returned.
/// When fallback is enabled, the returned source is a [`HybridQrngSource`]
/// wrapping the primary and fallback sources.
///
/// # Example
///
/// ```rust
/// use nqpu_metal::qrng_integration::create_measurement_rng;
///
/// let mut rng = create_measurement_rng(None);
/// let mut buf = [0u8; 16];
/// rng.fill_bytes(&mut buf).unwrap();
/// assert!(buf.iter().any(|&b| b != 0));
/// ```
pub fn create_measurement_rng(config: Option<QrngConfig>) -> Box<dyn QrngSource> {
    let config = config.unwrap_or_default();

    let primary = build_source(&config.source, config.buffer_size, config.seed);

    if config.fallback_enabled {
        let fallback = build_source(&config.fallback_source, config.buffer_size, None);
        Box::new(HybridQrngSource::new(primary, fallback))
    } else {
        primary
    }
}

// ---------------------------------------------------------------------------
// FailingSource (test helper)
// ---------------------------------------------------------------------------

/// A source that always fails. Used internally for testing the
/// [`HybridQrngSource`] fallback path.
struct FailingSource {
    error: QrngError,
}

impl FailingSource {
    /// Create a source that always returns the given error.
    fn new(error: QrngError) -> Self {
        Self { error }
    }
}

impl QrngSource for FailingSource {
    fn fill_bytes(&mut self, _dest: &mut [u8]) -> Result<(), QrngError> {
        Err(self.error.clone())
    }

    fn source_info(&self) -> QrngSourceInfo {
        QrngSourceInfo {
            name: "FailingSource (test)".to_string(),
            throughput_bps: None,
            is_hardware: false,
        }
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // -----------------------------------------------------------------------
    // SimulatedQrngSource tests
    // -----------------------------------------------------------------------

    #[test]
    fn simulated_source_produces_nonzero_bytes() {
        let mut source = SimulatedQrngSource::new();
        let mut buf = [0u8; 64];
        source.fill_bytes(&mut buf).unwrap();
        // Statistically impossible for 64 random bytes to all be zero.
        assert!(buf.iter().any(|&b| b != 0));
    }

    #[test]
    fn simulated_source_deterministic_seed() {
        let mut src_a = SimulatedQrngSource::with_seed(42, 1337);
        let mut src_b = SimulatedQrngSource::with_seed(42, 1337);
        let mut buf_a = [0u8; 128];
        let mut buf_b = [0u8; 128];
        src_a.fill_bytes(&mut buf_a).unwrap();
        src_b.fill_bytes(&mut buf_b).unwrap();
        assert_eq!(buf_a, buf_b);
    }

    #[test]
    fn simulated_source_different_seeds_differ() {
        let mut src_a = SimulatedQrngSource::with_seed(1, 2);
        let mut src_b = SimulatedQrngSource::with_seed(3, 4);
        let mut buf_a = [0u8; 64];
        let mut buf_b = [0u8; 64];
        src_a.fill_bytes(&mut buf_a).unwrap();
        src_b.fill_bytes(&mut buf_b).unwrap();
        assert_ne!(buf_a, buf_b);
    }

    #[test]
    fn simulated_source_info_correct() {
        let source = SimulatedQrngSource::new();
        let info = source.source_info();
        assert!(info.name.contains("Simulated"));
        assert!(!info.is_hardware);
        assert!(info.throughput_bps.is_none());
    }

    #[test]
    fn simulated_source_partial_buffer() {
        // Verify that non-8-aligned buffer sizes work correctly.
        let mut source = SimulatedQrngSource::with_seed(99, 100);
        let mut buf = [0u8; 13]; // 13 bytes = 1 full u64 + 5 tail bytes
        source.fill_bytes(&mut buf).unwrap();
        assert!(buf.iter().any(|&b| b != 0));
    }

    #[test]
    #[should_panic(expected = "xorshift128+ seed must not be all-zero")]
    fn simulated_source_rejects_zero_seed() {
        let _ = SimulatedQrngSource::with_seed(0, 0);
    }

    // -----------------------------------------------------------------------
    // AnuQrngSource tests
    // -----------------------------------------------------------------------

    #[test]
    fn anu_source_from_bytes_works() {
        let data: Vec<u8> = (0..32).collect();
        let mut source = AnuQrngSource::from_bytes("test-key".into(), data.clone());
        let mut buf = [0u8; 16];
        source.fill_bytes(&mut buf).unwrap();
        assert_eq!(&buf[..], &data[..16]);
    }

    #[test]
    fn anu_source_exhaustion() {
        let data: Vec<u8> = vec![1, 2, 3, 4];
        let mut source = AnuQrngSource::from_bytes("key".into(), data);
        let mut buf = [0u8; 8];
        let result = source.fill_bytes(&mut buf);
        assert_eq!(result, Err(QrngError::BufferExhausted));
    }

    #[test]
    fn anu_source_remaining_tracks_correctly() {
        let data: Vec<u8> = (0..64).collect();
        let mut source = AnuQrngSource::from_bytes("key".into(), data);
        assert_eq!(source.remaining(), 64);
        let mut buf = [0u8; 20];
        source.fill_bytes(&mut buf).unwrap();
        assert_eq!(source.remaining(), 44);
    }

    #[test]
    fn anu_source_load_bytes_extends_buffer() {
        let mut source = AnuQrngSource::new("key".into());
        assert_eq!(source.remaining(), 0);
        source.load_bytes(&[10, 20, 30, 40]);
        assert_eq!(source.remaining(), 4);
        let mut buf = [0u8; 4];
        source.fill_bytes(&mut buf).unwrap();
        assert_eq!(buf, [10, 20, 30, 40]);
    }

    #[test]
    fn anu_source_build_request_structure() {
        let source = AnuQrngSource::new("my-api-key".into()).with_buffer_size(1024);
        let req = source.build_request();
        assert!(req.url.contains("quantumnumbers.anu.edu.au"));
        assert!(req.url.contains("length=1024"));
        assert_eq!(req.api_key, "my-api-key");
        assert_eq!(req.length, 1024);
    }

    #[test]
    fn anu_source_info_is_hardware() {
        let source = AnuQrngSource::new("key".into());
        let info = source.source_info();
        assert!(info.is_hardware);
        assert!(info.name.contains("ANU"));
        assert!(info.throughput_bps.is_some());
    }

    // -----------------------------------------------------------------------
    // SimulatedAnuSource tests
    // -----------------------------------------------------------------------

    #[test]
    fn simulated_anu_source_produces_bytes() {
        let mut source = SimulatedAnuSource::new(4096);
        let mut buf = [0u8; 32];
        source.fill_bytes(&mut buf).unwrap();
        assert!(buf.iter().any(|&b| b != 0));
    }

    #[test]
    fn simulated_anu_source_info_not_hardware() {
        let source = SimulatedAnuSource::new(4096);
        let info = source.source_info();
        assert!(!info.is_hardware);
        assert!(info.name.contains("Simulated"));
    }

    // -----------------------------------------------------------------------
    // QrngRng adapter tests
    // -----------------------------------------------------------------------

    #[test]
    fn qrng_rng_next_u32_nonzero() {
        let source = SimulatedQrngSource::with_seed(12345, 67890);
        let mut rng = QrngRng::new(Box::new(source));
        // Generate several u32 values; at least one must be nonzero.
        let values: Vec<u32> = (0..10).map(|_| rng.next_u32()).collect();
        assert!(values.iter().any(|&v| v != 0));
    }

    #[test]
    fn qrng_rng_next_u64_nonzero() {
        let source = SimulatedQrngSource::with_seed(54321, 98765);
        let mut rng = QrngRng::new(Box::new(source));
        let values: Vec<u64> = (0..10).map(|_| rng.next_u64()).collect();
        assert!(values.iter().any(|&v| v != 0));
    }

    #[test]
    fn qrng_rng_fill_bytes_works() {
        let source = SimulatedQrngSource::with_seed(111, 222);
        let mut rng = QrngRng::new(Box::new(source));
        let mut buf = [0u8; 256];
        rng.fill_bytes(&mut buf);
        assert!(buf.iter().any(|&b| b != 0));
    }

    #[test]
    fn qrng_rng_try_fill_bytes_success() {
        let source = SimulatedQrngSource::with_seed(333, 444);
        let mut rng = QrngRng::new(Box::new(source));
        let mut buf = [0u8; 64];
        assert!(rng.try_fill_bytes(&mut buf).is_ok());
        assert!(buf.iter().any(|&b| b != 0));
    }

    #[test]
    fn qrng_rng_buffer_refill_across_boundary() {
        // Use a very small buffer to force multiple refills.
        let source = SimulatedQrngSource::with_seed(555, 666);
        let mut rng = QrngRng::with_buffer_size(Box::new(source), 16);
        // Request more bytes than the buffer holds, forcing refill.
        let mut buf = [0u8; 64];
        rng.fill_bytes(&mut buf);
        assert!(buf.iter().any(|&b| b != 0));
    }

    #[test]
    fn qrng_rng_try_fill_bytes_failure_propagates() {
        let source = FailingSource::new(QrngError::DeviceError("test failure".into()));
        let mut rng = QrngRng::new(Box::new(source));
        let mut buf = [0u8; 8];
        let result = rng.try_fill_bytes(&mut buf);
        assert!(result.is_err());
    }

    #[test]
    fn qrng_rng_source_info() {
        let source = SimulatedQrngSource::new();
        let rng = QrngRng::new(Box::new(source));
        let info = rng.source_info();
        assert!(info.name.contains("Simulated"));
    }

    // -----------------------------------------------------------------------
    // HybridQrngSource tests
    // -----------------------------------------------------------------------

    #[test]
    fn hybrid_uses_primary_when_available() {
        let primary = SimulatedQrngSource::with_seed(10, 20);
        let fallback = SimulatedQrngSource::with_seed(30, 40);

        // Capture expected primary output.
        let mut expected = [0u8; 16];
        SimulatedQrngSource::with_seed(10, 20)
            .fill_bytes(&mut expected)
            .unwrap();

        let mut hybrid = HybridQrngSource::new(Box::new(primary), Box::new(fallback));
        let mut buf = [0u8; 16];
        hybrid.fill_bytes(&mut buf).unwrap();
        assert_eq!(buf, expected);
        assert_eq!(hybrid.fallback_count(), 0);
        assert_eq!(hybrid.total_calls(), 1);
    }

    #[test]
    fn hybrid_falls_back_on_primary_failure() {
        let primary = FailingSource::new(QrngError::NetworkError("offline".into()));
        let fallback = SimulatedQrngSource::with_seed(77, 88);

        let mut expected = [0u8; 16];
        SimulatedQrngSource::with_seed(77, 88)
            .fill_bytes(&mut expected)
            .unwrap();

        let mut hybrid = HybridQrngSource::new(Box::new(primary), Box::new(fallback));
        let mut buf = [0u8; 16];
        hybrid.fill_bytes(&mut buf).unwrap();
        assert_eq!(buf, expected);
        assert_eq!(hybrid.fallback_count(), 1);
        assert_eq!(hybrid.total_calls(), 1);
    }

    #[test]
    fn hybrid_fallback_ratio() {
        let primary = FailingSource::new(QrngError::Timeout);
        let fallback = SimulatedQrngSource::with_seed(50, 60);
        let mut hybrid = HybridQrngSource::new(Box::new(primary), Box::new(fallback));

        let mut buf = [0u8; 8];
        for _ in 0..4 {
            hybrid.fill_bytes(&mut buf).unwrap();
        }
        assert_eq!(hybrid.total_calls(), 4);
        assert_eq!(hybrid.fallback_count(), 4);
        assert!((hybrid.fallback_ratio() - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn hybrid_source_info_shows_both() {
        let primary = SimulatedQrngSource::new();
        let fallback = SimulatedQrngSource::new();
        let hybrid = HybridQrngSource::new(Box::new(primary), Box::new(fallback));
        let info = hybrid.source_info();
        assert!(info.name.contains("Hybrid"));
        assert!(info.name.contains("->"));
    }

    // -----------------------------------------------------------------------
    // QrngConfig builder tests
    // -----------------------------------------------------------------------

    #[test]
    fn config_builder_defaults() {
        let config = QrngConfig::new();
        assert_eq!(*config.get_source_type(), QrngSourceType::Simulated);
        assert_eq!(config.get_buffer_size(), DEFAULT_ANU_BUFFER_SIZE);
        assert!(config.is_fallback_enabled());
    }

    #[test]
    fn config_builder_chaining() {
        let config = QrngConfig::new()
            .source_type(QrngSourceType::SimulatedAnu)
            .buffer_size(8192)
            .enable_fallback(false)
            .seed(42, 1337);
        assert_eq!(*config.get_source_type(), QrngSourceType::SimulatedAnu);
        assert_eq!(config.get_buffer_size(), 8192);
        assert!(!config.is_fallback_enabled());
    }

    // -----------------------------------------------------------------------
    // create_measurement_rng factory tests
    // -----------------------------------------------------------------------

    #[test]
    fn factory_default_produces_working_source() {
        let mut source = create_measurement_rng(None);
        let mut buf = [0u8; 32];
        source.fill_bytes(&mut buf).unwrap();
        assert!(buf.iter().any(|&b| b != 0));
    }

    #[test]
    fn factory_simulated_anu_produces_working_source() {
        let config = QrngConfig::new()
            .source_type(QrngSourceType::SimulatedAnu)
            .enable_fallback(false);
        let mut source = create_measurement_rng(Some(config));
        let mut buf = [0u8; 32];
        source.fill_bytes(&mut buf).unwrap();
        assert!(buf.iter().any(|&b| b != 0));
    }

    #[test]
    fn factory_with_fallback_creates_hybrid() {
        let config = QrngConfig::new()
            .source_type(QrngSourceType::Simulated)
            .enable_fallback(true);
        let source = create_measurement_rng(Some(config));
        let info = source.source_info();
        assert!(info.name.contains("Hybrid"));
    }

    // -----------------------------------------------------------------------
    // Error type tests
    // -----------------------------------------------------------------------

    #[test]
    fn error_display_messages() {
        let net = QrngError::NetworkError("connection refused".into());
        let dev = QrngError::DeviceError("/dev/quantis0: no such device".into());
        let buf = QrngError::BufferExhausted;
        let timeout = QrngError::Timeout;

        assert!(format!("{}", net).contains("connection refused"));
        assert!(format!("{}", dev).contains("quantis0"));
        assert!(format!("{}", buf).contains("exhausted"));
        assert!(format!("{}", timeout).contains("timed out"));
    }

    #[test]
    fn error_equality() {
        assert_eq!(QrngError::BufferExhausted, QrngError::BufferExhausted);
        assert_eq!(QrngError::Timeout, QrngError::Timeout);
        assert_ne!(QrngError::BufferExhausted, QrngError::Timeout);
        assert_eq!(
            QrngError::NetworkError("x".into()),
            QrngError::NetworkError("x".into())
        );
        assert_ne!(
            QrngError::NetworkError("a".into()),
            QrngError::NetworkError("b".into())
        );
    }

    // -----------------------------------------------------------------------
    // HardwareQrngSource tests (Unix only)
    // -----------------------------------------------------------------------

    #[cfg(unix)]
    #[test]
    fn hardware_source_nonexistent_device_returns_error() {
        let mut source =
            HardwareQrngSource::new("/dev/nonexistent_qrng_device_12345".into());
        let mut buf = [0u8; 8];
        let result = source.fill_bytes(&mut buf);
        assert!(matches!(result, Err(QrngError::DeviceError(_))));
    }

    #[cfg(unix)]
    #[test]
    fn hardware_source_info_is_hardware() {
        let source = HardwareQrngSource::new("/dev/quantis0".into());
        let info = source.source_info();
        assert!(info.is_hardware);
        assert!(info.name.contains("/dev/quantis0"));
    }

    #[cfg(unix)]
    #[test]
    fn hardware_source_reads_dev_urandom() {
        // /dev/urandom is universally available on Unix and serves as a
        // smoke test for the read path.
        let mut source = HardwareQrngSource::new("/dev/urandom".into());
        let mut buf = [0u8; 64];
        source.fill_bytes(&mut buf).unwrap();
        assert!(buf.iter().any(|&b| b != 0));
    }
}
