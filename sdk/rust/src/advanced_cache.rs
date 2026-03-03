//! Advanced Caching System for Quantum Computations
//!
//! This module provides intelligent caching for:
//! - Circuit execution results
//! - Gate operations
//! - State vectors
//! - Computed matrices

use std::collections::{hash_map::DefaultHasher, HashMap};
use std::f64::consts::PI;
use std::hash::{Hash, Hasher};
use std::time::{Duration, Instant};

use crate::gates::{Gate, GateType};
use crate::{QuantumState, C64};

/// Cache key for quantum states (uses string signature for simplicity)
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct StateCacheKey {
    pub num_qubits: usize,
    pub gates_signature: String,
}

impl StateCacheKey {
    pub fn from_gates(num_qubits: usize, gates: &[Gate]) -> Self {
        let mut hasher = DefaultHasher::new();

        for gate in gates {
            // Hash gate type as string
            match &gate.gate_type {
                GateType::H => hasher.write_u8(1),
                GateType::X => hasher.write_u8(2),
                GateType::Y => hasher.write_u8(3),
                GateType::Z => hasher.write_u8(4),
                GateType::S => hasher.write_u8(5),
                GateType::T => hasher.write_u8(6),
                GateType::Rx(a) => {
                    hasher.write_u8(7);
                    a.to_bits().hash(&mut hasher);
                }
                GateType::Ry(a) => {
                    hasher.write_u8(8);
                    a.to_bits().hash(&mut hasher);
                }
                GateType::Rz(a) => {
                    hasher.write_u8(9);
                    a.to_bits().hash(&mut hasher);
                }
                GateType::CNOT => hasher.write_u8(10),
                GateType::CZ => hasher.write_u8(11),
                GateType::SWAP => hasher.write_u8(12),
                _ => hasher.write_u8(0),
            }
            gate.targets.hash(&mut hasher);
            gate.controls.hash(&mut hasher);
        }

        StateCacheKey {
            num_qubits,
            gates_signature: format!("{:x}", hasher.finish()),
        }
    }
}

/// Cache entry with metadata
#[derive(Clone)]
pub struct CacheEntry<T> {
    pub data: T,
    pub created_at: Instant,
    pub last_accessed: Instant,
    pub access_count: usize,
    pub size_bytes: usize,
}

impl<T> CacheEntry<T> {
    fn new(data: T, size_bytes: usize) -> Self {
        let now = Instant::now();
        CacheEntry {
            data,
            created_at: now,
            last_accessed: now,
            access_count: 0,
            size_bytes,
        }
    }

    fn mark_accessed(&mut self) {
        self.last_accessed = Instant::now();
        self.access_count += 1;
    }
}

/// Quantum computation cache
pub struct QuantumCache<T> {
    entries: HashMap<StateCacheKey, CacheEntry<T>>,
    max_size_bytes: usize,
    current_size_bytes: usize,
    max_age: Duration,
    ttl: Duration,
}

impl<T: Clone> QuantumCache<T> {
    pub fn new(max_size_bytes: usize, max_age: Duration) -> Self {
        QuantumCache {
            entries: HashMap::new(),
            max_size_bytes,
            current_size_bytes: 0,
            max_age,
            ttl: Duration::from_secs(300), // 5 minutes default TTL
        }
    }

    /// Get a value from the cache
    pub fn get(&mut self, key: &StateCacheKey) -> Option<T> {
        // First check if entry exists and get its created_at time
        let created_at = self.entries.get(key).map(|e| e.created_at);
        let _size_bytes = self.entries.get(key).map(|e| e.size_bytes);

        // Check expiration
        let now = Instant::now();
        if let Some(created) = created_at {
            if now.duration_since(created) > self.ttl {
                if let Some(entry) = self.entries.remove(key) {
                    self.current_size_bytes -= entry.size_bytes;
                }
                return None;
            }
        }

        // Get and mark as accessed
        if let Some(entry) = self.entries.get_mut(key) {
            entry.last_accessed = now;
            entry.access_count += 1;
            Some(entry.data.clone())
        } else {
            None
        }
    }

    /// Insert a value into the cache
    pub fn insert(&mut self, key: StateCacheKey, value: T, size_bytes: usize) {
        // Evict if necessary
        while self.current_size_bytes + size_bytes > self.max_size_bytes {
            self.evict_lru();
        }

        self.current_size_bytes += size_bytes;
        self.entries.insert(key, CacheEntry::new(value, size_bytes));
    }

    /// Remove least recently used entry
    fn evict_lru(&mut self) {
        let lru_key = self
            .entries
            .iter()
            .min_by_key(|(_, entry)| entry.last_accessed)
            .map(|(key, _)| key.clone());

        if let Some(key) = lru_key {
            if let Some(entry) = self.entries.remove(&key) {
                self.current_size_bytes -= entry.size_bytes;
            }
        }
    }

    /// Clear all expired entries
    pub fn clear_expired(&mut self) {
        let now = Instant::now();
        let expired_keys: Vec<_> = self
            .entries
            .iter()
            .filter(|(_, entry)| now.duration_since(entry.created_at) > self.ttl)
            .map(|(key, _)| key.clone())
            .collect();

        for key in expired_keys {
            if let Some(entry) = self.entries.remove(&key) {
                self.current_size_bytes -= entry.size_bytes;
            }
        }
    }

    /// Get cache statistics
    pub fn stats(&self) -> CacheStats {
        let total_accesses: usize = self.entries.values().map(|e| e.access_count).sum();
        let now = Instant::now();

        CacheStats {
            num_entries: self.entries.len(),
            total_size_bytes: self.current_size_bytes,
            max_size_bytes: self.max_size_bytes,
            total_accesses,
            hit_rate: 0.0, // Would need to track misses
            oldest_entry: self
                .entries
                .values()
                .map(|e| now.duration_since(e.created_at))
                .min()
                .unwrap_or(Duration::ZERO),
        }
    }

    /// Clear all entries
    pub fn clear(&mut self) {
        self.entries.clear();
        self.current_size_bytes = 0;
    }
}

/// Cache statistics
#[derive(Clone, Debug)]
pub struct CacheStats {
    pub num_entries: usize,
    pub total_size_bytes: usize,
    pub max_size_bytes: usize,
    pub total_accesses: usize,
    pub hit_rate: f64,
    pub oldest_entry: Duration,
}

/// Thread-safe shared cache (simplified - not actually thread-safe)
/// For full thread safety, use proper synchronization primitives
pub struct SharedQuantumCache<T> {
    inner: QuantumCache<T>,
}

impl<T: Clone> SharedQuantumCache<T> {
    pub fn new(max_size_bytes: usize, max_age: Duration) -> Self {
        SharedQuantumCache {
            inner: QuantumCache::new(max_size_bytes, max_age),
        }
    }

    pub fn get(&mut self, key: &StateCacheKey) -> Option<T> {
        self.inner.get(key)
    }

    pub fn insert(&mut self, key: StateCacheKey, value: T, size_bytes: usize) {
        self.inner.insert(key, value, size_bytes);
    }

    pub fn stats(&self) -> CacheStats {
        self.inner.stats()
    }
}

/// Circuit result cache
pub struct CircuitResultCache {
    state_cache: QuantumCache<QuantumState>,
    matrix_cache: QuantumCache<Vec<Vec<C64>>>,
}

impl CircuitResultCache {
    pub fn new(max_size_mb: usize) -> Self {
        let max_size_bytes = max_size_mb * 1024 * 1024;
        let max_age = Duration::from_secs(3600); // 1 hour

        CircuitResultCache {
            state_cache: QuantumCache::new(max_size_bytes / 2, max_age),
            matrix_cache: QuantumCache::new(max_size_bytes / 2, max_age),
        }
    }

    /// Cache a quantum state result
    pub fn cache_state(&mut self, gates: &[Gate], state: QuantumState) {
        let key = Self::compute_cache_key(gates);
        let size_bytes = state.dim * 16; // 16 bytes per complex number
        self.state_cache.insert(key, state, size_bytes);
    }

    /// Get cached quantum state
    pub fn get_cached_state(&mut self, gates: &[Gate]) -> Option<QuantumState> {
        let key = Self::compute_cache_key(gates);
        self.state_cache.get(&key)
    }

    /// Cache a matrix computation
    pub fn cache_matrix(&mut self, gates: &[Gate], matrix: Vec<Vec<C64>>) {
        let key = Self::compute_cache_key(gates);
        let size_bytes = matrix.len() * matrix.first().map_or(0, |row| row.len()) * 16;
        self.matrix_cache.insert(key, matrix, size_bytes);
    }

    /// Get cached matrix
    pub fn get_cached_matrix(&mut self, gates: &[Gate]) -> Option<Vec<Vec<C64>>> {
        let key = Self::compute_cache_key(gates);
        self.matrix_cache.get(&key)
    }

    /// Compute hash-based cache key from gates
    fn compute_cache_key(gates: &[Gate]) -> StateCacheKey {
        StateCacheKey::from_gates(
            gates
                .iter()
                .flat_map(|g| g.targets.iter().chain(g.controls.iter()))
                .max()
                .map(|m| m + 1)
                .unwrap_or(0),
            gates,
        )
    }

    /// Get cache statistics
    pub fn stats(&self) -> CacheStatistics {
        CacheStatistics {
            state_cache: self.state_cache.stats(),
            matrix_cache: self.matrix_cache.stats(),
        }
    }

    /// Clear all caches
    pub fn clear(&mut self) {
        self.state_cache.clear();
        self.matrix_cache.clear();
    }
}

/// Combined cache statistics
#[derive(Clone, Debug)]
pub struct CacheStatistics {
    pub state_cache: CacheStats,
    pub matrix_cache: CacheStats,
}

/// Precompute and cache common gate sequences
pub struct GateSequenceCache {
    cached_sequences: HashMap<String, Vec<Gate>>,
}

impl GateSequenceCache {
    pub fn new() -> Self {
        let mut cache = GateSequenceCache {
            cached_sequences: HashMap::new(),
        };

        // Pre-populate with common sequences
        cache.populate_common_sequences();
        cache
    }

    fn populate_common_sequences(&mut self) {
        // Bell state preparation
        self.cached_sequences
            .insert("bell_pair".to_string(), vec![Gate::h(0), Gate::cnot(0, 1)]);

        // GHZ state preparation
        self.cached_sequences.insert(
            "ghz_3".to_string(),
            vec![Gate::h(0), Gate::cnot(0, 1), Gate::cnot(1, 2)],
        );

        // QFT (simplified)
        self.cached_sequences.insert(
            "qft_2".to_string(),
            vec![
                Gate::h(0),
                Gate::new(GateType::CR(PI / 4.0), vec![1], vec![0]),
                Gate::h(1),
            ],
        );
    }

    /// Get a cached gate sequence
    pub fn get_sequence(&self, name: &str) -> Option<&[Gate]> {
        self.cached_sequences.get(name).map(|v| v.as_slice())
    }

    /// Add a custom sequence to the cache
    pub fn add_sequence(&mut self, name: String, gates: Vec<Gate>) {
        self.cached_sequences.insert(name, gates);
    }
}

impl Default for GateSequenceCache {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cache_insert_get() {
        let mut cache: QuantumCache<Vec<i32>> = QuantumCache::new(1024, Duration::from_secs(60));

        let key = StateCacheKey {
            num_qubits: 2,
            gates_signature: "test123".to_string(),
        };

        cache.insert(key.clone(), vec![1, 2, 3], 24);

        let result = cache.get(&key);
        assert_eq!(result, Some(vec![1, 2, 3]));
    }

    #[test]
    fn test_cache_eviction() {
        let mut cache: QuantumCache<Vec<i32>> = QuantumCache::new(100, Duration::from_secs(60));

        let key1 = StateCacheKey {
            num_qubits: 1,
            gates_signature: "key1".to_string(),
        };
        let key2 = StateCacheKey {
            num_qubits: 1,
            gates_signature: "key2".to_string(),
        };

        // Insert first entry
        cache.insert(key1.clone(), vec![1; 10], 40);
        assert_eq!(cache.current_size_bytes, 40);

        // Insert second entry that exceeds max size
        cache.insert(key2.clone(), vec![2; 70], 70);

        // First entry should be evicted
        assert!(cache.get(&key1).is_none());
        assert!(cache.get(&key2).is_some());
    }

    #[test]
    fn test_circuit_cache() {
        let mut cache = CircuitResultCache::new(1); // 1 MB

        let gates = vec![Gate::h(0), Gate::x(1)];
        let mut state = QuantumState::new(2);
        crate::GateOperations::h(&mut state, 0);
        crate::GateOperations::x(&mut state, 1);

        cache.cache_state(&gates, state.clone());
        let cached = cache.get_cached_state(&gates);

        assert!(cached.is_some());
        let cached_state = cached.unwrap();
        assert_eq!(cached_state.num_qubits, state.num_qubits);
    }

    #[test]
    fn test_sequence_cache() {
        let cache = GateSequenceCache::new();

        let bell = cache.get_sequence("bell_pair");
        assert!(bell.is_some());
        assert_eq!(bell.unwrap().len(), 2);
    }
}
