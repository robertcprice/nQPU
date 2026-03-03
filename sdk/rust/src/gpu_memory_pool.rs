//! GPU Memory Pool for Efficient Resource Management
//!
//! This module provides a memory pool system for Metal GPU buffers,
//! reducing allocation overhead and improving performance through:
//! - Pre-allocated buffer pools
//! - Smart buffer reuse
//! - Memory fragmentation reduction
//! - Automatic memory management

use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex};

/// GPU memory pool entry
#[derive(Clone, Debug)]
struct PoolEntry {
    size: usize,
    in_use: bool,
}

/// A pool of GPU buffers for efficient memory management
#[derive(Clone)]
pub struct GpuMemoryPool {
    entries: Arc<Mutex<Vec<PoolEntry>>>,
    total_memory: usize,
    used_memory: Arc<Mutex<usize>>,
    max_pool_size: usize,
}

impl GpuMemoryPool {
    /// Create a new GPU memory pool
    pub fn new(total_memory: usize, max_pool_size: usize) -> Self {
        GpuMemoryPool {
            entries: Arc::new(Mutex::new(Vec::new())),
            total_memory,
            used_memory: Arc::new(Mutex::new(0)),
            max_pool_size,
        }
    }

    /// Allocate memory from the pool
    pub fn allocate(&self, size: usize) -> Option<PoolAllocation> {
        let mut entries = self.entries.lock().ok()?;
        let mut used = self.used_memory.lock().ok()?;

        // Check if we have enough memory
        if *used + size > self.total_memory {
            return None;
        }

        // Try to find a free entry of the right size
        for (i, entry) in entries.iter().enumerate() {
            if !entry.in_use && entry.size == size {
                entries[i].in_use = true;
                *used += size;
                return Some(PoolAllocation {
                    index: i,
                    size,
                    pool: self.clone(),
                });
            }
        }

        // Check pool size limit
        if entries.len() >= self.max_pool_size {
            return None;
        }

        // Create new entry
        let index = entries.len();
        entries.push(PoolEntry { size, in_use: true });
        *used += size;

        Some(PoolAllocation {
            index,
            size,
            pool: self.clone(),
        })
    }

    /// Deallocate memory back to the pool
    fn deallocate(&self, index: usize, size: usize) {
        let mut entries = self.entries.lock().unwrap();
        let mut used = self.used_memory.lock().unwrap();

        if index < entries.len() {
            entries[index].in_use = false;
            *used = used.saturating_sub(size);
        }
    }

    /// Get current memory usage statistics
    pub fn stats(&self) -> MemoryStats {
        let entries = self.entries.lock().unwrap();
        let used = *self.used_memory.lock().unwrap();

        let allocated_entries = entries.iter().filter(|e| e.in_use).count();
        let total_entries = entries.len();

        MemoryStats {
            total_memory: self.total_memory,
            used_memory: used,
            free_memory: self.total_memory - used,
            allocated_entries,
            free_entries: total_entries - allocated_entries,
            total_entries,
        }
    }

    /// Compact the pool by removing unused entries
    pub fn compact(&self) {
        let mut entries = self.entries.lock().unwrap();
        entries.retain(|e| e.in_use);
    }

    /// Reset the pool, clearing all allocations
    pub fn reset(&self) {
        let mut entries = self.entries.lock().unwrap();
        let mut used = self.used_memory.lock().unwrap();
        entries.clear();
        *used = 0;
    }
}

/// Memory pool allocation handle
#[derive(Clone)]
pub struct PoolAllocation {
    index: usize,
    size: usize,
    pool: GpuMemoryPool,
}

impl PoolAllocation {
    pub fn size(&self) -> usize {
        self.size
    }
}

impl Drop for PoolAllocation {
    fn drop(&mut self) {
        self.pool.deallocate(self.index, self.size);
    }
}

/// Memory statistics
#[derive(Clone, Debug)]
pub struct MemoryStats {
    pub total_memory: usize,
    pub used_memory: usize,
    pub free_memory: usize,
    pub allocated_entries: usize,
    pub free_entries: usize,
    pub total_entries: usize,
}

impl MemoryStats {
    pub fn usage_percentage(&self) -> f64 {
        (self.used_memory as f64 / self.total_memory as f64) * 100.0
    }

    pub fn fragmentation_ratio(&self) -> f64 {
        if self.total_entries == 0 {
            return 0.0;
        }
        self.free_entries as f64 / self.total_entries as f64
    }
}

/// Smart buffer cache for reusing GPU buffers
pub struct BufferCache {
    cache: HashMap<usize, VecDeque<BufferEntry>>,
    max_cached_per_size: usize,
}

#[derive(Clone)]
struct BufferEntry {
    size: usize,
    /// Actual CPU-side buffer data for caching GPU buffer contents.
    data: Vec<u8>,
}

impl BufferCache {
    pub fn new(max_cached_per_size: usize) -> Self {
        BufferCache {
            cache: HashMap::new(),
            max_cached_per_size,
        }
    }

    /// Get a buffer from cache or create new one.
    ///
    /// If a previously released buffer of the same size exists in the cache,
    /// it is reused (the backing `Vec<u8>` is recycled). Otherwise a fresh
    /// zero-initialized buffer is allocated.
    pub fn acquire(&mut self, size: usize) -> CachedBuffer {
        if let Some(bucket) = self.cache.get_mut(&size) {
            if let Some(entry) = bucket.pop_front() {
                return CachedBuffer {
                    size,
                    data: entry.data,
                    returned: false,
                };
            }
        }

        CachedBuffer {
            size,
            data: vec![0u8; size],
            returned: false,
        }
    }

    /// Return a buffer to the cache.
    ///
    /// The backing memory is preserved so it can be reused by a future
    /// `acquire` call of the same size, avoiding reallocation.
    pub fn release(&mut self, mut buffer: CachedBuffer) {
        let bucket = self.cache.entry(buffer.size).or_insert_with(VecDeque::new);

        if bucket.len() < self.max_cached_per_size {
            let data = std::mem::take(&mut buffer.data);
            bucket.push_back(BufferEntry {
                size: buffer.size,
                data,
            });
            buffer.returned = true;
        }
        // If the bucket is full, the buffer's Vec<u8> is simply dropped.
    }

    /// Clear the cache
    pub fn clear(&mut self) {
        self.cache.clear();
    }
}

/// Cached buffer handle with actual backing storage.
///
/// The `data` field holds the CPU-side buffer bytes. When the buffer is
/// returned to the cache via `BufferCache::release`, the allocation is
/// recycled rather than freed.
pub struct CachedBuffer {
    size: usize,
    /// Actual backing storage.
    data: Vec<u8>,
    returned: bool,
}

impl CachedBuffer {
    pub fn size(&self) -> usize {
        self.size
    }

    /// Get an immutable slice of the buffer contents.
    pub fn as_slice(&self) -> &[u8] {
        &self.data
    }

    /// Get a mutable slice of the buffer contents.
    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        &mut self.data
    }
}

impl Drop for CachedBuffer {
    fn drop(&mut self) {
        self.returned = true;
        // The Vec<u8> is dropped here if the buffer was not returned
        // to the cache via `release`. If it was released, the data was
        // moved out before drop runs and `self.data` is an empty Vec.
    }
}

/// Memory-aligned buffer allocator for GPU operations.
///
/// Allocates buffers whose starting address is aligned to the specified
/// boundary. This is important for GPU DMA transfers which often require
/// page-aligned or cache-line-aligned memory.
pub struct AlignedAllocator {
    alignment: usize,
}

impl AlignedAllocator {
    pub fn new(alignment: usize) -> Self {
        assert!(alignment > 0 && alignment.is_power_of_two(),
            "Alignment must be a positive power of two");
        AlignedAllocator { alignment }
    }

    /// Allocate aligned memory of at least `size` bytes.
    ///
    /// The returned buffer is zero-initialized and its length is rounded
    /// up to the next multiple of the alignment.
    pub fn allocate(&self, size: usize) -> Option<AlignedMemory> {
        if size == 0 {
            return None;
        }
        let aligned_size = (size + self.alignment - 1) / self.alignment * self.alignment;

        // Use the global allocator with the requested layout alignment.
        // This ensures the memory address is properly aligned for GPU operations.
        let layout = std::alloc::Layout::from_size_align(aligned_size, self.alignment).ok()?;
        let ptr = unsafe { std::alloc::alloc_zeroed(layout) };
        if ptr.is_null() {
            return None;
        }

        Some(AlignedMemory {
            ptr,
            layout,
            size: aligned_size,
            alignment: self.alignment,
        })
    }
}

/// Aligned memory handle with actual backing storage.
///
/// The memory is freed when this handle is dropped. The pointer is
/// guaranteed to be aligned to `alignment` bytes.
pub struct AlignedMemory {
    ptr: *mut u8,
    layout: std::alloc::Layout,
    size: usize,
    alignment: usize,
}

// SAFETY: The raw pointer is exclusively owned by AlignedMemory.
unsafe impl Send for AlignedMemory {}
unsafe impl Sync for AlignedMemory {}

impl AlignedMemory {
    pub fn size(&self) -> usize {
        self.size
    }

    pub fn alignment(&self) -> usize {
        self.alignment
    }

    /// Get an immutable slice of the allocated memory.
    pub fn as_slice(&self) -> &[u8] {
        unsafe { std::slice::from_raw_parts(self.ptr, self.size) }
    }

    /// Get a mutable slice of the allocated memory.
    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        unsafe { std::slice::from_raw_parts_mut(self.ptr, self.size) }
    }

    /// Get the raw pointer to the aligned memory.
    pub fn as_ptr(&self) -> *const u8 {
        self.ptr
    }

    /// Get a mutable raw pointer to the aligned memory.
    pub fn as_mut_ptr(&mut self) -> *mut u8 {
        self.ptr
    }
}

impl Drop for AlignedMemory {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            unsafe {
                std::alloc::dealloc(self.ptr, self.layout);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_memory_pool_allocation() {
        let pool = GpuMemoryPool::new(1024 * 1024, 100);

        let alloc1 = pool.allocate(256);
        assert!(alloc1.is_some());

        let alloc2 = pool.allocate(512);
        assert!(alloc2.is_some());

        let stats = pool.stats();
        assert_eq!(stats.used_memory, 256 + 512);
        assert_eq!(stats.allocated_entries, 2);
    }

    #[test]
    fn test_memory_pool_reuse() {
        let pool = GpuMemoryPool::new(1024, 100);

        {
            let _alloc1 = pool.allocate(256);
            let stats = pool.stats();
            assert_eq!(stats.used_memory, 256);
        }

        // After drop, memory should be freed
        let stats = pool.stats();
        assert_eq!(stats.used_memory, 0);
        assert_eq!(stats.allocated_entries, 0);
        assert_eq!(stats.free_entries, 1); // Entry is now free
    }

    #[test]
    fn test_memory_stats() {
        let pool = GpuMemoryPool::new(1000, 100);

        let _alloc1 = pool.allocate(250);
        let _alloc2 = pool.allocate(250);

        let stats = pool.stats();
        assert_eq!(stats.usage_percentage(), 50.0);
    }

    #[test]
    fn test_buffer_cache() {
        let mut cache = BufferCache::new(10);

        let buf1 = cache.acquire(256);
        assert_eq!(buf1.size(), 256);

        cache.release(buf1);

        let buf2 = cache.acquire(256);
        assert_eq!(buf2.size(), 256);
    }

    #[test]
    fn test_aligned_allocation() {
        let allocator = AlignedAllocator::new(256);

        let mem = allocator.allocate(100);
        assert!(mem.is_some());

        let mem = mem.unwrap();
        assert_eq!(mem.alignment(), 256);
        assert_eq!(mem.size(), 256); // Should be aligned up
    }
}
