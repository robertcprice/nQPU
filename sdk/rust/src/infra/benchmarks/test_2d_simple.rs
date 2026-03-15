//! Simple 2D algorithm tests without Python bindings

use crate::snake_mapping::{GridCoord, SnakeMapper};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_snake_mapper_basic() {
        let mapper = SnakeMapper::new(4, 4);

        // Test basic mapping
        assert_eq!(mapper.map_2d_to_1d(0, 0), 0);
        assert_eq!(mapper.map_2d_to_1d(3, 0), 3);
        assert_eq!(mapper.map_2d_to_1d(0, 1), 7);

        // Test inverse mapping: index 7 is the last position in odd row 1,
        // which snakes right-to-left, so it maps back to (0, 1).
        let coord = mapper.map_1d_to_2d(7);
        assert_eq!(coord.x, 0);
        assert_eq!(coord.y, 1);
    }

    #[test]
    fn test_2d_basic_gates() {
        // This is a compile-only test to verify 2D gate types compile
        // Actual tests require MPS simulator
        use crate::tensor_network::MPSSimulator;

        let mapper = SnakeMapper::new(2, 2);
        let mut sim = MPSSimulator::new(4, None);

        // Test single gates by index
        sim.h(0);
        sim.x(1);
        sim.cnot(1, 2);

        assert_eq!(sim.num_qubits(), 4);
    }
}
