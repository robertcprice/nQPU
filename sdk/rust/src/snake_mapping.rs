// Snake Mapping for 2D Grid Circuits
//
// Maps 2D quantum grid to 1D MPS chain using snake pattern.
// This enables basic 2D circuit simulation with existing MPS infrastructure.


/// 2D grid coordinates
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GridCoord {
    pub x: usize,
    pub y: usize,
}

impl GridCoord {
    pub fn new(x: usize, y: usize) -> Self {
        Self { x, y }
    }
}

/// Snake pattern mapper for 2D grid to 1D chain
///
/// Layout example (4×4):
/// ```text
/// 0  →  1  →  2  →  3
///              ↓
/// 7  ←  6  ←  5  ←  4
///              ↓
/// 8  →  9  → 10 → 11
///              ↓
/// 15 ← 14 ← 13 ← 12
/// ```
#[derive(Debug, Clone)]
pub struct SnakeMapper {
    width: usize,
    height: usize,
    size: usize,
}

impl SnakeMapper {
    /// Create a new snake mapper for width×height grid
    pub fn new(width: usize, height: usize) -> Self {
        let size = width * height;
        Self {
            width,
            height,
            size,
        }
    }

    /// Map 2D coordinate to 1D index in snake pattern
    ///
    /// Even rows: left to right (0, 1, 2, ...)
    /// Odd rows: right to left (width-1, width-2, ...)
    pub fn map_2d_to_1d(&self, x: usize, y: usize) -> usize {
        assert!(x < self.width, "x coordinate out of bounds");
        assert!(y < self.height, "y coordinate out of bounds");

        let row = y;
        let col = if y % 2 == 0 {
            // Even row: left to right
            x
        } else {
            // Odd row: right to left (snake)
            self.width - 1 - x
        };

        row * self.width + col
    }

    /// Map 1D index back to 2D coordinates
    pub fn map_1d_to_2d(&self, index: usize) -> GridCoord {
        assert!(index < self.size, "index out of bounds");

        let row = index / self.width;
        let col_in_row = index % self.width;

        let col = if row % 2 == 0 {
            // Even row: normal column
            col_in_row
        } else {
            // Odd row: reversed column
            self.width - 1 - col_in_row
        };

        GridCoord::new(col, row)
    }

    /// Get total number of qubits in the grid
    pub fn size(&self) -> usize {
        self.size
    }

    /// Get grid dimensions
    pub fn dimensions(&self) -> (usize, usize) {
        (self.width, self.height)
    }

    /// Get nearest neighbors of a 2D coordinate
    ///
    /// Returns up to 4 neighbors (up, down, left, right)
    pub fn get_neighbors(&self, x: usize, y: usize) -> Vec<GridCoord> {
        let mut neighbors = Vec::new();

        // Up
        if y > 0 {
            neighbors.push(GridCoord::new(x, y - 1));
        }
        // Down
        if y < self.height - 1 {
            neighbors.push(GridCoord::new(x, y + 1));
        }
        // Left
        if x > 0 {
            neighbors.push(GridCoord::new(x - 1, y));
        }
        // Right
        if x < self.width - 1 {
            neighbors.push(GridCoord::new(x + 1, y));
        }

        neighbors
    }

    /// Calculate Manhattan distance between two 2D coordinates
    ///
    /// Higher distance = more entanglement in MPS (worse performance)
    pub fn distance(&self, c1: &GridCoord, c2: &GridCoord) -> usize {
        let dx = (c1.x as i32 - c2.x as i32).abs() as usize;
        let dy = (c1.y as i32 - c2.y as i32).abs() as usize;
        dx + dy
    }

    /// Get maximum possible distance in this grid
    pub fn max_distance(&self) -> usize {
        self.width + self.height - 2
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_snake_mapping_basic() {
        let mapper = SnakeMapper::new(4, 4);

        // First row: left to right
        assert_eq!(mapper.map_2d_to_1d(0, 0), 0);
        assert_eq!(mapper.map_2d_to_1d(1, 0), 1);
        assert_eq!(mapper.map_2d_to_1d(2, 0), 2);
        assert_eq!(mapper.map_2d_to_1d(3, 0), 3);

        // Second row: right to left (snake)
        assert_eq!(mapper.map_2d_to_1d(3, 1), 4);
        assert_eq!(mapper.map_2d_to_1d(2, 1), 5);
        assert_eq!(mapper.map_2d_to_1d(1, 1), 6);
        assert_eq!(mapper.map_2d_to_1d(0, 1), 7);

        // Third row: left to right
        assert_eq!(mapper.map_2d_to_1d(0, 2), 8);
        assert_eq!(mapper.map_2d_to_1d(1, 2), 9);

        println!("Snake mapping basic test passed!");
    }

    #[test]
    fn test_inverse_mapping() {
        let mapper = SnakeMapper::new(4, 4);

        for y in 0..4 {
            for x in 0..4 {
                let idx = mapper.map_2d_to_1d(x, y);
                let coord = mapper.map_1d_to_2d(idx);
                assert_eq!(coord.x, x, "Inverse mapping x failed");
                assert_eq!(coord.y, y, "Inverse mapping y failed");
            }
        }

        println!("Inverse mapping test passed!");
    }

    #[test]
    fn test_neighbors() {
        let mapper = SnakeMapper::new(4, 4);

        // Corner: (0,0) has neighbors down and right
        let neighbors = mapper.get_neighbors(0, 0);
        assert_eq!(neighbors.len(), 2);
        assert!(neighbors.contains(&GridCoord::new(1, 0))); // right
        assert!(neighbors.contains(&GridCoord::new(0, 1))); // down

        // Center: (1,1) has all 4 neighbors
        let neighbors = mapper.get_neighbors(1, 1);
        assert_eq!(neighbors.len(), 4);

        // Edge: (0,1) has up, down, right
        let neighbors = mapper.get_neighbors(0, 1);
        assert_eq!(neighbors.len(), 3);

        println!("Neighbors test passed!");
    }

    #[test]
    fn test_distance() {
        let mapper = SnakeMapper::new(10, 10);

        // Same position
        assert_eq!(
            mapper.distance(&GridCoord::new(0, 0), &GridCoord::new(0, 0)),
            0
        );

        // Adjacent
        assert_eq!(
            mapper.distance(&GridCoord::new(0, 0), &GridCoord::new(1, 0)),
            1
        );

        // Diagonal
        assert_eq!(
            mapper.distance(&GridCoord::new(0, 0), &GridCoord::new(1, 1)),
            2
        );

        // Far corners
        let dist = mapper.distance(&GridCoord::new(0, 0), &GridCoord::new(9, 9));
        assert_eq!(dist, 18); // 9 + 9

        println!("Distance test passed!");
    }
}
