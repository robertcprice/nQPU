//! Quantum Chess Engine
//!
//! **BLEEDING EDGE**: A playable quantum chess engine where pieces exist in
//! superposition, can be entangled via quantum splits, and collapse upon
//! measurement (capture). Inspired by Google's Quantum Chess (Chris Cantwell).
//!
//! In classical chess every piece occupies exactly one square. In quantum chess
//! a single piece can be *split* into a superposition of two squares. The game
//! state becomes a weighted superposition of classical board configurations
//! (branches). When a capture is attempted on a square whose occupancy differs
//! across branches, a *measurement* collapses the superposition --- the piece
//! is either there or it is not, with probabilities given by Born's rule.
//!
//! # Core Mechanics
//!
//! 1. **Classical move** -- applied identically to every branch.
//! 2. **Quantum split** -- a piece at square `from` moves to `to_a` in half
//!    the branches and `to_b` in the other half, each with amplitude / sqrt(2).
//! 3. **Measurement / capture** -- when a piece attempts to move to a square
//!    that is occupied in *some* branches but empty in others, the branches
//!    are measured and collapsed. A seeded LCG PRNG ensures determinism.
//! 4. **Branch merging** -- identical board states have their amplitudes summed.
//! 5. **Normalization** -- the sum of |amplitude|^2 is always kept at 1.0.
//!
//! # Example
//!
//! ```rust
//! use nqpu_metal::quantum_chess::*;
//!
//! let mut game = QuantumChessGame::new();
//!
//! // Classical opening: e2 -> e4
//! game.make_move(QuantumChessMove::Classical {
//!     from: Square::from_algebraic("e2"),
//!     to:   Square::from_algebraic("e4"),
//! }).unwrap();
//!
//! // Black responds classically: e7 -> e5
//! game.make_move(QuantumChessMove::Classical {
//!     from: Square::from_algebraic("e7"),
//!     to:   Square::from_algebraic("e5"),
//! }).unwrap();
//!
//! // White quantum-splits the knight into superposition!
//! game.make_move(QuantumChessMove::QuantumSplit {
//!     from: Square::from_algebraic("g1"),
//!     to_a: Square::from_algebraic("f3"),
//!     to_b: Square::from_algebraic("h3"),
//! }).unwrap();
//!
//! assert_eq!(game.num_branches(), 2);
//! ```
//!
//! # References
//!
//! - Cantwell, C. (2019). *Quantum Chess*. <https://quantumchess.net>
//! - Google AI Quantum (2020). Quantum Chess demonstration

use num_complex::Complex64;
use std::collections::HashMap;
use std::fmt;

type C64 = Complex64;

// ============================================================
// LCG PRNG (inline, no external deps)
// ============================================================

/// Advance the LCG state and return the new value.
/// Constants from Knuth's MMIX.
#[inline]
fn lcg_next(state: &mut u64) -> u64 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    *state
}

/// Return a uniform f64 in [0, 1) from the LCG.
#[inline]
fn lcg_f64(state: &mut u64) -> f64 {
    (lcg_next(state) >> 11) as f64 / (1u64 << 53) as f64
}

// ============================================================
// CORE TYPES
// ============================================================

/// A square on the 8x8 board, stored as `row * 8 + col` where row 0 = rank 1.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Square(pub u8);

impl Square {
    /// Create a square from row (0-7) and column (0-7).
    #[inline]
    pub fn new(row: u8, col: u8) -> Self {
        debug_assert!(row < 8 && col < 8);
        Square(row * 8 + col)
    }

    /// Row index (0 = rank 1, 7 = rank 8).
    #[inline]
    pub fn row(self) -> u8 {
        self.0 / 8
    }

    /// Column index (0 = file a, 7 = file h).
    #[inline]
    pub fn col(self) -> u8 {
        self.0 % 8
    }

    /// Parse algebraic notation like "e4".
    pub fn from_algebraic(s: &str) -> Self {
        let bytes = s.as_bytes();
        assert!(bytes.len() == 2, "algebraic notation must be 2 chars");
        let col = bytes[0] - b'a';
        let row = bytes[1] - b'1';
        Square::new(row, col)
    }

    /// Convert to algebraic notation like "e4".
    pub fn to_algebraic(self) -> String {
        let col_char = (b'a' + self.col()) as char;
        let row_char = (b'1' + self.row()) as char;
        format!("{}{}", col_char, row_char)
    }

    /// Whether the index is valid (0..63).
    #[inline]
    pub fn is_valid(self) -> bool {
        self.0 < 64
    }
}

impl fmt::Display for Square {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.to_algebraic())
    }
}

/// The six standard chess piece kinds.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum PieceKind {
    King,
    Queen,
    Rook,
    Bishop,
    Knight,
    Pawn,
}

impl PieceKind {
    /// Single-character representation (uppercase).
    pub fn symbol(self) -> char {
        match self {
            PieceKind::King => 'K',
            PieceKind::Queen => 'Q',
            PieceKind::Rook => 'R',
            PieceKind::Bishop => 'B',
            PieceKind::Knight => 'N',
            PieceKind::Pawn => 'P',
        }
    }
}

impl fmt::Display for PieceKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let name = match self {
            PieceKind::King => "King",
            PieceKind::Queen => "Queen",
            PieceKind::Rook => "Rook",
            PieceKind::Bishop => "Bishop",
            PieceKind::Knight => "Knight",
            PieceKind::Pawn => "Pawn",
        };
        write!(f, "{}", name)
    }
}

/// Player colour.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Color {
    White,
    Black,
}

impl Color {
    /// The other colour.
    #[inline]
    pub fn opposite(self) -> Color {
        match self {
            Color::White => Color::Black,
            Color::Black => Color::White,
        }
    }
}

impl fmt::Display for Color {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Color::White => write!(f, "White"),
            Color::Black => write!(f, "Black"),
        }
    }
}

/// A chess piece with unique identity.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Piece {
    pub kind: PieceKind,
    pub color: Color,
    /// Unique identifier so we can track a piece across branches.
    pub id: u8,
}

impl Piece {
    /// Display character: uppercase for white, lowercase for black.
    pub fn display_char(&self) -> char {
        let c = self.kind.symbol();
        match self.color {
            Color::White => c,
            Color::Black => c.to_ascii_lowercase(),
        }
    }
}

impl fmt::Display for Piece {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.display_char())
    }
}

// ============================================================
// BOARD STATE (one classical configuration)
// ============================================================

/// A single classical board configuration.
#[derive(Clone, Debug)]
pub struct BoardState {
    pub squares: [Option<Piece>; 64],
}

impl BoardState {
    /// Empty board.
    pub fn empty() -> Self {
        BoardState {
            squares: [None; 64],
        }
    }

    /// Standard starting position.
    pub fn standard_setup(id_offset: &mut u8) -> Self {
        let mut board = Self::empty();

        // Helper to place a piece and bump the ID counter.
        let mut place = |row: u8, col: u8, kind: PieceKind, color: Color| {
            let id = *id_offset;
            *id_offset += 1;
            board.squares[Square::new(row, col).0 as usize] = Some(Piece { kind, color, id });
        };

        // --- White pieces (rows 0-1 = ranks 1-2) ---
        let back_rank = [
            PieceKind::Rook,
            PieceKind::Knight,
            PieceKind::Bishop,
            PieceKind::Queen,
            PieceKind::King,
            PieceKind::Bishop,
            PieceKind::Knight,
            PieceKind::Rook,
        ];
        for col in 0..8u8 {
            place(0, col, back_rank[col as usize], Color::White);
        }
        for col in 0..8u8 {
            place(1, col, PieceKind::Pawn, Color::White);
        }

        // --- Black pieces (rows 6-7 = ranks 7-8) ---
        for col in 0..8u8 {
            place(6, col, PieceKind::Pawn, Color::Black);
        }
        for col in 0..8u8 {
            place(7, col, back_rank[col as usize], Color::Black);
        }

        board
    }

    /// Get the piece at a square.
    #[inline]
    pub fn piece_at(&self, sq: Square) -> Option<Piece> {
        self.squares[sq.0 as usize]
    }

    /// Set the piece at a square.
    #[inline]
    pub fn set_piece(&mut self, sq: Square, piece: Option<Piece>) {
        self.squares[sq.0 as usize] = piece;
    }

    /// Count all pieces on the board.
    pub fn piece_count(&self) -> usize {
        self.squares.iter().filter(|s| s.is_some()).count()
    }

    /// Check structural equality (same piece layout).
    pub fn is_identical_to(&self, other: &BoardState) -> bool {
        self.squares == other.squares
    }
}

impl PartialEq for BoardState {
    fn eq(&self, other: &Self) -> bool {
        self.is_identical_to(other)
    }
}

impl Eq for BoardState {}

// ============================================================
// MOVES
// ============================================================

/// A move in quantum chess.
#[derive(Clone, Debug)]
pub enum QuantumChessMove {
    /// Standard chess move, applied to every branch.
    Classical { from: Square, to: Square },
    /// Quantum split: piece at `from` enters superposition over `to_a` and `to_b`.
    QuantumSplit {
        from: Square,
        to_a: Square,
        to_b: Square,
    },
    /// Force measurement of a specific square, collapsing branches.
    Measure { square: Square },
}

impl fmt::Display for QuantumChessMove {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            QuantumChessMove::Classical { from, to } => {
                write!(f, "{} -> {}", from, to)
            }
            QuantumChessMove::QuantumSplit { from, to_a, to_b } => {
                write!(f, "{} ~> ({} | {})", from, to_a, to_b)
            }
            QuantumChessMove::Measure { square } => {
                write!(f, "Measure({})", square)
            }
        }
    }
}

// ============================================================
// GAME STATUS
// ============================================================

/// Current game status.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum GameStatus {
    InProgress,
    WhiteWins,
    BlackWins,
    Draw,
}

impl fmt::Display for GameStatus {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            GameStatus::InProgress => write!(f, "In progress"),
            GameStatus::WhiteWins => write!(f, "White wins"),
            GameStatus::BlackWins => write!(f, "Black wins"),
            GameStatus::Draw => write!(f, "Draw"),
        }
    }
}

// ============================================================
// MOVE RESULT
// ============================================================

/// Information returned after a successful move.
#[derive(Clone, Debug)]
pub struct MoveResult {
    /// Piece captured (if any) during measurement.
    pub captured: Option<Piece>,
    /// Whether a quantum measurement occurred.
    pub measurement_occurred: bool,
    /// Number of branches after the move.
    pub branches_after: usize,
    /// Human-readable description of what happened.
    pub description: String,
}

impl fmt::Display for MoveResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.description)
    }
}

// ============================================================
// ERROR TYPE
// ============================================================

/// Errors that can occur during a quantum chess game.
#[derive(Clone, Debug)]
pub enum ChessError {
    /// Square index >= 64 or otherwise invalid.
    InvalidSquare(Square),
    /// No piece found at the given square in any branch.
    NoPieceAt(Square),
    /// Attempted to move a piece of the wrong colour.
    NotYourTurn {
        expected: Color,
        found: Color,
    },
    /// The requested move violates piece movement rules.
    IllegalMove(String),
    /// The game has already ended.
    GameOver,
    /// Too many branches (exceeds safety limit).
    BranchLimitExceeded {
        current: usize,
        limit: usize,
    },
    /// Quantum split targets must differ from each other and from origin.
    InvalidSplit(String),
}

impl fmt::Display for ChessError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ChessError::InvalidSquare(sq) => write!(f, "invalid square: {}", sq.0),
            ChessError::NoPieceAt(sq) => write!(f, "no piece at {}", sq),
            ChessError::NotYourTurn { expected, found } => {
                write!(f, "not your turn: expected {}, found {}", expected, found)
            }
            ChessError::IllegalMove(msg) => write!(f, "illegal move: {}", msg),
            ChessError::GameOver => write!(f, "game is over"),
            ChessError::BranchLimitExceeded { current, limit } => {
                write!(f, "branch limit exceeded: {} >= {}", current, limit)
            }
            ChessError::InvalidSplit(msg) => write!(f, "invalid quantum split: {}", msg),
        }
    }
}

impl std::error::Error for ChessError {}

// ============================================================
// GAME CONFIGURATION (builder pattern)
// ============================================================

/// Configuration for a quantum chess game.
#[derive(Clone, Debug)]
pub struct QuantumChessConfig {
    /// Maximum number of branches before the game refuses new splits.
    pub max_branches: usize,
    /// PRNG seed for deterministic measurement.
    pub seed: u64,
    /// Whether to auto-merge identical branches after every move.
    pub auto_merge: bool,
    /// Whether to auto-normalize amplitudes after every move.
    pub auto_normalize: bool,
}

impl Default for QuantumChessConfig {
    fn default() -> Self {
        Self {
            max_branches: 1024,
            seed: 42,
            auto_merge: true,
            auto_normalize: true,
        }
    }
}

impl QuantumChessConfig {
    /// Create a new config builder.
    pub fn builder() -> QuantumChessConfigBuilder {
        QuantumChessConfigBuilder {
            config: Self::default(),
        }
    }
}

/// Builder for [`QuantumChessConfig`].
pub struct QuantumChessConfigBuilder {
    config: QuantumChessConfig,
}

impl QuantumChessConfigBuilder {
    pub fn max_branches(mut self, n: usize) -> Self {
        self.config.max_branches = n;
        self
    }

    pub fn seed(mut self, s: u64) -> Self {
        self.config.seed = s;
        self
    }

    pub fn auto_merge(mut self, b: bool) -> Self {
        self.config.auto_merge = b;
        self
    }

    pub fn auto_normalize(mut self, b: bool) -> Self {
        self.config.auto_normalize = b;
        self
    }

    pub fn build(self) -> QuantumChessConfig {
        self.config
    }
}

// ============================================================
// MOVE LEGALITY
// ============================================================

/// Check whether a piece of `kind` belonging to `color` can move from `from`
/// to `to` on the given `board`. This is simplified chess: no en passant,
/// no castling, no promotion. Path obstruction is checked for sliding pieces.
fn is_legal_piece_move(
    kind: PieceKind,
    color: Color,
    from: Square,
    to: Square,
    board: &BoardState,
) -> bool {
    if from == to {
        return false;
    }
    let dr = to.row() as i8 - from.row() as i8;
    let dc = to.col() as i8 - from.col() as i8;
    let abs_dr = dr.unsigned_abs();
    let abs_dc = dc.unsigned_abs();

    // Cannot capture own piece.
    if let Some(target) = board.piece_at(to) {
        if target.color == color {
            return false;
        }
    }

    match kind {
        PieceKind::Knight => abs_dr + abs_dc == 3 && abs_dr >= 1 && abs_dc >= 1,

        PieceKind::King => abs_dr <= 1 && abs_dc <= 1,

        PieceKind::Rook => {
            if dr != 0 && dc != 0 {
                return false;
            }
            path_clear(from, to, board)
        }

        PieceKind::Bishop => {
            if abs_dr != abs_dc {
                return false;
            }
            path_clear(from, to, board)
        }

        PieceKind::Queen => {
            if dr != 0 && dc != 0 && abs_dr != abs_dc {
                return false;
            }
            path_clear(from, to, board)
        }

        PieceKind::Pawn => {
            let forward: i8 = match color {
                Color::White => 1,
                Color::Black => -1,
            };
            let start_row: u8 = match color {
                Color::White => 1,
                Color::Black => 6,
            };

            let is_capture = board.piece_at(to).is_some();

            if is_capture {
                // Diagonal capture: one step forward, one step sideways.
                dr == forward && abs_dc == 1
            } else {
                // Forward move.
                if dc != 0 {
                    return false;
                }
                if dr == forward {
                    true
                } else if dr == 2 * forward && from.row() == start_row {
                    // Two-square advance from starting row, path must be clear.
                    let mid = Square::new((from.row() as i8 + forward) as u8, from.col());
                    board.piece_at(mid).is_none()
                } else {
                    false
                }
            }
        }
    }
}

/// Check that the path between `from` and `to` is clear (exclusive of
/// endpoints). Used for rook, bishop, and queen.
fn path_clear(from: Square, to: Square, board: &BoardState) -> bool {
    let dr = (to.row() as i8 - from.row() as i8).signum();
    let dc = (to.col() as i8 - from.col() as i8).signum();

    let mut r = from.row() as i8 + dr;
    let mut c = from.col() as i8 + dc;

    while (r, c) != (to.row() as i8, to.col() as i8) {
        if r < 0 || r > 7 || c < 0 || c > 7 {
            return false;
        }
        if board.piece_at(Square::new(r as u8, c as u8)).is_some() {
            return false;
        }
        r += dr;
        c += dc;
    }
    true
}

// ============================================================
// QUANTUM CHESS GAME
// ============================================================

/// The main quantum chess game engine.
///
/// The game state is a superposition of classical board states (`branches`),
/// each weighted by a complex amplitude. Quantum splits double the number
/// of branches (up to `config.max_branches`), while measurements and
/// merging reduce them.
#[derive(Clone, Debug)]
pub struct QuantumChessGame {
    /// Superposition of board states with complex amplitudes.
    branches: Vec<(BoardState, C64)>,
    /// History of all moves played.
    move_history: Vec<QuantumChessMove>,
    /// Whose turn it is.
    turn: Color,
    /// Current game status.
    status: GameStatus,
    /// Next unique piece ID (used when pieces must be distinguished).
    next_piece_id: u8,
    /// PRNG state for deterministic measurement.
    rng_state: u64,
    /// Game configuration.
    config: QuantumChessConfig,
}

impl QuantumChessGame {
    // --------------------------------------------------------
    // Constructors
    // --------------------------------------------------------

    /// Create a new game with the standard starting position and default config.
    pub fn new() -> Self {
        Self::with_config(QuantumChessConfig::default())
    }

    /// Create a new game with a specific PRNG seed (for reproducibility).
    pub fn from_seed(seed: u64) -> Self {
        let config = QuantumChessConfig {
            seed,
            ..Default::default()
        };
        Self::with_config(config)
    }

    /// Create a new game with full configuration.
    pub fn with_config(config: QuantumChessConfig) -> Self {
        let mut next_id: u8 = 0;
        let board = BoardState::standard_setup(&mut next_id);
        let amplitude = C64::new(1.0, 0.0);

        QuantumChessGame {
            branches: vec![(board, amplitude)],
            move_history: Vec::new(),
            turn: Color::White,
            status: GameStatus::InProgress,
            next_piece_id: next_id,
            rng_state: config.seed,
            config,
        }
    }

    /// Create a game from a custom board state (for testing / puzzles).
    pub fn from_board(board: BoardState, turn: Color, seed: u64) -> Self {
        let next_id = board
            .squares
            .iter()
            .filter_map(|p| p.as_ref())
            .map(|p| p.id)
            .max()
            .unwrap_or(0)
            + 1;

        QuantumChessGame {
            branches: vec![(board, C64::new(1.0, 0.0))],
            move_history: Vec::new(),
            turn,
            status: GameStatus::InProgress,
            next_piece_id: next_id,
            rng_state: seed,
            config: QuantumChessConfig {
                seed,
                ..Default::default()
            },
        }
    }

    // --------------------------------------------------------
    // Accessors
    // --------------------------------------------------------

    /// Whose turn is it?
    #[inline]
    pub fn current_turn(&self) -> Color {
        self.turn
    }

    /// Current game status.
    #[inline]
    pub fn status(&self) -> &GameStatus {
        &self.status
    }

    /// Number of active branches in the superposition.
    #[inline]
    pub fn num_branches(&self) -> usize {
        self.branches.len()
    }

    /// Number of moves played so far.
    #[inline]
    pub fn move_count(&self) -> usize {
        self.move_history.len()
    }

    /// Read-only access to the branch superposition.
    pub fn branches(&self) -> &[(BoardState, C64)] {
        &self.branches
    }

    /// Read-only access to move history.
    pub fn history(&self) -> &[QuantumChessMove] {
        &self.move_history
    }

    // --------------------------------------------------------
    // Probability queries
    // --------------------------------------------------------

    /// For a given square, return each piece that might be there and its
    /// probability (sum of |amplitude|^2 over matching branches).
    pub fn piece_at_probability(&self, sq: Square) -> Vec<(Piece, f64)> {
        let mut map: HashMap<(PieceKind, Color, u8), f64> = HashMap::new();
        for (board, amp) in &self.branches {
            if let Some(piece) = board.piece_at(sq) {
                let key = (piece.kind, piece.color, piece.id);
                *map.entry(key).or_insert(0.0) += amp.norm_sqr();
            }
        }
        let mut result: Vec<(Piece, f64)> = map
            .into_iter()
            .map(|((kind, color, id), prob)| (Piece { kind, color, id }, prob))
            .collect();
        result.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        result
    }

    /// Total probability that a given square is occupied by *any* piece.
    pub fn square_occupied_probability(&self, sq: Square) -> f64 {
        self.branches
            .iter()
            .filter(|(board, _)| board.piece_at(sq).is_some())
            .map(|(_, amp)| amp.norm_sqr())
            .sum()
    }

    /// Shannon entropy of the branch distribution, measuring how "quantum"
    /// the current game state is. A purely classical state has entropy 0.
    pub fn board_entropy(&self) -> f64 {
        let mut entropy = 0.0f64;
        for (_, amp) in &self.branches {
            let p = amp.norm_sqr();
            if p > 1e-15 {
                entropy -= p * p.ln();
            }
        }
        entropy
    }

    /// Probability that the king of the given colour is still alive (exists
    /// on at least one square) across all branches.
    pub fn is_king_alive(&self, color: Color) -> f64 {
        let mut prob = 0.0f64;
        for (board, amp) in &self.branches {
            let has_king = board
                .squares
                .iter()
                .any(|sq| matches!(sq, Some(p) if p.kind == PieceKind::King && p.color == color));
            if has_king {
                prob += amp.norm_sqr();
            }
        }
        prob
    }

    // --------------------------------------------------------
    // Making moves
    // --------------------------------------------------------

    /// Execute a move. Returns a [`MoveResult`] on success.
    pub fn make_move(&mut self, mv: QuantumChessMove) -> Result<MoveResult, ChessError> {
        if self.status != GameStatus::InProgress {
            return Err(ChessError::GameOver);
        }

        let result = match &mv {
            QuantumChessMove::Classical { from, to } => {
                self.apply_classical(*from, *to)?
            }
            QuantumChessMove::QuantumSplit { from, to_a, to_b } => {
                self.apply_quantum_split(*from, *to_a, *to_b)?
            }
            QuantumChessMove::Measure { square } => {
                self.apply_measure(*square)?
            }
        };

        // Post-move bookkeeping.
        if self.config.auto_merge {
            self.merge_identical_branches();
        }
        if self.config.auto_normalize {
            self.normalize();
        }
        self.update_game_status();

        self.move_history.push(mv);

        // Alternate turn (Measure does not consume a turn).
        if !matches!(self.move_history.last(), Some(QuantumChessMove::Measure { .. })) {
            self.turn = self.turn.opposite();
        }

        Ok(MoveResult {
            captured: result.0,
            measurement_occurred: result.1,
            branches_after: self.branches.len(),
            description: result.2,
        })
    }

    // --------------------------------------------------------
    // Classical move
    // --------------------------------------------------------

    /// Apply a classical move to every branch.
    fn apply_classical(
        &mut self,
        from: Square,
        to: Square,
    ) -> Result<(Option<Piece>, bool, String), ChessError> {
        self.validate_square(from)?;
        self.validate_square(to)?;

        // Check that the moving piece exists in at least one branch and
        // belongs to the current player.
        let piece = self.find_piece_for_move(from)?;

        // Validate legality against at least one branch.
        let mut any_legal = false;
        for (board, _) in &self.branches {
            if let Some(p) = board.piece_at(from) {
                if p.color == self.turn
                    && is_legal_piece_move(p.kind, p.color, from, to, board)
                {
                    any_legal = true;
                    break;
                }
            }
        }
        if !any_legal {
            return Err(ChessError::IllegalMove(format!(
                "{} cannot move from {} to {}",
                piece.kind, from, to
            )));
        }

        // Check if any branch has the target square occupied by the opponent
        // while another branch has it empty --- this triggers measurement.
        let needs_measurement = self.square_needs_measurement(to);
        let mut captured: Option<Piece> = None;

        if needs_measurement {
            captured = self.measure_square(to);
        }

        // Now apply the move to every surviving branch.
        for (board, _) in self.branches.iter_mut() {
            if let Some(p) = board.piece_at(from) {
                if p.color == self.turn
                    && is_legal_piece_move(p.kind, p.color, from, to, board)
                {
                    // Capture whatever is on the target square.
                    if captured.is_none() {
                        if let Some(cap) = board.piece_at(to) {
                            if cap.color != self.turn {
                                captured = Some(cap);
                            }
                        }
                    }
                    board.set_piece(to, Some(p));
                    board.set_piece(from, None);
                }
            }
        }

        let desc = if let Some(cap) = captured {
            format!(
                "{} {} -> {} captures {}",
                piece.kind, from, to, cap.kind
            )
        } else {
            format!("{} {} -> {}", piece.kind, from, to)
        };

        Ok((captured, needs_measurement, desc))
    }

    // --------------------------------------------------------
    // Quantum split
    // --------------------------------------------------------

    /// Split a piece into superposition over two target squares.
    fn apply_quantum_split(
        &mut self,
        from: Square,
        to_a: Square,
        to_b: Square,
    ) -> Result<(Option<Piece>, bool, String), ChessError> {
        self.validate_square(from)?;
        self.validate_square(to_a)?;
        self.validate_square(to_b)?;

        if to_a == to_b {
            return Err(ChessError::InvalidSplit(
                "split targets must be different squares".to_string(),
            ));
        }
        if from == to_a || from == to_b {
            return Err(ChessError::InvalidSplit(
                "split target cannot be the origin square".to_string(),
            ));
        }

        // Check branch limit.
        if self.branches.len() * 2 > self.config.max_branches {
            return Err(ChessError::BranchLimitExceeded {
                current: self.branches.len() * 2,
                limit: self.config.max_branches,
            });
        }

        let piece = self.find_piece_for_move(from)?;

        // Validate that both targets are legal moves for this piece.
        let mut legal_a = false;
        let mut legal_b = false;
        for (board, _) in &self.branches {
            if let Some(p) = board.piece_at(from) {
                if p.color == self.turn {
                    if is_legal_piece_move(p.kind, p.color, from, to_a, board) {
                        legal_a = true;
                    }
                    if is_legal_piece_move(p.kind, p.color, from, to_b, board) {
                        legal_b = true;
                    }
                }
            }
        }
        if !legal_a {
            return Err(ChessError::IllegalMove(format!(
                "{} cannot move from {} to {} (split target A)",
                piece.kind, from, to_a
            )));
        }
        if !legal_b {
            return Err(ChessError::IllegalMove(format!(
                "{} cannot move from {} to {} (split target B)",
                piece.kind, from, to_b
            )));
        }

        // Measurement may be needed if either target is occupied in some branches.
        let mut measurement_occurred = false;
        let mut captured: Option<Piece> = None;
        if self.square_needs_measurement(to_a) {
            captured = self.measure_square(to_a);
            measurement_occurred = true;
        }
        if self.square_needs_measurement(to_b) {
            let cap_b = self.measure_square(to_b);
            if captured.is_none() {
                captured = cap_b;
            }
            measurement_occurred = true;
        }

        // Split: duplicate every branch. In the first copy the piece goes to
        // to_a, in the second copy it goes to to_b. Amplitude is scaled by
        // 1/sqrt(2).
        let inv_sqrt2 = 1.0 / 2.0f64.sqrt();
        let original_branches: Vec<(BoardState, C64)> = self.branches.drain(..).collect();
        self.branches.reserve(original_branches.len() * 2);

        for (board, amp) in &original_branches {
            if board.piece_at(from).is_some() {
                // Branch A: piece goes to to_a.
                let mut board_a = board.clone();
                let p = board_a.piece_at(from).unwrap();
                board_a.set_piece(to_a, Some(p));
                board_a.set_piece(from, None);
                self.branches
                    .push((board_a, amp * C64::new(inv_sqrt2, 0.0)));

                // Branch B: piece goes to to_b.
                let mut board_b = board.clone();
                let p = board_b.piece_at(from).unwrap();
                board_b.set_piece(to_b, Some(p));
                board_b.set_piece(from, None);
                self.branches
                    .push((board_b, amp * C64::new(inv_sqrt2, 0.0)));
            } else {
                // Piece not at `from` in this branch (already moved elsewhere
                // in a prior split). Keep the branch unchanged.
                self.branches.push((board.clone(), *amp));
            }
        }

        let desc = format!(
            "{} {} ~> ({} | {})  [{} branches]",
            piece.kind,
            from,
            to_a,
            to_b,
            self.branches.len()
        );

        Ok((captured, measurement_occurred, desc))
    }

    // --------------------------------------------------------
    // Measurement
    // --------------------------------------------------------

    /// Force measurement of a specific square.
    fn apply_measure(
        &mut self,
        square: Square,
    ) -> Result<(Option<Piece>, bool, String), ChessError> {
        self.validate_square(square)?;

        let occupied_prob = self.square_occupied_probability(square);

        if (occupied_prob - 0.0).abs() < 1e-12 || (occupied_prob - 1.0).abs() < 1e-12 {
            // No superposition on this square --- nothing to measure.
            return Ok((
                None,
                false,
                format!("Measure {}: no superposition (prob = {:.2})", square, occupied_prob),
            ));
        }

        let captured = self.measure_square(square);

        let desc = format!(
            "Measure {}: collapsed (occupied prob was {:.1}%)",
            square,
            occupied_prob * 100.0
        );
        Ok((captured, true, desc))
    }

    // --------------------------------------------------------
    // Internal helpers
    // --------------------------------------------------------

    /// Validate that a square index is in range.
    fn validate_square(&self, sq: Square) -> Result<(), ChessError> {
        if !sq.is_valid() {
            return Err(ChessError::InvalidSquare(sq));
        }
        Ok(())
    }

    /// Find the piece at `from` that belongs to the current player in at
    /// least one branch. Returns an error if no such piece exists.
    fn find_piece_for_move(&self, from: Square) -> Result<Piece, ChessError> {
        for (board, _) in &self.branches {
            if let Some(p) = board.piece_at(from) {
                if p.color == self.turn {
                    return Ok(p);
                } else {
                    return Err(ChessError::NotYourTurn {
                        expected: self.turn,
                        found: p.color,
                    });
                }
            }
        }
        Err(ChessError::NoPieceAt(from))
    }

    /// Check whether a square is occupied in some branches but not others
    /// (or occupied by different pieces), requiring measurement.
    fn square_needs_measurement(&self, sq: Square) -> bool {
        if self.branches.len() <= 1 {
            return false;
        }
        let first_occupied = self.branches[0].0.piece_at(sq);
        for (board, _) in &self.branches[1..] {
            let this_occupied = board.piece_at(sq);
            // Compare by piece identity, not just occupancy.
            if first_occupied != this_occupied {
                return true;
            }
        }
        false
    }

    /// Measure whether a square is occupied. Collapses branches accordingly
    /// using the seeded PRNG. Returns any captured piece.
    fn measure_square(&mut self, sq: Square) -> Option<Piece> {
        let occupied_prob: f64 = self
            .branches
            .iter()
            .filter(|(b, _)| b.piece_at(sq).is_some())
            .map(|(_, a)| a.norm_sqr())
            .sum();

        let roll = lcg_f64(&mut self.rng_state);
        let piece_exists = roll < occupied_prob;

        let mut captured_piece: Option<Piece> = None;

        if piece_exists {
            // Keep only branches where the square is occupied.
            self.branches.retain(|(b, _)| b.piece_at(sq).is_some());
        } else {
            // Keep only branches where the square is empty.
            // The piece that was "there" in discarded branches is effectively captured
            // (removed from the game).
            for (board, _) in &self.branches {
                if let Some(p) = board.piece_at(sq) {
                    captured_piece = Some(p);
                    break;
                }
            }
            self.branches.retain(|(b, _)| b.piece_at(sq).is_none());
        }

        // Re-normalize after measurement.
        self.normalize();
        captured_piece
    }

    /// Merge branches with identical board states by summing amplitudes.
    fn merge_identical_branches(&mut self) {
        if self.branches.len() <= 1 {
            return;
        }

        let mut merged: Vec<(BoardState, C64)> = Vec::new();

        'outer: for (board, amp) in self.branches.drain(..) {
            for (existing_board, existing_amp) in merged.iter_mut() {
                if existing_board.is_identical_to(&board) {
                    *existing_amp += amp;
                    continue 'outer;
                }
            }
            merged.push((board, amp));
        }

        // Remove branches with negligible amplitude.
        merged.retain(|(_, amp)| amp.norm_sqr() > 1e-30);

        self.branches = merged;
    }

    /// Normalize amplitudes so that sum of |a|^2 = 1.
    fn normalize(&mut self) {
        let total: f64 = self.branches.iter().map(|(_, a)| a.norm_sqr()).sum();
        if total < 1e-30 {
            return;
        }
        let scale = 1.0 / total.sqrt();
        for (_, amp) in self.branches.iter_mut() {
            *amp *= C64::new(scale, 0.0);
        }
    }

    /// Update the game status based on king survival.
    fn update_game_status(&mut self) {
        let white_king_prob = self.is_king_alive(Color::White);
        let black_king_prob = self.is_king_alive(Color::Black);

        if white_king_prob < 1e-12 && black_king_prob < 1e-12 {
            self.status = GameStatus::Draw;
        } else if white_king_prob < 1e-12 {
            self.status = GameStatus::BlackWins;
        } else if black_king_prob < 1e-12 {
            self.status = GameStatus::WhiteWins;
        }
    }

    // --------------------------------------------------------
    // Collapse / display
    // --------------------------------------------------------

    /// Collapse the entire game state into a single classical board.
    /// Uses the seeded PRNG to pick a branch weighted by |amplitude|^2.
    pub fn collapse_all(&mut self) -> BoardState {
        if self.branches.len() == 1 {
            return self.branches[0].0.clone();
        }

        let roll = lcg_f64(&mut self.rng_state);
        let mut cumulative = 0.0;
        let mut chosen_idx = 0;

        for (i, (_, amp)) in self.branches.iter().enumerate() {
            cumulative += amp.norm_sqr();
            if roll < cumulative {
                chosen_idx = i;
                break;
            }
        }

        let chosen = self.branches[chosen_idx].0.clone();
        self.branches = vec![(chosen.clone(), C64::new(1.0, 0.0))];
        chosen
    }

    /// Generate an ASCII representation of the board showing probabilities
    /// for pieces in superposition.
    pub fn display_board(&self) -> String {
        let mut lines: Vec<String> = Vec::new();
        lines.push("  a  b  c  d  e  f  g  h".to_string());

        // Build probability annotations for squares where a piece is in
        // superposition (probability strictly between 0 and 1).
        let mut annotations: HashMap<(u8, u8), (char, f64)> = HashMap::new();

        for row in 0..8u8 {
            for col in 0..8u8 {
                let sq = Square::new(row, col);
                let probs = self.piece_at_probability(sq);
                if let Some((piece, prob)) = probs.first() {
                    if *prob > 1e-6 {
                        annotations.insert(
                            (row, col),
                            (piece.display_char(), *prob),
                        );
                    }
                }
            }
        }

        for row in (0..8u8).rev() {
            let rank = row + 1;
            let mut rank_line = format!("{}", rank);

            for col in 0..8u8 {
                if let Some((ch, prob)) = annotations.get(&(row, col)) {
                    if (*prob - 1.0).abs() < 1e-6 {
                        // Definitely present.
                        rank_line.push_str(&format!(" [{}]", ch));
                    } else {
                        // In superposition.
                        rank_line.push_str(&format!(" [{}]", ch));
                    }
                } else {
                    rank_line.push_str("  . ");
                }
            }
            lines.push(rank_line);

            // Add probability annotation line if any piece on this rank is
            // in superposition.
            let mut has_annotation = false;
            let mut ann_line = " ".to_string();
            for col in 0..8u8 {
                if let Some((_, prob)) = annotations.get(&(row, col)) {
                    if (*prob - 1.0).abs() > 1e-6 && *prob > 1e-6 {
                        ann_line.push_str(&format!("{:>3}%", (prob * 100.0).round() as u32));
                        has_annotation = true;
                    } else {
                        ann_line.push_str("    ");
                    }
                } else {
                    ann_line.push_str("    ");
                }
            }
            if has_annotation {
                lines.push(ann_line);
            }
        }

        lines.join("\n")
    }

    /// One-line summary of the game state.
    pub fn summary(&self) -> String {
        format!(
            "Turn: {} | Branches: {} | Entropy: {:.3} | Status: {}",
            self.turn,
            self.branches.len(),
            self.board_entropy(),
            self.status,
        )
    }
}

impl Default for QuantumChessGame {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Display for QuantumChessGame {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "{}", self.display_board())?;
        writeln!(f)?;
        write!(f, "{}", self.summary())
    }
}

// ============================================================
// SCENARIO HELPERS
// ============================================================

/// Pre-built scenarios for demonstration and testing.
pub struct QuantumChessScenarios;

impl QuantumChessScenarios {
    /// The "quantum knight fork" scenario.
    ///
    /// White has a knight on d4 that splits to c6 and e6, simultaneously
    /// threatening Black's queen on b8 and rook on f8.
    pub fn quantum_knight_fork() -> QuantumChessGame {
        let mut board = BoardState::empty();
        let mut id: u8 = 0;

        let mut place = |row: u8, col: u8, kind: PieceKind, color: Color| {
            let piece = Piece { kind, color, id };
            id += 1;
            board.set_piece(Square::new(row, col), Some(piece));
        };

        // Kings (required)
        place(0, 4, PieceKind::King, Color::White);
        place(7, 4, PieceKind::King, Color::Black);
        // White knight on d4
        place(3, 3, PieceKind::Knight, Color::White);
        // Black queen on b8 and rook on f8
        place(7, 1, PieceKind::Queen, Color::Black);
        place(7, 5, PieceKind::Rook, Color::Black);

        QuantumChessGame::from_board(board, Color::White, 42)
    }

    /// A simplified endgame: king and queen vs king.
    pub fn endgame_kq_vs_k() -> QuantumChessGame {
        let mut board = BoardState::empty();
        let mut id: u8 = 0;

        let mut place = |row: u8, col: u8, kind: PieceKind, color: Color| {
            let piece = Piece { kind, color, id };
            id += 1;
            board.set_piece(Square::new(row, col), Some(piece));
        };

        place(0, 4, PieceKind::King, Color::White);
        place(3, 3, PieceKind::Queen, Color::White);
        place(7, 4, PieceKind::King, Color::Black);

        QuantumChessGame::from_board(board, Color::White, 123)
    }
}

// ============================================================
// TESTS
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;

    // --- Test 1: Initial board setup ---
    #[test]
    fn test_initial_board_has_32_pieces() {
        let game = QuantumChessGame::new();
        assert_eq!(game.num_branches(), 1);
        let board = &game.branches()[0].0;
        assert_eq!(board.piece_count(), 32);
    }

    // --- Test 2: Classical e2-e4 ---
    #[test]
    fn test_classical_e2_e4() {
        let mut game = QuantumChessGame::new();
        let result = game
            .make_move(QuantumChessMove::Classical {
                from: Square::from_algebraic("e2"),
                to: Square::from_algebraic("e4"),
            })
            .unwrap();

        assert!(!result.measurement_occurred);
        assert!(result.captured.is_none());
        assert_eq!(result.branches_after, 1);

        let board = &game.branches()[0].0;
        assert!(board.piece_at(Square::from_algebraic("e2")).is_none());
        let pawn = board.piece_at(Square::from_algebraic("e4")).unwrap();
        assert_eq!(pawn.kind, PieceKind::Pawn);
        assert_eq!(pawn.color, Color::White);
    }

    // --- Test 3: Classical Nf3 ---
    #[test]
    fn test_classical_knight_f3() {
        let mut game = QuantumChessGame::new();
        let result = game
            .make_move(QuantumChessMove::Classical {
                from: Square::from_algebraic("g1"),
                to: Square::from_algebraic("f3"),
            })
            .unwrap();

        assert!(!result.measurement_occurred);
        assert!(result.captured.is_none());

        let board = &game.branches()[0].0;
        assert!(board.piece_at(Square::from_algebraic("g1")).is_none());
        let knight = board.piece_at(Square::from_algebraic("f3")).unwrap();
        assert_eq!(knight.kind, PieceKind::Knight);
    }

    // --- Test 4: Quantum split creates 2 branches ---
    #[test]
    fn test_quantum_split_doubles_branches() {
        let mut game = QuantumChessGame::new();
        let result = game
            .make_move(QuantumChessMove::QuantumSplit {
                from: Square::from_algebraic("g1"),
                to_a: Square::from_algebraic("f3"),
                to_b: Square::from_algebraic("h3"),
            })
            .unwrap();

        assert_eq!(result.branches_after, 2);
        assert_eq!(game.num_branches(), 2);
    }

    // --- Test 5: Amplitudes normalised after split ---
    #[test]
    fn test_amplitudes_normalized_after_split() {
        let mut game = QuantumChessGame::new();
        game.make_move(QuantumChessMove::QuantumSplit {
            from: Square::from_algebraic("g1"),
            to_a: Square::from_algebraic("f3"),
            to_b: Square::from_algebraic("h3"),
        })
        .unwrap();

        let total: f64 = game.branches().iter().map(|(_, a)| a.norm_sqr()).sum();
        assert!(
            (total - 1.0).abs() < 1e-10,
            "amplitudes not normalised: sum = {}",
            total
        );
    }

    // --- Test 6: Piece probability sums correctly ---
    #[test]
    fn test_piece_probability_sum() {
        let mut game = QuantumChessGame::new();
        game.make_move(QuantumChessMove::QuantumSplit {
            from: Square::from_algebraic("g1"),
            to_a: Square::from_algebraic("f3"),
            to_b: Square::from_algebraic("h3"),
        })
        .unwrap();

        let prob_f3 = game.square_occupied_probability(Square::from_algebraic("f3"));
        let prob_h3 = game.square_occupied_probability(Square::from_algebraic("h3"));

        assert!(
            (prob_f3 - 0.5).abs() < 1e-10,
            "f3 probability: {}",
            prob_f3
        );
        assert!(
            (prob_h3 - 0.5).abs() < 1e-10,
            "h3 probability: {}",
            prob_h3
        );
        // The knight left g1 in both branches, so g1 is empty.
        let prob_g1 = game.square_occupied_probability(Square::from_algebraic("g1"));
        assert!(
            prob_g1 < 1e-10,
            "g1 should be empty, prob = {}",
            prob_g1
        );
    }

    // --- Test 7: Square occupied probability for non-split piece ---
    #[test]
    fn test_square_occupied_probability_classical() {
        let game = QuantumChessGame::new();
        // e1 has the white king in the only branch.
        let prob = game.square_occupied_probability(Square::from_algebraic("e1"));
        assert!((prob - 1.0).abs() < 1e-10);

        // e4 is empty.
        let prob_empty = game.square_occupied_probability(Square::from_algebraic("e4"));
        assert!(prob_empty < 1e-10);
    }

    // --- Test 8: Board entropy = 0 for classical state ---
    #[test]
    fn test_board_entropy_classical_is_zero() {
        let game = QuantumChessGame::new();
        let entropy = game.board_entropy();
        assert!(
            entropy.abs() < 1e-10,
            "classical state entropy should be 0, got {}",
            entropy
        );
    }

    // --- Test 9: Board entropy > 0 after quantum move ---
    #[test]
    fn test_board_entropy_positive_after_split() {
        let mut game = QuantumChessGame::new();
        game.make_move(QuantumChessMove::QuantumSplit {
            from: Square::from_algebraic("g1"),
            to_a: Square::from_algebraic("f3"),
            to_b: Square::from_algebraic("h3"),
        })
        .unwrap();

        let entropy = game.board_entropy();
        assert!(
            entropy > 0.0,
            "entropy should be positive after split, got {}",
            entropy
        );
        // For 2 equal branches: -2 * 0.5 * ln(0.5) = ln(2) ~ 0.693
        assert!(
            (entropy - 2.0f64.ln()).abs() < 1e-10,
            "entropy should be ln(2) for equal split, got {}",
            entropy
        );
    }

    // --- Test 10: Capture on superposed square triggers measurement ---
    #[test]
    fn test_capture_triggers_measurement() {
        // Set up a board where a white knight is in superposition on f3/h3
        // and a black pawn advances to f3 to capture.
        let mut game = QuantumChessGame::new();

        // White splits knight to f3 | h3.
        game.make_move(QuantumChessMove::QuantumSplit {
            from: Square::from_algebraic("g1"),
            to_a: Square::from_algebraic("f3"),
            to_b: Square::from_algebraic("h3"),
        })
        .unwrap();
        assert_eq!(game.num_branches(), 2);

        // Black moves pawn from e7 to e6 (classical, no collision).
        game.make_move(QuantumChessMove::Classical {
            from: Square::from_algebraic("e7"),
            to: Square::from_algebraic("e6"),
        })
        .unwrap();

        // White moves pawn d2 to d3 (classical).
        game.make_move(QuantumChessMove::Classical {
            from: Square::from_algebraic("d2"),
            to: Square::from_algebraic("d3"),
        })
        .unwrap();

        // Black pushes pawn from e6 to e5.
        game.make_move(QuantumChessMove::Classical {
            from: Square::from_algebraic("e6"),
            to: Square::from_algebraic("e5"),
        })
        .unwrap();

        // White plays another filler.
        game.make_move(QuantumChessMove::Classical {
            from: Square::from_algebraic("a2"),
            to: Square::from_algebraic("a3"),
        })
        .unwrap();

        // Black pawn e5 -> e4.
        game.make_move(QuantumChessMove::Classical {
            from: Square::from_algebraic("e5"),
            to: Square::from_algebraic("e4"),
        })
        .unwrap();

        // White filler.
        game.make_move(QuantumChessMove::Classical {
            from: Square::from_algebraic("a3"),
            to: Square::from_algebraic("a4"),
        })
        .unwrap();

        // Black pawn e4 -> f3 (diagonal capture). f3 is occupied only in
        // 50% of branches, so measurement occurs.
        let before = game.num_branches();
        assert_eq!(before, 2);

        let result = game
            .make_move(QuantumChessMove::Classical {
                from: Square::from_algebraic("e4"),
                to: Square::from_algebraic("f3"),
            })
            .unwrap();

        // After measurement, branches collapse to 1.
        assert_eq!(result.branches_after, 1);
        assert!(result.measurement_occurred);
    }

    // --- Test 11: Branch merging for identical states ---
    #[test]
    fn test_branch_merging() {
        // Create a game and manually add an identical branch to test merging.
        let mut game = QuantumChessGame::new();
        let existing_board = game.branches()[0].0.clone();
        let existing_amp = game.branches()[0].1;

        // Duplicate the single branch.
        game.branches
            .push((existing_board, existing_amp * C64::new(0.5, 0.0)));

        // Merge should combine them.
        game.merge_identical_branches();
        assert_eq!(game.num_branches(), 1);
    }

    // --- Test 12: Collapse all produces valid classical board ---
    #[test]
    fn test_collapse_all_produces_valid_board() {
        let mut game = QuantumChessGame::new();
        game.make_move(QuantumChessMove::QuantumSplit {
            from: Square::from_algebraic("g1"),
            to_a: Square::from_algebraic("f3"),
            to_b: Square::from_algebraic("h3"),
        })
        .unwrap();

        assert_eq!(game.num_branches(), 2);
        let collapsed = game.collapse_all();
        assert_eq!(game.num_branches(), 1);

        // The knight should be at f3 or h3 but not both.
        let at_f3 = collapsed.piece_at(Square::from_algebraic("f3"));
        let at_h3 = collapsed.piece_at(Square::from_algebraic("h3"));
        assert!(
            at_f3.is_some() != at_h3.is_some(),
            "knight must be at exactly one of f3/h3"
        );
        // 32 pieces total (no capture occurred).
        assert_eq!(collapsed.piece_count(), 32);
    }

    // --- Test 13: Turn alternation ---
    #[test]
    fn test_turn_alternation() {
        let mut game = QuantumChessGame::new();
        assert_eq!(game.current_turn(), Color::White);

        game.make_move(QuantumChessMove::Classical {
            from: Square::from_algebraic("e2"),
            to: Square::from_algebraic("e4"),
        })
        .unwrap();
        assert_eq!(game.current_turn(), Color::Black);

        game.make_move(QuantumChessMove::Classical {
            from: Square::from_algebraic("e7"),
            to: Square::from_algebraic("e5"),
        })
        .unwrap();
        assert_eq!(game.current_turn(), Color::White);
    }

    // --- Test 14: Illegal move rejected (wrong colour) ---
    #[test]
    fn test_reject_wrong_colour() {
        let game_result = {
            let mut game = QuantumChessGame::new();
            // Try moving a black piece on white's turn.
            game.make_move(QuantumChessMove::Classical {
                from: Square::from_algebraic("e7"),
                to: Square::from_algebraic("e5"),
            })
        };

        assert!(game_result.is_err());
        match game_result {
            Err(ChessError::NotYourTurn { expected, found }) => {
                assert_eq!(expected, Color::White);
                assert_eq!(found, Color::Black);
            }
            other => panic!("expected NotYourTurn, got {:?}", other),
        }
    }

    // --- Test 15: Illegal knight move rejected ---
    #[test]
    fn test_reject_illegal_knight_move() {
        let mut game = QuantumChessGame::new();
        let result = game.make_move(QuantumChessMove::Classical {
            from: Square::from_algebraic("g1"),
            to: Square::from_algebraic("g3"), // invalid for knight
        });
        assert!(result.is_err());
        assert!(matches!(result, Err(ChessError::IllegalMove(_))));
    }

    // --- Test 16: Display board produces output ---
    #[test]
    fn test_display_board() {
        let game = QuantumChessGame::new();
        let display = game.display_board();

        // Should contain file labels.
        assert!(display.contains("a  b  c  d  e  f  g  h"));
        // Should contain pieces.
        assert!(display.contains("[R]")); // white rook
        assert!(display.contains("[r]")); // black rook
        assert!(display.contains("[K]")); // white king
        assert!(display.contains("[k]")); // black king
        // Should have rank numbers.
        assert!(display.contains("1"));
        assert!(display.contains("8"));
    }

    // --- Test 17: Quantum knight fork scenario ---
    #[test]
    fn test_quantum_knight_fork() {
        let mut game = QuantumChessScenarios::quantum_knight_fork();

        // Split the knight d4 -> c6 | e6.
        let result = game
            .make_move(QuantumChessMove::QuantumSplit {
                from: Square::from_algebraic("d4"),
                to_a: Square::from_algebraic("c6"),
                to_b: Square::from_algebraic("e6"),
            })
            .unwrap();

        assert_eq!(result.branches_after, 2);

        // In branch A, knight on c6 threatens queen on b8.
        // In branch B, knight on e6 threatens rook (and is closer to king).
        let prob_c6 = game.square_occupied_probability(Square::from_algebraic("c6"));
        let prob_e6 = game.square_occupied_probability(Square::from_algebraic("e6"));
        assert!((prob_c6 - 0.5).abs() < 1e-10);
        assert!((prob_e6 - 0.5).abs() < 1e-10);
    }

    // --- Test 18: Multiple quantum moves grow branches ---
    #[test]
    fn test_multiple_splits_grow_branches() {
        let mut game = QuantumChessGame::new();

        // White splits knight.
        game.make_move(QuantumChessMove::QuantumSplit {
            from: Square::from_algebraic("g1"),
            to_a: Square::from_algebraic("f3"),
            to_b: Square::from_algebraic("h3"),
        })
        .unwrap();
        assert_eq!(game.num_branches(), 2);

        // Black classical move.
        game.make_move(QuantumChessMove::Classical {
            from: Square::from_algebraic("e7"),
            to: Square::from_algebraic("e5"),
        })
        .unwrap();
        assert_eq!(game.num_branches(), 2);

        // White splits the other knight.
        game.make_move(QuantumChessMove::QuantumSplit {
            from: Square::from_algebraic("b1"),
            to_a: Square::from_algebraic("a3"),
            to_b: Square::from_algebraic("c3"),
        })
        .unwrap();
        assert_eq!(game.num_branches(), 4);
    }

    // --- Test 19: King alive probability = 1.0 at start ---
    #[test]
    fn test_king_alive_at_start() {
        let game = QuantumChessGame::new();
        let white = game.is_king_alive(Color::White);
        let black = game.is_king_alive(Color::Black);
        assert!((white - 1.0).abs() < 1e-10);
        assert!((black - 1.0).abs() < 1e-10);
    }

    // --- Test 20: Seed determinism ---
    #[test]
    fn test_seed_determinism() {
        let run = |seed: u64| -> BoardState {
            let mut game = QuantumChessGame::from_seed(seed);
            game.make_move(QuantumChessMove::QuantumSplit {
                from: Square::from_algebraic("g1"),
                to_a: Square::from_algebraic("f3"),
                to_b: Square::from_algebraic("h3"),
            })
            .unwrap();
            game.collapse_all()
        };

        let board_a = run(999);
        let board_b = run(999);
        assert!(board_a.is_identical_to(&board_b), "same seed must produce identical collapse");

        // Different seed should (very likely) produce a different board.
        let board_c = run(1);
        // With 50/50 odds and different seeds, at least one should differ.
        // We run multiple seeds to be robust.
        let board_d = run(12345);
        let any_differ = !board_a.is_identical_to(&board_c)
            || !board_a.is_identical_to(&board_d);
        assert!(any_differ, "different seeds should usually produce different collapses");
    }

    // --- Test 21: MoveResult description is informative ---
    #[test]
    fn test_move_result_description() {
        let mut game = QuantumChessGame::new();
        let result = game
            .make_move(QuantumChessMove::Classical {
                from: Square::from_algebraic("e2"),
                to: Square::from_algebraic("e4"),
            })
            .unwrap();

        assert!(result.description.contains("Pawn"));
        assert!(result.description.contains("e2"));
        assert!(result.description.contains("e4"));
    }

    // --- Test 22: Square algebraic roundtrip ---
    #[test]
    fn test_square_algebraic_roundtrip() {
        for row in 0..8u8 {
            for col in 0..8u8 {
                let sq = Square::new(row, col);
                let alg = sq.to_algebraic();
                let sq2 = Square::from_algebraic(&alg);
                assert_eq!(sq, sq2, "roundtrip failed for ({}, {})", row, col);
            }
        }
    }

    // --- Test 23: Empty square has no probabilities ---
    #[test]
    fn test_empty_square_no_pieces() {
        let game = QuantumChessGame::new();
        let probs = game.piece_at_probability(Square::from_algebraic("e4"));
        assert!(probs.is_empty());
    }

    // --- Test 24: Pawn cannot move backwards ---
    #[test]
    fn test_pawn_cannot_move_backwards() {
        let mut game = QuantumChessGame::new();
        // Move pawn forward first.
        game.make_move(QuantumChessMove::Classical {
            from: Square::from_algebraic("e2"),
            to: Square::from_algebraic("e4"),
        })
        .unwrap();

        // Black plays.
        game.make_move(QuantumChessMove::Classical {
            from: Square::from_algebraic("a7"),
            to: Square::from_algebraic("a6"),
        })
        .unwrap();

        // White tries to move pawn backwards.
        let result = game.make_move(QuantumChessMove::Classical {
            from: Square::from_algebraic("e4"),
            to: Square::from_algebraic("e3"),
        });
        assert!(result.is_err());
    }

    // --- Test 25: Cannot move after game over ---
    #[test]
    fn test_cannot_move_after_game_over() {
        let mut game = QuantumChessGame::new();
        // Force game over.
        game.status = GameStatus::WhiteWins;

        let result = game.make_move(QuantumChessMove::Classical {
            from: Square::from_algebraic("e2"),
            to: Square::from_algebraic("e4"),
        });
        assert!(matches!(result, Err(ChessError::GameOver)));
    }

    // --- Test 26: Quantum split with same targets is rejected ---
    #[test]
    fn test_split_same_targets_rejected() {
        let mut game = QuantumChessGame::new();
        let result = game.make_move(QuantumChessMove::QuantumSplit {
            from: Square::from_algebraic("g1"),
            to_a: Square::from_algebraic("f3"),
            to_b: Square::from_algebraic("f3"),
        });
        assert!(matches!(result, Err(ChessError::InvalidSplit(_))));
    }

    // --- Test 27: Quantum split from origin to same as origin rejected ---
    #[test]
    fn test_split_to_origin_rejected() {
        let mut game = QuantumChessGame::new();
        let result = game.make_move(QuantumChessMove::QuantumSplit {
            from: Square::from_algebraic("g1"),
            to_a: Square::from_algebraic("g1"),
            to_b: Square::from_algebraic("f3"),
        });
        assert!(matches!(result, Err(ChessError::InvalidSplit(_))));
    }

    // --- Test 28: Branch limit enforced ---
    #[test]
    fn test_branch_limit_enforced() {
        let config = QuantumChessConfig::builder()
            .max_branches(2)
            .seed(42)
            .build();
        let mut game = QuantumChessGame::with_config(config);

        // First split: 1 -> 2, should succeed.
        game.make_move(QuantumChessMove::QuantumSplit {
            from: Square::from_algebraic("g1"),
            to_a: Square::from_algebraic("f3"),
            to_b: Square::from_algebraic("h3"),
        })
        .unwrap();

        // Black move.
        game.make_move(QuantumChessMove::Classical {
            from: Square::from_algebraic("a7"),
            to: Square::from_algebraic("a6"),
        })
        .unwrap();

        // Second split: 2 -> 4, should fail.
        let result = game.make_move(QuantumChessMove::QuantumSplit {
            from: Square::from_algebraic("b1"),
            to_a: Square::from_algebraic("a3"),
            to_b: Square::from_algebraic("c3"),
        });
        assert!(matches!(
            result,
            Err(ChessError::BranchLimitExceeded { .. })
        ));
    }

    // --- Test 29: Measure on non-superposed square is no-op ---
    #[test]
    fn test_measure_non_superposed_is_noop() {
        let mut game = QuantumChessGame::new();
        let result = game
            .make_move(QuantumChessMove::Measure {
                square: Square::from_algebraic("e1"),
            })
            .unwrap();

        assert!(!result.measurement_occurred);
        assert_eq!(result.branches_after, 1);
    }

    // --- Test 30: Config builder works ---
    #[test]
    fn test_config_builder() {
        let config = QuantumChessConfig::builder()
            .max_branches(512)
            .seed(999)
            .auto_merge(false)
            .auto_normalize(false)
            .build();

        assert_eq!(config.max_branches, 512);
        assert_eq!(config.seed, 999);
        assert!(!config.auto_merge);
        assert!(!config.auto_normalize);
    }

    // --- Test 31: LCG produces values in [0, 1) ---
    #[test]
    fn test_lcg_range() {
        let mut state: u64 = 42;
        for _ in 0..1000 {
            let val = lcg_f64(&mut state);
            assert!(val >= 0.0 && val < 1.0, "LCG out of range: {}", val);
        }
    }

    // --- Test 32: LCG is deterministic ---
    #[test]
    fn test_lcg_deterministic() {
        let mut s1: u64 = 12345;
        let mut s2: u64 = 12345;
        for _ in 0..100 {
            assert_eq!(lcg_next(&mut s1), lcg_next(&mut s2));
        }
    }

    // --- Test 33: Bishop moves diagonally ---
    #[test]
    fn test_bishop_diagonal_move() {
        let mut board = BoardState::empty();
        let bishop = Piece {
            kind: PieceKind::Bishop,
            color: Color::White,
            id: 0,
        };
        board.set_piece(Square::from_algebraic("c1"), Some(bishop));

        // Diagonal: c1 -> e3 is legal.
        assert!(is_legal_piece_move(
            PieceKind::Bishop,
            Color::White,
            Square::from_algebraic("c1"),
            Square::from_algebraic("e3"),
            &board,
        ));

        // Straight: c1 -> c3 is not legal for bishop.
        assert!(!is_legal_piece_move(
            PieceKind::Bishop,
            Color::White,
            Square::from_algebraic("c1"),
            Square::from_algebraic("c3"),
            &board,
        ));
    }

    // --- Test 34: Rook moves straight ---
    #[test]
    fn test_rook_straight_move() {
        let mut board = BoardState::empty();
        let rook = Piece {
            kind: PieceKind::Rook,
            color: Color::White,
            id: 0,
        };
        board.set_piece(Square::from_algebraic("a1"), Some(rook));

        // Vertical: a1 -> a5 legal.
        assert!(is_legal_piece_move(
            PieceKind::Rook,
            Color::White,
            Square::from_algebraic("a1"),
            Square::from_algebraic("a5"),
            &board,
        ));

        // Diagonal: a1 -> c3 not legal for rook.
        assert!(!is_legal_piece_move(
            PieceKind::Rook,
            Color::White,
            Square::from_algebraic("a1"),
            Square::from_algebraic("c3"),
            &board,
        ));
    }

    // --- Test 35: Queen moves both straight and diagonal ---
    #[test]
    fn test_queen_moves() {
        let mut board = BoardState::empty();
        let queen = Piece {
            kind: PieceKind::Queen,
            color: Color::White,
            id: 0,
        };
        board.set_piece(Square::from_algebraic("d4"), Some(queen));

        // Straight.
        assert!(is_legal_piece_move(
            PieceKind::Queen,
            Color::White,
            Square::from_algebraic("d4"),
            Square::from_algebraic("d8"),
            &board,
        ));
        // Diagonal.
        assert!(is_legal_piece_move(
            PieceKind::Queen,
            Color::White,
            Square::from_algebraic("d4"),
            Square::from_algebraic("g7"),
            &board,
        ));
        // L-shape (illegal for queen).
        assert!(!is_legal_piece_move(
            PieceKind::Queen,
            Color::White,
            Square::from_algebraic("d4"),
            Square::from_algebraic("e6"),
            &board,
        ));
    }

    // --- Test 36: Path obstruction blocks sliding pieces ---
    #[test]
    fn test_path_obstruction() {
        let mut board = BoardState::empty();
        let rook = Piece {
            kind: PieceKind::Rook,
            color: Color::White,
            id: 0,
        };
        let blocker = Piece {
            kind: PieceKind::Pawn,
            color: Color::White,
            id: 1,
        };
        board.set_piece(Square::from_algebraic("a1"), Some(rook));
        board.set_piece(Square::from_algebraic("a3"), Some(blocker));

        // a1 -> a2 is fine.
        assert!(is_legal_piece_move(
            PieceKind::Rook,
            Color::White,
            Square::from_algebraic("a1"),
            Square::from_algebraic("a2"),
            &board,
        ));

        // a1 -> a5 is blocked by own pawn on a3.
        assert!(!is_legal_piece_move(
            PieceKind::Rook,
            Color::White,
            Square::from_algebraic("a1"),
            Square::from_algebraic("a5"),
            &board,
        ));
    }

    // --- Test 37: Display format includes superposition percentages ---
    #[test]
    fn test_display_shows_probabilities() {
        let mut game = QuantumChessGame::new();
        game.make_move(QuantumChessMove::QuantumSplit {
            from: Square::from_algebraic("g1"),
            to_a: Square::from_algebraic("f3"),
            to_b: Square::from_algebraic("h3"),
        })
        .unwrap();

        let display = game.display_board();
        // Should show "50%" annotation for the superposed knight.
        assert!(
            display.contains("50%"),
            "display should show 50% for superposed piece:\n{}",
            display
        );
    }

    // --- Test 38: Game summary string ---
    #[test]
    fn test_summary_string() {
        let game = QuantumChessGame::new();
        let summary = game.summary();
        assert!(summary.contains("White"));
        assert!(summary.contains("Branches: 1"));
        assert!(summary.contains("In progress"));
    }

    // --- Test 39: Piece display characters ---
    #[test]
    fn test_piece_display_chars() {
        let white_king = Piece {
            kind: PieceKind::King,
            color: Color::White,
            id: 0,
        };
        let black_queen = Piece {
            kind: PieceKind::Queen,
            color: Color::Black,
            id: 1,
        };

        assert_eq!(white_king.display_char(), 'K');
        assert_eq!(black_queen.display_char(), 'q');
    }

    // --- Test 40: Pawn two-square advance from start ---
    #[test]
    fn test_pawn_two_square_from_start() {
        let mut game = QuantumChessGame::new();
        // d2 -> d4 (two squares from start row).
        let result = game.make_move(QuantumChessMove::Classical {
            from: Square::from_algebraic("d2"),
            to: Square::from_algebraic("d4"),
        });
        assert!(result.is_ok());
    }

    // --- Test 41: Pawn two-square advance blocked ---
    #[test]
    fn test_pawn_two_square_blocked() {
        let mut board = BoardState::empty();
        let pawn = Piece {
            kind: PieceKind::Pawn,
            color: Color::White,
            id: 0,
        };
        let blocker = Piece {
            kind: PieceKind::Pawn,
            color: Color::Black,
            id: 1,
        };
        board.set_piece(Square::from_algebraic("e2"), Some(pawn));
        board.set_piece(Square::from_algebraic("e3"), Some(blocker));

        // e2 -> e4 should be blocked by piece on e3.
        assert!(!is_legal_piece_move(
            PieceKind::Pawn,
            Color::White,
            Square::from_algebraic("e2"),
            Square::from_algebraic("e4"),
            &board,
        ));
    }

    // --- Test 42: Error display messages ---
    #[test]
    fn test_error_display() {
        let err = ChessError::IllegalMove("test".to_string());
        assert_eq!(format!("{}", err), "illegal move: test");

        let err = ChessError::NoPieceAt(Square::from_algebraic("e4"));
        assert!(format!("{}", err).contains("e4"));

        let err = ChessError::BranchLimitExceeded {
            current: 100,
            limit: 64,
        };
        assert!(format!("{}", err).contains("100"));
    }

    // --- Test 43: Endgame scenario constructs correctly ---
    #[test]
    fn test_endgame_scenario() {
        let game = QuantumChessScenarios::endgame_kq_vs_k();
        assert_eq!(game.num_branches(), 1);

        let board = &game.branches()[0].0;
        assert_eq!(board.piece_count(), 3);

        let white_king = board.piece_at(Square::from_algebraic("e1")).unwrap();
        assert_eq!(white_king.kind, PieceKind::King);
        assert_eq!(white_king.color, Color::White);

        let white_queen = board.piece_at(Square::from_algebraic("d4")).unwrap();
        assert_eq!(white_queen.kind, PieceKind::Queen);

        let black_king = board.piece_at(Square::from_algebraic("e8")).unwrap();
        assert_eq!(black_king.kind, PieceKind::King);
        assert_eq!(black_king.color, Color::Black);
    }

    // --- Test 44: Measure explicitly on a superposed square ---
    #[test]
    fn test_explicit_measure() {
        let mut game = QuantumChessGame::new();

        // White knight splits.
        game.make_move(QuantumChessMove::QuantumSplit {
            from: Square::from_algebraic("g1"),
            to_a: Square::from_algebraic("f3"),
            to_b: Square::from_algebraic("h3"),
        })
        .unwrap();
        assert_eq!(game.num_branches(), 2);

        // Explicitly measure f3.
        let result = game
            .make_move(QuantumChessMove::Measure {
                square: Square::from_algebraic("f3"),
            })
            .unwrap();

        assert!(result.measurement_occurred);
        assert_eq!(result.branches_after, 1);
    }

    // --- Test 45: GameStatus display ---
    #[test]
    fn test_game_status_display() {
        assert_eq!(format!("{}", GameStatus::InProgress), "In progress");
        assert_eq!(format!("{}", GameStatus::WhiteWins), "White wins");
        assert_eq!(format!("{}", GameStatus::BlackWins), "Black wins");
        assert_eq!(format!("{}", GameStatus::Draw), "Draw");
    }

    // --- Test 46: Move display ---
    #[test]
    fn test_move_display() {
        let classical = QuantumChessMove::Classical {
            from: Square::from_algebraic("e2"),
            to: Square::from_algebraic("e4"),
        };
        assert_eq!(format!("{}", classical), "e2 -> e4");

        let split = QuantumChessMove::QuantumSplit {
            from: Square::from_algebraic("g1"),
            to_a: Square::from_algebraic("f3"),
            to_b: Square::from_algebraic("h3"),
        };
        assert!(format!("{}", split).contains("~>"));

        let measure = QuantumChessMove::Measure {
            square: Square::from_algebraic("e4"),
        };
        assert!(format!("{}", measure).contains("Measure"));
    }

    // --- Test 47: Default game trait ---
    #[test]
    fn test_default_trait() {
        let game: QuantumChessGame = Default::default();
        assert_eq!(game.num_branches(), 1);
        assert_eq!(game.current_turn(), Color::White);
        assert_eq!(*game.status(), GameStatus::InProgress);
    }

    // --- Test 48: King one square move ---
    #[test]
    fn test_king_one_square() {
        let mut board = BoardState::empty();
        let king = Piece {
            kind: PieceKind::King,
            color: Color::White,
            id: 0,
        };
        board.set_piece(Square::from_algebraic("e4"), Some(king));

        // One square in any direction is fine.
        for &target in &["d3", "d4", "d5", "e3", "e5", "f3", "f4", "f5"] {
            assert!(
                is_legal_piece_move(
                    PieceKind::King,
                    Color::White,
                    Square::from_algebraic("e4"),
                    Square::from_algebraic(target),
                    &board,
                ),
                "King should be able to move to {}",
                target
            );
        }

        // Two squares is not.
        assert!(!is_legal_piece_move(
            PieceKind::King,
            Color::White,
            Square::from_algebraic("e4"),
            Square::from_algebraic("e6"),
            &board,
        ));
    }

    // --- Test 49: Colour opposite ---
    #[test]
    fn test_colour_opposite() {
        assert_eq!(Color::White.opposite(), Color::Black);
        assert_eq!(Color::Black.opposite(), Color::White);
    }

    // --- Test 50: Knight L-shape exhaustive ---
    #[test]
    fn test_knight_l_shape_exhaustive() {
        let board = BoardState::empty();
        let from = Square::from_algebraic("d4");

        let valid_targets = [
            "c6", "e6", "f5", "f3", "e2", "c2", "b3", "b5",
        ];
        for target in &valid_targets {
            assert!(
                is_legal_piece_move(
                    PieceKind::Knight,
                    Color::White,
                    from,
                    Square::from_algebraic(target),
                    &board,
                ),
                "Knight should reach {} from d4",
                target
            );
        }

        // Non-L-shape targets.
        let invalid_targets = ["d5", "d6", "e4", "e5", "c4", "d3"];
        for target in &invalid_targets {
            assert!(
                !is_legal_piece_move(
                    PieceKind::Knight,
                    Color::White,
                    from,
                    Square::from_algebraic(target),
                    &board,
                ),
                "Knight should NOT reach {} from d4",
                target
            );
        }
    }

    // --- Test 51: Black pawn moves downward ---
    #[test]
    fn test_black_pawn_direction() {
        let mut board = BoardState::empty();
        let pawn = Piece {
            kind: PieceKind::Pawn,
            color: Color::Black,
            id: 0,
        };
        board.set_piece(Square::from_algebraic("e7"), Some(pawn));

        // Black pawn should move down: e7 -> e6.
        assert!(is_legal_piece_move(
            PieceKind::Pawn,
            Color::Black,
            Square::from_algebraic("e7"),
            Square::from_algebraic("e6"),
            &board,
        ));

        // Two squares from start: e7 -> e5.
        assert!(is_legal_piece_move(
            PieceKind::Pawn,
            Color::Black,
            Square::from_algebraic("e7"),
            Square::from_algebraic("e5"),
            &board,
        ));

        // Should NOT move up: e7 -> e8.
        assert!(!is_legal_piece_move(
            PieceKind::Pawn,
            Color::Black,
            Square::from_algebraic("e7"),
            Square::from_algebraic("e8"),
            &board,
        ));
    }

    // --- Test 52: Full game Display trait ---
    #[test]
    fn test_game_display_trait() {
        let game = QuantumChessGame::new();
        let output = format!("{}", game);
        assert!(output.contains("Turn:"));
        assert!(output.contains("Branches:"));
    }
}
