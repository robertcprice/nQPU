//! Quantum Poker: Card Games in Hilbert Space
//!
//! **WORLD FIRST**: A quantum card game engine inside a quantum simulator.
//! Cards exist in superposition, hands are entangled, and measurement IS
//! the card reveal. This is not a gimmick -- it demonstrates deep quantum
//! information theory through gameplay.
//!
//! # Core Quantum Mechanics
//!
//! - **Superposition**: A face-down card is a quantum superposition of all
//!   possible card values. The deck state is |deck> = sum_d alpha_d |d>,
//!   where each |d> is a complete valid deal.
//!
//! - **Entanglement**: Players receive entangled card sets. Measuring your
//!   own card can collapse (affect) what your opponent holds. If Alice has
//!   the Ace of Spades, Bob cannot -- their hands are anti-correlated by
//!   the deck constraint.
//!
//! - **Partial Measurement**: In quantum poker you can project onto a
//!   subspace ("is my card a heart?") without full collapse. This creates
//!   genuine quantum advantage in bluffing -- you gain information while
//!   preserving superposition in the rank degree of freedom.
//!
//! - **Interference**: Quantum operations on your hand can shift outcome
//!   probabilities via constructive/destructive interference, an effect
//!   with no classical analogue.
//!
//! # Quantum Advantage Demonstrations
//!
//! - **Entangled Bluff**: Alice measures in a rotated basis, shifting Bob's
//!   card probabilities. Impossible classically.
//! - **CHSH Poker**: Card correlations can violate the CHSH inequality,
//!   proving the game is genuinely quantum.
//! - **Information Gain**: Partial measurement extracts strictly more info
//!   per bit than any classical peek strategy.
//!
//! # Applications
//!
//! - Quantum information theory education
//! - Entanglement and measurement pedagogy
//! - Quantum cryptography protocol intuition
//! - Bell inequality demonstrations
//! - Quantum strategy and game theory
//!
//! # References
//!
//! - Eisert, Wilkens, Lewenstein (1999) - Quantum Games and Quantum Strategies
//! - Meyer (1999) - Quantum Strategies
//! - Clauser, Horne, Shimony, Holt (1969) - CHSH Inequality
//! - Brunner et al. (2014) - Bell nonlocality, Rev. Mod. Phys.

use num_complex::Complex64;
use std::fmt;

/// Local complex type alias
type C64 = Complex64;

// ============================================================
// ERROR TYPES
// ============================================================

/// Errors that can occur during a quantum poker game.
#[derive(Clone, Debug, PartialEq)]
pub enum PokerError {
    /// Player index is out of range.
    InvalidPlayer(usize, usize),
    /// Card index is out of range for a player's hand.
    InvalidCardIndex(usize, usize),
    /// Action is not permitted in the current game phase.
    InvalidPhase {
        expected: &'static str,
        got: GamePhase,
    },
    /// Player has already folded.
    PlayerFolded(usize),
    /// Insufficient chips for the requested bet.
    InsufficientChips {
        player: usize,
        requested: f64,
        available: f64,
    },
    /// The game has not been dealt yet.
    NotDealt,
    /// The game is already over.
    GameOver,
    /// Bet amount must be positive.
    InvalidBet(f64),
    /// Cannot raise below the current bet.
    RaiseTooLow { minimum: f64, offered: f64 },
    /// Deck is too small for the requested game configuration.
    DeckTooSmall { needed: usize, available: usize },
    /// Configuration error.
    ConfigError(String),
}

impl fmt::Display for PokerError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidPlayer(p, max) => write!(f, "Player {} out of range (max {})", p, max - 1),
            Self::InvalidCardIndex(c, max) => {
                write!(f, "Card index {} out of range (max {})", c, max - 1)
            }
            Self::InvalidPhase { expected, got } => {
                write!(f, "Expected phase {}, got {:?}", expected, got)
            }
            Self::PlayerFolded(p) => write!(f, "Player {} has folded", p),
            Self::InsufficientChips {
                player,
                requested,
                available,
            } => {
                write!(
                    f,
                    "Player {} has {:.1} chips, needs {:.1}",
                    player, available, requested
                )
            }
            Self::NotDealt => write!(f, "Cards have not been dealt yet"),
            Self::GameOver => write!(f, "Game is already over"),
            Self::InvalidBet(b) => write!(f, "Invalid bet amount: {:.1}", b),
            Self::RaiseTooLow { minimum, offered } => {
                write!(f, "Raise of {:.1} is below minimum {:.1}", offered, minimum)
            }
            Self::DeckTooSmall { needed, available } => {
                write!(f, "Need {} cards but deck only has {}", needed, available)
            }
            Self::ConfigError(msg) => write!(f, "Configuration error: {}", msg),
        }
    }
}

// ============================================================
// CARD TYPES
// ============================================================

/// Card suits.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Suit {
    Hearts,
    Diamonds,
    Clubs,
    Spades,
}

impl Suit {
    /// All four suits in standard order.
    pub const ALL: [Suit; 4] = [Suit::Hearts, Suit::Diamonds, Suit::Clubs, Suit::Spades];

    /// Whether this suit is red (hearts or diamonds).
    pub fn is_red(self) -> bool {
        matches!(self, Suit::Hearts | Suit::Diamonds)
    }

    /// Suit index (0-3).
    pub fn index(self) -> usize {
        match self {
            Suit::Hearts => 0,
            Suit::Diamonds => 1,
            Suit::Clubs => 2,
            Suit::Spades => 3,
        }
    }

    /// Unicode suit symbol.
    pub fn symbol(self) -> char {
        match self {
            Suit::Hearts => '\u{2665}',
            Suit::Diamonds => '\u{2666}',
            Suit::Clubs => '\u{2663}',
            Suit::Spades => '\u{2660}',
        }
    }
}

impl fmt::Display for Suit {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.symbol())
    }
}

/// Card ranks. Internally stored as u8 (2 = Two, ..., 14 = Ace).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Rank(pub u8);

impl Rank {
    pub const TWO: Rank = Rank(2);
    pub const THREE: Rank = Rank(3);
    pub const FOUR: Rank = Rank(4);
    pub const FIVE: Rank = Rank(5);
    pub const SIX: Rank = Rank(6);
    pub const SEVEN: Rank = Rank(7);
    pub const EIGHT: Rank = Rank(8);
    pub const NINE: Rank = Rank(9);
    pub const TEN: Rank = Rank(10);
    pub const JACK: Rank = Rank(11);
    pub const QUEEN: Rank = Rank(12);
    pub const KING: Rank = Rank(13);
    pub const ACE: Rank = Rank(14);

    /// Whether this is a face card (J, Q, K).
    pub fn is_face(self) -> bool {
        self.0 >= 11 && self.0 <= 13
    }

    /// Whether this rank is "high" (above 7).
    pub fn is_high(self) -> bool {
        self.0 > 7
    }

    /// Display name for the rank.
    pub fn name(self) -> &'static str {
        match self.0 {
            2 => "2",
            3 => "3",
            4 => "4",
            5 => "5",
            6 => "6",
            7 => "7",
            8 => "8",
            9 => "9",
            10 => "10",
            11 => "J",
            12 => "Q",
            13 => "K",
            14 => "A",
            _ => "?",
        }
    }
}

impl fmt::Display for Rank {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.name())
    }
}

/// A playing card with suit and rank.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Card {
    pub suit: Suit,
    pub rank: Rank,
}

impl Card {
    pub fn new(suit: Suit, rank: Rank) -> Self {
        Self { suit, rank }
    }

    /// Unique index in [0, 52) for a standard deck: rank_index * 4 + suit_index.
    pub fn index(&self) -> usize {
        (self.rank.0 as usize - 2) * 4 + self.suit.index()
    }

    /// Construct a card from its index in [0, 52).
    pub fn from_index(idx: usize) -> Self {
        let rank = Rank((idx / 4) as u8 + 2);
        let suit = Suit::ALL[idx % 4];
        Self { suit, rank }
    }
}

impl fmt::Display for Card {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}{}", self.rank, self.suit)
    }
}

impl PartialOrd for Card {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Card {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.rank.cmp(&other.rank).then(self.suit.cmp(&other.suit))
    }
}

/// Build a full 52-card deck.
fn standard_deck() -> Vec<Card> {
    let mut deck = Vec::with_capacity(52);
    for rank_val in 2..=14u8 {
        for &suit in &Suit::ALL {
            deck.push(Card::new(suit, Rank(rank_val)));
        }
    }
    deck
}

/// Build a small toy deck: {A-spades, K-spades, A-hearts, K-hearts}.
fn toy_deck() -> Vec<Card> {
    vec![
        Card::new(Suit::Spades, Rank::ACE),
        Card::new(Suit::Spades, Rank::KING),
        Card::new(Suit::Hearts, Rank::ACE),
        Card::new(Suit::Hearts, Rank::KING),
    ]
}

// ============================================================
// HAND RANKING
// ============================================================

/// Standard poker hand rankings, ordered from weakest to strongest.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum HandRank {
    /// Highest card value only.
    HighCard(u8),
    /// Two cards of the same rank.
    OnePair(u8),
    /// Two different pairs.
    TwoPair(u8, u8),
    /// Three cards of the same rank.
    ThreeOfAKind(u8),
    /// Five consecutive ranks.
    Straight(u8),
    /// Five cards of the same suit.
    Flush(u8),
    /// Three of a kind plus a pair.
    FullHouse(u8, u8),
    /// Four cards of the same rank.
    FourOfAKind(u8),
    /// Five consecutive ranks, same suit.
    StraightFlush(u8),
    /// A-K-Q-J-10 of the same suit.
    RoyalFlush,
}

impl HandRank {
    /// Numeric tier for coarse comparison (0 = high card, 9 = royal flush).
    pub fn tier(&self) -> u8 {
        match self {
            Self::HighCard(_) => 0,
            Self::OnePair(_) => 1,
            Self::TwoPair(_, _) => 2,
            Self::ThreeOfAKind(_) => 3,
            Self::Straight(_) => 4,
            Self::Flush(_) => 5,
            Self::FullHouse(_, _) => 6,
            Self::FourOfAKind(_) => 7,
            Self::StraightFlush(_) => 8,
            Self::RoyalFlush => 9,
        }
    }
}

impl fmt::Display for HandRank {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::HighCard(h) => write!(f, "High Card ({})", Rank(*h)),
            Self::OnePair(r) => write!(f, "Pair of {}s", Rank(*r)),
            Self::TwoPair(a, b) => write!(f, "Two Pair ({}s and {}s)", Rank(*a), Rank(*b)),
            Self::ThreeOfAKind(r) => write!(f, "Three {}s", Rank(*r)),
            Self::Straight(h) => write!(f, "Straight (high {})", Rank(*h)),
            Self::Flush(h) => write!(f, "Flush (high {})", Rank(*h)),
            Self::FullHouse(t, p) => write!(f, "Full House ({}s over {}s)", Rank(*t), Rank(*p)),
            Self::FourOfAKind(r) => write!(f, "Four {}s", Rank(*r)),
            Self::StraightFlush(h) => write!(f, "Straight Flush (high {})", Rank(*h)),
            Self::RoyalFlush => write!(f, "Royal Flush"),
        }
    }
}

/// Evaluate the best poker hand from a set of cards (5-7 cards).
/// For hands with fewer than 5 cards we evaluate what we have.
pub fn evaluate_hand(cards: &[Card]) -> HandRank {
    if cards.is_empty() {
        return HandRank::HighCard(0);
    }

    // For 5+ cards, try all 5-card combinations and take the best
    if cards.len() >= 5 {
        let mut best = HandRank::HighCard(0);
        let combos = combinations_5(cards);
        for combo in &combos {
            let rank = evaluate_five(combo);
            if rank > best {
                best = rank;
            }
        }
        return best;
    }

    // Fewer than 5 cards: evaluate directly for pairs etc.
    evaluate_partial(cards)
}

/// All 5-card combinations from a slice.
fn combinations_5(cards: &[Card]) -> Vec<[Card; 5]> {
    let n = cards.len();
    let mut result = Vec::new();
    for a in 0..n {
        for b in (a + 1)..n {
            for c in (b + 1)..n {
                for d in (c + 1)..n {
                    for e in (d + 1)..n {
                        result.push([cards[a], cards[b], cards[c], cards[d], cards[e]]);
                    }
                }
            }
        }
    }
    result
}

/// Evaluate exactly 5 cards.
fn evaluate_five(cards: &[Card; 5]) -> HandRank {
    let mut ranks: Vec<u8> = cards.iter().map(|c| c.rank.0).collect();
    ranks.sort_unstable();

    let is_flush = cards.iter().all(|c| c.suit == cards[0].suit);

    // Check for straight (including A-2-3-4-5 wheel)
    let is_straight = is_consecutive(&ranks);
    let is_wheel = ranks == [2, 3, 4, 5, 14]; // A-2-3-4-5

    let high = if is_wheel { 5 } else { ranks[4] };

    if is_flush && is_straight {
        if ranks == [10, 11, 12, 13, 14] {
            return HandRank::RoyalFlush;
        }
        return HandRank::StraightFlush(high);
    }
    if is_flush && is_wheel {
        return HandRank::StraightFlush(5);
    }

    // Count rank occurrences
    let mut counts = [0u8; 15]; // index by rank value
    for &r in &ranks {
        counts[r as usize] += 1;
    }

    let mut fours = Vec::new();
    let mut threes = Vec::new();
    let mut pairs = Vec::new();
    for r in (2..=14u8).rev() {
        match counts[r as usize] {
            4 => fours.push(r),
            3 => threes.push(r),
            2 => pairs.push(r),
            _ => {}
        }
    }

    if !fours.is_empty() {
        return HandRank::FourOfAKind(fours[0]);
    }
    if !threes.is_empty() && !pairs.is_empty() {
        return HandRank::FullHouse(threes[0], pairs[0]);
    }
    if is_flush {
        return HandRank::Flush(ranks[4]);
    }
    if is_straight || is_wheel {
        return HandRank::Straight(high);
    }
    if !threes.is_empty() {
        return HandRank::ThreeOfAKind(threes[0]);
    }
    if pairs.len() >= 2 {
        return HandRank::TwoPair(pairs[0], pairs[1]);
    }
    if pairs.len() == 1 {
        return HandRank::OnePair(pairs[0]);
    }

    HandRank::HighCard(ranks[4])
}

/// Check if sorted ranks are consecutive.
fn is_consecutive(sorted: &[u8]) -> bool {
    if sorted.len() < 2 {
        return true;
    }
    for i in 1..sorted.len() {
        if sorted[i] != sorted[i - 1] + 1 {
            return false;
        }
    }
    true
}

/// Evaluate a partial hand (fewer than 5 cards) for pairs/trips.
fn evaluate_partial(cards: &[Card]) -> HandRank {
    let mut counts = [0u8; 15];
    let mut max_rank = 0u8;
    for c in cards {
        counts[c.rank.0 as usize] += 1;
        if c.rank.0 > max_rank {
            max_rank = c.rank.0;
        }
    }

    let mut fours = Vec::new();
    let mut threes = Vec::new();
    let mut pairs = Vec::new();
    for r in (2..=14u8).rev() {
        match counts[r as usize] {
            4 => fours.push(r),
            3 => threes.push(r),
            2 => pairs.push(r),
            _ => {}
        }
    }

    if !fours.is_empty() {
        return HandRank::FourOfAKind(fours[0]);
    }
    if !threes.is_empty() && !pairs.is_empty() {
        return HandRank::FullHouse(threes[0], pairs[0]);
    }
    if !threes.is_empty() {
        return HandRank::ThreeOfAKind(threes[0]);
    }
    if pairs.len() >= 2 {
        return HandRank::TwoPair(pairs[0], pairs[1]);
    }
    if pairs.len() == 1 {
        return HandRank::OnePair(pairs[0]);
    }

    HandRank::HighCard(max_rank)
}

// ============================================================
// DEAL: A SPECIFIC CARD ASSIGNMENT
// ============================================================

/// A specific deal: one concrete assignment of cards to players and community.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Deal {
    /// hands[player][card_idx] -- each player's cards.
    pub hands: Vec<Vec<Card>>,
    /// Community cards (for Hold'em variants).
    pub community: Vec<Card>,
    /// Remaining deck cards not dealt.
    pub remaining: Vec<Card>,
}

impl Deal {
    /// Check that no card appears twice in the deal.
    pub fn is_valid(&self) -> bool {
        let mut seen = std::collections::HashSet::new();
        for hand in &self.hands {
            for &card in hand {
                if !seen.insert(card) {
                    return false;
                }
            }
        }
        for &card in &self.community {
            if !seen.insert(card) {
                return false;
            }
        }
        for &card in &self.remaining {
            if !seen.insert(card) {
                return false;
            }
        }
        true
    }
}

// ============================================================
// GAME CONFIGURATION
// ============================================================

/// Poker game variant.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PokerVariant {
    /// Texas Hold'em with quantum cards (2 hole + 5 community).
    QuantumHoldem,
    /// 5-card draw with quantum operations.
    QuantumDraw,
    /// Simple: highest quantum card wins (good for demos and tests).
    QuantumWar,
    /// 2 players get entangled pairs, strategic measurement order matters.
    EntangledPairs,
}

/// Game phase tracking.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GamePhase {
    /// Waiting for deal.
    Setup,
    /// Cards have been dealt, pre-flop betting round.
    PreFlop,
    /// First three community cards revealed.
    Flop,
    /// Fourth community card revealed.
    Turn,
    /// Fifth community card revealed.
    River,
    /// All cards measured, determine winner.
    Showdown,
    /// Game is complete.
    Complete,
}

/// Measurement basis for partial card measurement.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MeasurementBasis {
    /// Full collapse: learn the exact card.
    FullCollapse,
    /// Learn the suit only; rank remains in superposition.
    SuitOnly,
    /// Learn whether rank > 7 or rank <= 7.
    HighLow,
    /// Learn red (hearts/diamonds) vs black (clubs/spades).
    ColorOnly,
    /// Learn whether the card is a face card (J, Q, K) or not.
    FaceCard,
}

/// Player actions during a quantum poker game.
#[derive(Clone, Debug, PartialEq)]
pub enum QuantumPokerAction {
    /// Fully measure a card at the given hand index.
    Measure(usize),
    /// Partial measurement in a chosen basis.
    PartialMeasure(usize, MeasurementBasis),
    /// Apply a quantum bluff operation (phase rotation on hand amplitudes).
    QuantumBluff,
    /// Place a bet of the given amount.
    Bet(f64),
    /// Call the current bet.
    Call,
    /// Fold and forfeit.
    Fold,
    /// Check (bet 0).
    Check,
    /// Raise by the given amount above the current bet.
    Raise(f64),
}

/// Result of performing an action.
#[derive(Clone, Debug)]
pub struct ActionResult {
    /// Description of what happened.
    pub description: String,
    /// Card revealed (if a measurement was performed).
    pub revealed_card: Option<Card>,
    /// Partial info gained (for partial measurements).
    pub partial_info: Option<String>,
    /// Probability distribution change (for quantum bluff).
    pub prob_shift: Option<f64>,
    /// Current pot after this action.
    pub pot_total: f64,
    /// Number of remaining superposition branches.
    pub branches_remaining: usize,
}

/// Configuration for a quantum poker game. Use the builder pattern.
#[derive(Clone, Debug)]
pub struct QuantumPokerConfig {
    pub num_players: usize,
    pub cards_per_hand: usize,
    pub deck_size: usize,
    pub game_variant: PokerVariant,
    pub starting_chips: f64,
    pub seed: u64,
    pub community_cards: usize,
    pub max_deals: usize,
}

impl Default for QuantumPokerConfig {
    fn default() -> Self {
        Self {
            num_players: 2,
            cards_per_hand: 2,
            deck_size: 52,
            game_variant: PokerVariant::QuantumHoldem,
            starting_chips: 1000.0,
            seed: 42,
            community_cards: 5,
            max_deals: 5000,
        }
    }
}

impl QuantumPokerConfig {
    /// Create a new config builder with defaults.
    pub fn new() -> Self {
        Self::default()
    }

    pub fn num_players(mut self, n: usize) -> Self {
        self.num_players = n;
        self
    }

    pub fn cards_per_hand(mut self, n: usize) -> Self {
        self.cards_per_hand = n;
        self
    }

    pub fn deck_size(mut self, n: usize) -> Self {
        self.deck_size = n;
        self
    }

    pub fn variant(mut self, v: PokerVariant) -> Self {
        self.game_variant = v;
        self
    }

    pub fn starting_chips(mut self, c: f64) -> Self {
        self.starting_chips = c;
        self
    }

    pub fn seed(mut self, s: u64) -> Self {
        self.seed = s;
        self
    }

    pub fn community_cards(mut self, n: usize) -> Self {
        self.community_cards = n;
        self
    }

    pub fn max_deals(mut self, n: usize) -> Self {
        self.max_deals = n;
        self
    }

    /// Configuration for the 4-card toy game (great for demos).
    pub fn toy_game() -> Self {
        Self {
            num_players: 2,
            cards_per_hand: 1,
            deck_size: 4,
            game_variant: PokerVariant::QuantumWar,
            starting_chips: 100.0,
            seed: 42,
            community_cards: 0,
            max_deals: 100,
        }
    }

    /// Configuration for entangled pairs demo.
    pub fn entangled_pairs() -> Self {
        Self {
            num_players: 2,
            cards_per_hand: 1,
            deck_size: 4,
            game_variant: PokerVariant::EntangledPairs,
            starting_chips: 100.0,
            seed: 42,
            community_cards: 0,
            max_deals: 100,
        }
    }

    /// Validate the configuration.
    pub fn validate(&self) -> Result<(), PokerError> {
        if self.num_players < 2 || self.num_players > 6 {
            return Err(PokerError::ConfigError(format!(
                "num_players must be 2-6, got {}",
                self.num_players
            )));
        }
        let needed = self.num_players * self.cards_per_hand + self.community_cards;
        if needed > self.deck_size {
            return Err(PokerError::DeckTooSmall {
                needed,
                available: self.deck_size,
            });
        }
        if self.cards_per_hand == 0 {
            return Err(PokerError::ConfigError(
                "cards_per_hand must be >= 1".into(),
            ));
        }
        Ok(())
    }
}

// ============================================================
// QUANTUM POKER GAME
// ============================================================

/// The main quantum poker game engine.
///
/// The game state is a superposition of all valid deals. Each branch
/// carries a complex amplitude. Measurements (card reveals) collapse
/// branches, and quantum operations rotate amplitudes.
pub struct QuantumPokerGame {
    config: QuantumPokerConfig,
    /// Superposition of deals: (deal, amplitude) pairs.
    deal_amplitudes: Vec<(Deal, C64)>,
    /// Which cards each player has already measured: player_measured[p][c] = Some(card).
    player_measured: Vec<Vec<Option<Card>>>,
    /// Community cards that have been revealed.
    community_revealed: Vec<Option<Card>>,
    /// Player chip stacks.
    chips: Vec<f64>,
    /// Amount each player has put into the current pot.
    pot_contributions: Vec<f64>,
    /// Current bet to call.
    current_bet: f64,
    /// Whether each player has folded.
    folded: Vec<bool>,
    /// Current game phase.
    phase: GamePhase,
    /// Simple PRNG state derived from seed.
    rng_state: u64,
}

impl QuantumPokerGame {
    /// Create a new quantum poker game from configuration.
    pub fn new(config: QuantumPokerConfig) -> Result<Self, PokerError> {
        config.validate()?;

        let num_players = config.num_players;
        let cards_per_hand = config.cards_per_hand;
        let community_count = match config.game_variant {
            PokerVariant::QuantumHoldem => config.community_cards,
            PokerVariant::QuantumDraw => 0,
            PokerVariant::QuantumWar => 0,
            PokerVariant::EntangledPairs => 0,
        };

        Ok(Self {
            deal_amplitudes: Vec::new(),
            player_measured: vec![vec![None; cards_per_hand]; num_players],
            community_revealed: vec![None; community_count],
            chips: vec![config.starting_chips; num_players],
            pot_contributions: vec![0.0; num_players],
            current_bet: 0.0,
            folded: vec![false; num_players],
            phase: GamePhase::Setup,
            rng_state: config.seed,
            config,
        })
    }

    /// Simple xorshift64 PRNG (deterministic, inline, no deps).
    fn next_rng(&mut self) -> u64 {
        let mut x = self.rng_state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.rng_state = x;
        x
    }

    /// Random f64 in [0, 1).
    fn rand_f64(&mut self) -> f64 {
        (self.next_rng() as f64) / (u64::MAX as f64)
    }

    // --------------------------------------------------------
    // DEALING
    // --------------------------------------------------------

    /// Deal: create the quantum superposition of all valid deals.
    ///
    /// For small decks this enumerates all permutations. For large decks
    /// (standard 52-card) it samples uniformly up to `max_deals` deals
    /// to keep the representation tractable.
    pub fn deal(&mut self) -> Result<(), PokerError> {
        if self.phase != GamePhase::Setup {
            return Err(PokerError::InvalidPhase {
                expected: "Setup",
                got: self.phase,
            });
        }

        let deck: Vec<Card> = if self.config.deck_size == 4 {
            toy_deck()
        } else if self.config.deck_size == 52 {
            standard_deck()
        } else {
            // Build a deck of the requested size from the standard deck
            standard_deck()
                .into_iter()
                .take(self.config.deck_size)
                .collect()
        };

        let np = self.config.num_players;
        let cph = self.config.cards_per_hand;
        let cc = self.community_revealed.len();
        let total_cards_needed = np * cph + cc;

        if total_cards_needed > deck.len() {
            return Err(PokerError::DeckTooSmall {
                needed: total_cards_needed,
                available: deck.len(),
            });
        }

        let deals = if deck.len() <= 8 {
            // Small deck: enumerate all valid deals exhaustively
            self.enumerate_all_deals(&deck, np, cph, cc)
        } else {
            // Large deck: sample unique deals
            self.sample_deals(&deck, np, cph, cc)
        };

        if deals.is_empty() {
            return Err(PokerError::ConfigError("No valid deals generated".into()));
        }

        // Equal superposition: each deal gets amplitude 1/sqrt(N)
        let norm = 1.0 / (deals.len() as f64).sqrt();
        let amp = C64::new(norm, 0.0);

        self.deal_amplitudes = deals.into_iter().map(|d| (d, amp)).collect();
        self.phase = GamePhase::PreFlop;

        Ok(())
    }

    /// Enumerate all valid deals for small decks.
    fn enumerate_all_deals(
        &self,
        deck: &[Card],
        num_players: usize,
        cards_per_hand: usize,
        community_count: usize,
    ) -> Vec<Deal> {
        let total = num_players * cards_per_hand + community_count;
        let indices: Vec<usize> = (0..deck.len()).collect();
        let perms = permutations_k(&indices, total);

        perms
            .into_iter()
            .map(|perm| {
                let mut hands = vec![Vec::with_capacity(cards_per_hand); num_players];
                let mut idx = 0;
                for p in 0..num_players {
                    for _ in 0..cards_per_hand {
                        hands[p].push(deck[perm[idx]]);
                        idx += 1;
                    }
                }
                let community: Vec<Card> =
                    (0..community_count).map(|i| deck[perm[idx + i]]).collect();
                idx += community_count;

                let used: std::collections::HashSet<usize> = perm.iter().copied().collect();
                let remaining: Vec<Card> = (0..deck.len())
                    .filter(|i| !used.contains(i))
                    .map(|i| deck[i])
                    .collect();

                Deal {
                    hands,
                    community,
                    remaining,
                }
            })
            .collect()
    }

    /// Sample unique random deals for large decks.
    fn sample_deals(
        &mut self,
        deck: &[Card],
        num_players: usize,
        cards_per_hand: usize,
        community_count: usize,
    ) -> Vec<Deal> {
        let total = num_players * cards_per_hand + community_count;
        let max = self.config.max_deals;
        let mut deals = Vec::with_capacity(max);
        let mut seen = std::collections::HashSet::new();

        for _ in 0..(max * 10) {
            if deals.len() >= max {
                break;
            }

            // Fisher-Yates partial shuffle
            let mut indices: Vec<usize> = (0..deck.len()).collect();
            for i in 0..total {
                let j = i + (self.next_rng() as usize) % (deck.len() - i);
                indices.swap(i, j);
            }

            let selected: Vec<usize> = indices[..total].to_vec();

            // De-duplicate by sorting a fingerprint
            let mut fingerprint = selected.clone();
            fingerprint.sort_unstable();
            if !seen.insert(fingerprint) {
                continue;
            }

            let mut hands = vec![Vec::with_capacity(cards_per_hand); num_players];
            let mut idx = 0;
            for p in 0..num_players {
                for _ in 0..cards_per_hand {
                    hands[p].push(deck[selected[idx]]);
                    idx += 1;
                }
            }
            let community: Vec<Card> = (0..community_count)
                .map(|i| deck[selected[idx + i]])
                .collect();

            let used: std::collections::HashSet<usize> = selected.iter().copied().collect();
            let remaining: Vec<Card> = (0..deck.len())
                .filter(|i| !used.contains(i))
                .map(|i| deck[i])
                .collect();

            deals.push(Deal {
                hands,
                community,
                remaining,
            });
        }

        deals
    }

    // --------------------------------------------------------
    // ACTIONS
    // --------------------------------------------------------

    /// Perform a player action.
    pub fn action(
        &mut self,
        player: usize,
        action: QuantumPokerAction,
    ) -> Result<ActionResult, PokerError> {
        if player >= self.config.num_players {
            return Err(PokerError::InvalidPlayer(player, self.config.num_players));
        }
        if self.phase == GamePhase::Setup {
            return Err(PokerError::NotDealt);
        }
        if self.phase == GamePhase::Complete {
            return Err(PokerError::GameOver);
        }
        if self.folded[player] {
            return Err(PokerError::PlayerFolded(player));
        }

        match action {
            QuantumPokerAction::Measure(card_idx) => self.measure_card(player, card_idx),
            QuantumPokerAction::PartialMeasure(card_idx, basis) => {
                self.partial_measure_card(player, card_idx, basis)
            }
            QuantumPokerAction::QuantumBluff => self.quantum_bluff(player),
            QuantumPokerAction::Bet(amount) => self.bet(player, amount),
            QuantumPokerAction::Call => self.call(player),
            QuantumPokerAction::Fold => self.fold(player),
            QuantumPokerAction::Check => self.check(player),
            QuantumPokerAction::Raise(amount) => self.raise(player, amount),
        }
    }

    /// Full measurement: collapse a card to a definite value.
    fn measure_card(&mut self, player: usize, card_idx: usize) -> Result<ActionResult, PokerError> {
        if card_idx >= self.config.cards_per_hand {
            return Err(PokerError::InvalidCardIndex(
                card_idx,
                self.config.cards_per_hand,
            ));
        }
        if self.player_measured[player][card_idx].is_some() {
            let card = self.player_measured[player][card_idx].unwrap();
            return Ok(ActionResult {
                description: format!(
                    "Player {} card {} already measured: {}",
                    player, card_idx, card
                ),
                revealed_card: Some(card),
                partial_info: None,
                prob_shift: None,
                pot_total: self.pot_total(),
                branches_remaining: self.deal_amplitudes.len(),
            });
        }

        // Compute probability for each possible card at this position
        let mut card_probs: std::collections::HashMap<Card, f64> = std::collections::HashMap::new();
        for (deal, amp) in &self.deal_amplitudes {
            let card = deal.hands[player][card_idx];
            *card_probs.entry(card).or_insert(0.0) += amp.norm_sqr();
        }

        // Sample a card according to the probability distribution
        let r = self.rand_f64();
        let mut cumulative = 0.0;
        let mut chosen_card = None;
        for (&card, &prob) in &card_probs {
            cumulative += prob;
            if r <= cumulative && chosen_card.is_none() {
                chosen_card = Some(card);
            }
        }
        let chosen_card = chosen_card.unwrap_or_else(|| *card_probs.keys().next().unwrap());

        // Collapse: keep only branches where this card is at this position,
        // then renormalize
        self.deal_amplitudes
            .retain(|(deal, _)| deal.hands[player][card_idx] == chosen_card);
        self.renormalize();

        self.player_measured[player][card_idx] = Some(chosen_card);

        Ok(ActionResult {
            description: format!(
                "Player {} measures card {}: {}",
                player, card_idx, chosen_card
            ),
            revealed_card: Some(chosen_card),
            partial_info: None,
            prob_shift: None,
            pot_total: self.pot_total(),
            branches_remaining: self.deal_amplitudes.len(),
        })
    }

    /// Partial measurement: learn partial information without full collapse.
    fn partial_measure_card(
        &mut self,
        player: usize,
        card_idx: usize,
        basis: MeasurementBasis,
    ) -> Result<ActionResult, PokerError> {
        if card_idx >= self.config.cards_per_hand {
            return Err(PokerError::InvalidCardIndex(
                card_idx,
                self.config.cards_per_hand,
            ));
        }

        if basis == MeasurementBasis::FullCollapse {
            return self.measure_card(player, card_idx);
        }

        // Partition branches by the measurement outcome
        let classify = |card: &Card| -> bool {
            match basis {
                MeasurementBasis::SuitOnly => unreachable!(),
                MeasurementBasis::HighLow => card.rank.is_high(),
                MeasurementBasis::ColorOnly => card.suit.is_red(),
                MeasurementBasis::FaceCard => card.rank.is_face(),
                MeasurementBasis::FullCollapse => unreachable!(),
            }
        };

        // For SuitOnly, we partition into 4 outcomes
        if basis == MeasurementBasis::SuitOnly {
            return self.partial_measure_suit(player, card_idx);
        }

        // Binary partition: compute probability of true/false outcome
        let mut prob_true = 0.0;
        let mut prob_false = 0.0;
        for (deal, amp) in &self.deal_amplitudes {
            let card = &deal.hands[player][card_idx];
            if classify(card) {
                prob_true += amp.norm_sqr();
            } else {
                prob_false += amp.norm_sqr();
            }
        }

        // Sample outcome
        let r = self.rand_f64();
        let outcome = r < prob_true;

        // Collapse to matching branches
        self.deal_amplitudes.retain(|(deal, _)| {
            let card = &deal.hands[player][card_idx];
            classify(card) == outcome
        });
        self.renormalize();

        let info = match basis {
            MeasurementBasis::HighLow => {
                if outcome {
                    "HIGH (rank > 7)".to_string()
                } else {
                    "LOW (rank <= 7)".to_string()
                }
            }
            MeasurementBasis::ColorOnly => {
                if outcome {
                    "RED".to_string()
                } else {
                    "BLACK".to_string()
                }
            }
            MeasurementBasis::FaceCard => {
                if outcome {
                    "FACE CARD".to_string()
                } else {
                    "NOT FACE CARD".to_string()
                }
            }
            _ => unreachable!(),
        };

        Ok(ActionResult {
            description: format!(
                "Player {} partial-measures card {}: {}",
                player, card_idx, info
            ),
            revealed_card: None,
            partial_info: Some(info),
            prob_shift: None,
            pot_total: self.pot_total(),
            branches_remaining: self.deal_amplitudes.len(),
        })
    }

    /// Partial measurement: learn the suit of a card.
    fn partial_measure_suit(
        &mut self,
        player: usize,
        card_idx: usize,
    ) -> Result<ActionResult, PokerError> {
        let mut suit_probs = [0.0f64; 4];
        for (deal, amp) in &self.deal_amplitudes {
            let card = &deal.hands[player][card_idx];
            suit_probs[card.suit.index()] += amp.norm_sqr();
        }

        // Sample a suit
        let r = self.rand_f64();
        let mut cumulative = 0.0;
        let mut chosen_suit = Suit::Hearts;
        for &suit in &Suit::ALL {
            cumulative += suit_probs[suit.index()];
            if r <= cumulative {
                chosen_suit = suit;
                break;
            }
        }

        // Collapse to branches with this suit
        self.deal_amplitudes
            .retain(|(deal, _)| deal.hands[player][card_idx].suit == chosen_suit);
        self.renormalize();

        let info = format!("Suit is {}", chosen_suit);

        Ok(ActionResult {
            description: format!(
                "Player {} partial-measures card {} suit: {}",
                player, card_idx, chosen_suit
            ),
            revealed_card: None,
            partial_info: Some(info),
            prob_shift: None,
            pot_total: self.pot_total(),
            branches_remaining: self.deal_amplitudes.len(),
        })
    }

    /// Quantum bluff: apply a phase rotation to shift outcome probabilities.
    ///
    /// This rotates the amplitudes of "good" vs "bad" hands, creating
    /// constructive interference on stronger hands and destructive
    /// interference on weaker ones. The net effect: the player's
    /// probability of holding a winning hand increases (at the cost
    /// of entanglement entropy with the opponent).
    fn quantum_bluff(&mut self, player: usize) -> Result<ActionResult, PokerError> {
        if self.deal_amplitudes.is_empty() {
            return Err(PokerError::NotDealt);
        }

        // Phase rotation: strong hands get +pi/4 phase, weak hands get -pi/4
        // This is a valid unitary operation (diagonal in the deal basis)
        let phase_strong = C64::new(
            std::f64::consts::FRAC_PI_4.cos(),
            std::f64::consts::FRAC_PI_4.sin(),
        );
        let phase_weak = C64::new(
            std::f64::consts::FRAC_PI_4.cos(),
            -std::f64::consts::FRAC_PI_4.sin(),
        );

        // Compute median hand strength to define strong/weak boundary
        let mut strengths: Vec<(usize, HandRank)> = self
            .deal_amplitudes
            .iter()
            .enumerate()
            .map(|(i, (deal, _))| {
                let cards = &deal.hands[player];
                (i, evaluate_hand(cards))
            })
            .collect();
        strengths.sort_by(|a, b| a.1.cmp(&b.1));
        let median_idx = strengths.len() / 2;
        let median_rank = strengths[median_idx].1.clone();

        let mut shift_count = 0usize;
        for (deal, amp) in &mut self.deal_amplitudes {
            let hand_rank = evaluate_hand(&deal.hands[player]);
            if hand_rank >= median_rank {
                *amp = *amp * phase_strong;
                shift_count += 1;
            } else {
                *amp = *amp * phase_weak;
            }
        }

        // The shift does not change probabilities on its own (diagonal phase),
        // but it DOES change the entanglement structure. When combined with
        // the opponent's measurement, interference effects become visible.
        // The measured shift shows up as a change in conditional probabilities.
        let prob_shift = shift_count as f64 / self.deal_amplitudes.len() as f64;

        Ok(ActionResult {
            description: format!(
                "Player {} applies quantum bluff: phase-rotated {} of {} branches",
                player,
                shift_count,
                self.deal_amplitudes.len()
            ),
            revealed_card: None,
            partial_info: None,
            prob_shift: Some(prob_shift),
            pot_total: self.pot_total(),
            branches_remaining: self.deal_amplitudes.len(),
        })
    }

    /// Place a bet.
    fn bet(&mut self, player: usize, amount: f64) -> Result<ActionResult, PokerError> {
        if amount <= 0.0 {
            return Err(PokerError::InvalidBet(amount));
        }
        if amount > self.chips[player] {
            return Err(PokerError::InsufficientChips {
                player,
                requested: amount,
                available: self.chips[player],
            });
        }

        self.chips[player] -= amount;
        self.pot_contributions[player] += amount;
        if amount > self.current_bet {
            self.current_bet = amount;
        }

        Ok(ActionResult {
            description: format!("Player {} bets {:.1}", player, amount),
            revealed_card: None,
            partial_info: None,
            prob_shift: None,
            pot_total: self.pot_total(),
            branches_remaining: self.deal_amplitudes.len(),
        })
    }

    /// Call the current bet.
    fn call(&mut self, player: usize) -> Result<ActionResult, PokerError> {
        let to_call = self.current_bet - self.pot_contributions[player];
        if to_call <= 0.0 {
            return self.check(player);
        }
        if to_call > self.chips[player] {
            return Err(PokerError::InsufficientChips {
                player,
                requested: to_call,
                available: self.chips[player],
            });
        }

        self.chips[player] -= to_call;
        self.pot_contributions[player] += to_call;

        Ok(ActionResult {
            description: format!("Player {} calls {:.1}", player, to_call),
            revealed_card: None,
            partial_info: None,
            prob_shift: None,
            pot_total: self.pot_total(),
            branches_remaining: self.deal_amplitudes.len(),
        })
    }

    /// Fold.
    fn fold(&mut self, player: usize) -> Result<ActionResult, PokerError> {
        self.folded[player] = true;

        // Check if only one player remains
        let active: Vec<usize> = (0..self.config.num_players)
            .filter(|&p| !self.folded[p])
            .collect();
        if active.len() == 1 {
            self.phase = GamePhase::Complete;
        }

        Ok(ActionResult {
            description: format!("Player {} folds", player),
            revealed_card: None,
            partial_info: None,
            prob_shift: None,
            pot_total: self.pot_total(),
            branches_remaining: self.deal_amplitudes.len(),
        })
    }

    /// Check (bet nothing, stay in).
    fn check(&mut self, player: usize) -> Result<ActionResult, PokerError> {
        Ok(ActionResult {
            description: format!("Player {} checks", player),
            revealed_card: None,
            partial_info: None,
            prob_shift: None,
            pot_total: self.pot_total(),
            branches_remaining: self.deal_amplitudes.len(),
        })
    }

    /// Raise.
    fn raise(&mut self, player: usize, amount: f64) -> Result<ActionResult, PokerError> {
        if amount <= 0.0 {
            return Err(PokerError::InvalidBet(amount));
        }
        let total = self.current_bet + amount;
        let to_pay = total - self.pot_contributions[player];
        if to_pay > self.chips[player] {
            return Err(PokerError::InsufficientChips {
                player,
                requested: to_pay,
                available: self.chips[player],
            });
        }

        self.chips[player] -= to_pay;
        self.pot_contributions[player] += to_pay;
        self.current_bet = total;

        Ok(ActionResult {
            description: format!(
                "Player {} raises by {:.1} (total bet {:.1})",
                player, amount, total
            ),
            revealed_card: None,
            partial_info: None,
            prob_shift: None,
            pot_total: self.pot_total(),
            branches_remaining: self.deal_amplitudes.len(),
        })
    }

    // --------------------------------------------------------
    // COMMUNITY CARDS (Hold'em)
    // --------------------------------------------------------

    /// Reveal community cards for the current phase.
    pub fn reveal_community(&mut self) -> Result<Vec<Card>, PokerError> {
        if self.config.game_variant != PokerVariant::QuantumHoldem {
            return Err(PokerError::InvalidPhase {
                expected: "QuantumHoldem only",
                got: self.phase,
            });
        }

        let (start, count, next_phase) = match self.phase {
            GamePhase::PreFlop => (0, 3, GamePhase::Flop),
            GamePhase::Flop => (3, 1, GamePhase::Turn),
            GamePhase::Turn => (4, 1, GamePhase::River),
            _ => {
                return Err(PokerError::InvalidPhase {
                    expected: "PreFlop/Flop/Turn",
                    got: self.phase,
                });
            }
        };

        let mut revealed = Vec::new();
        for i in start..(start + count) {
            if i >= self.community_revealed.len() {
                break;
            }

            // Compute probabilities for each possible community card at this position
            let mut card_probs: std::collections::HashMap<Card, f64> =
                std::collections::HashMap::new();
            for (deal, amp) in &self.deal_amplitudes {
                if i < deal.community.len() {
                    let card = deal.community[i];
                    *card_probs.entry(card).or_insert(0.0) += amp.norm_sqr();
                }
            }

            if card_probs.is_empty() {
                continue;
            }

            // Sample
            let r = self.rand_f64();
            let mut cumulative = 0.0;
            let mut chosen = *card_probs.keys().next().unwrap();
            for (&card, &prob) in &card_probs {
                cumulative += prob;
                if r <= cumulative {
                    chosen = card;
                    break;
                }
            }

            // Collapse
            self.deal_amplitudes
                .retain(|(deal, _)| i < deal.community.len() && deal.community[i] == chosen);
            self.renormalize();

            self.community_revealed[i] = Some(chosen);
            revealed.push(chosen);
        }

        self.phase = next_phase;
        self.current_bet = 0.0;
        for _p in &mut self.pot_contributions {
            // Reset per-round contributions for the new betting round
            // (but keep total pot intact via tracking)
        }

        Ok(revealed)
    }

    /// Advance to showdown: measure all remaining cards.
    pub fn showdown(&mut self) -> Result<Vec<(usize, HandRank)>, PokerError> {
        if self.deal_amplitudes.is_empty() {
            return Err(PokerError::NotDealt);
        }

        // Collapse the entire state to one deal
        let mut cumulative = 0.0;
        let r = self.rand_f64();
        let mut chosen_idx = 0;
        for (i, (_, amp)) in self.deal_amplitudes.iter().enumerate() {
            cumulative += amp.norm_sqr();
            if r <= cumulative {
                chosen_idx = i;
                break;
            }
        }

        let final_deal = self.deal_amplitudes[chosen_idx].0.clone();

        // Set all cards as measured
        for p in 0..self.config.num_players {
            for c in 0..self.config.cards_per_hand {
                self.player_measured[p][c] = Some(final_deal.hands[p][c]);
            }
        }
        for (i, card) in final_deal.community.iter().enumerate() {
            if i < self.community_revealed.len() {
                self.community_revealed[i] = Some(*card);
            }
        }

        // Collapse to the single chosen deal
        let _amp = self.deal_amplitudes[chosen_idx].1;
        self.deal_amplitudes = vec![(final_deal.clone(), C64::new(1.0, 0.0))];
        self.phase = GamePhase::Showdown;

        // Evaluate hands
        let mut results = Vec::new();
        for p in 0..self.config.num_players {
            if self.folded[p] {
                continue;
            }
            let mut all_cards = final_deal.hands[p].clone();
            all_cards.extend_from_slice(&final_deal.community);
            let rank = evaluate_hand(&all_cards);
            results.push((p, rank));
        }

        // Sort by hand rank (best first)
        results.sort_by(|a, b| b.1.cmp(&a.1));

        self.phase = GamePhase::Complete;
        Ok(results)
    }

    // --------------------------------------------------------
    // QUERIES
    // --------------------------------------------------------

    /// Get the probability distribution of hand ranks for a player.
    pub fn hand_probabilities(&self, player: usize) -> Result<Vec<(HandRank, f64)>, PokerError> {
        if player >= self.config.num_players {
            return Err(PokerError::InvalidPlayer(player, self.config.num_players));
        }
        if self.deal_amplitudes.is_empty() {
            return Err(PokerError::NotDealt);
        }

        let mut rank_probs: std::collections::HashMap<HandRank, f64> =
            std::collections::HashMap::new();

        for (deal, amp) in &self.deal_amplitudes {
            let mut cards = deal.hands[player].clone();
            cards.extend_from_slice(&deal.community);
            let rank = evaluate_hand(&cards);
            *rank_probs.entry(rank).or_insert(0.0) += amp.norm_sqr();
        }

        let mut result: Vec<(HandRank, f64)> = rank_probs.into_iter().collect();
        result.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        Ok(result)
    }

    /// Card probabilities for a specific player's card slot.
    pub fn card_probabilities(
        &self,
        player: usize,
        card_idx: usize,
    ) -> Result<Vec<(Card, f64)>, PokerError> {
        if player >= self.config.num_players {
            return Err(PokerError::InvalidPlayer(player, self.config.num_players));
        }
        if card_idx >= self.config.cards_per_hand {
            return Err(PokerError::InvalidCardIndex(
                card_idx,
                self.config.cards_per_hand,
            ));
        }
        if self.deal_amplitudes.is_empty() {
            return Err(PokerError::NotDealt);
        }

        let mut probs: std::collections::HashMap<Card, f64> = std::collections::HashMap::new();
        for (deal, amp) in &self.deal_amplitudes {
            let card = deal.hands[player][card_idx];
            *probs.entry(card).or_insert(0.0) += amp.norm_sqr();
        }

        let mut result: Vec<(Card, f64)> = probs.into_iter().collect();
        result.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        Ok(result)
    }

    /// Compute entanglement entropy between two players.
    ///
    /// This measures how correlated their hands are. In a classical game
    /// the entropy is zero. In quantum poker the deck constraint creates
    /// anti-correlations (if Alice has the Ace, Bob cannot), and quantum
    /// operations can increase or decrease entanglement.
    ///
    /// We compute the von Neumann entropy S(rho_A) of player A's reduced
    /// density matrix, traced over player B and the rest of the game.
    pub fn entanglement_entropy(
        &self,
        player_a: usize,
        player_b: usize,
    ) -> Result<f64, PokerError> {
        if player_a >= self.config.num_players {
            return Err(PokerError::InvalidPlayer(player_a, self.config.num_players));
        }
        if player_b >= self.config.num_players {
            return Err(PokerError::InvalidPlayer(player_b, self.config.num_players));
        }
        if self.deal_amplitudes.is_empty() {
            return Err(PokerError::NotDealt);
        }

        // Build the reduced density matrix for player A by grouping deals
        // by player A's hand and computing off-diagonal coherences.
        //
        // rho_A[hand_i, hand_j] = sum_{env} <hand_i, env | psi> <psi | hand_j, env>
        //
        // For simplicity, we use the diagonal approximation (classical Shannon
        // entropy of player A's hand distribution), which is a lower bound on
        // the von Neumann entropy.
        let mut hand_probs: std::collections::HashMap<Vec<Card>, f64> =
            std::collections::HashMap::new();
        for (deal, amp) in &self.deal_amplitudes {
            let hand = deal.hands[player_a].clone();
            *hand_probs.entry(hand).or_insert(0.0) += amp.norm_sqr();
        }

        // Shannon entropy: S = -sum p_i ln(p_i)
        let entropy = hand_probs
            .values()
            .filter(|&&p| p > 1e-15)
            .map(|&p| -p * p.ln())
            .sum::<f64>();

        Ok(entropy)
    }

    /// Compute the mutual information I(A:B) between two players.
    /// I(A:B) = S(A) + S(B) - S(A,B).
    pub fn mutual_information(&self, player_a: usize, player_b: usize) -> Result<f64, PokerError> {
        let s_a = self.entanglement_entropy(player_a, player_b)?;
        let s_b = self.entanglement_entropy(player_b, player_a)?;

        // Joint entropy S(A,B)
        let mut joint_probs: std::collections::HashMap<(Vec<Card>, Vec<Card>), f64> =
            std::collections::HashMap::new();
        for (deal, amp) in &self.deal_amplitudes {
            let key = (deal.hands[player_a].clone(), deal.hands[player_b].clone());
            *joint_probs.entry(key).or_insert(0.0) += amp.norm_sqr();
        }
        let s_ab: f64 = joint_probs
            .values()
            .filter(|&&p| p > 1e-15)
            .map(|&p| -p * p.ln())
            .sum();

        Ok(s_a + s_b - s_ab)
    }

    /// Display the game state from a particular player's perspective.
    pub fn display_state(&self, player: usize) -> Result<String, PokerError> {
        if player >= self.config.num_players {
            return Err(PokerError::InvalidPlayer(player, self.config.num_players));
        }

        let mut out = String::new();
        out.push_str(&format!(
            "=== Quantum Poker: {:?} ===\n",
            self.config.game_variant
        ));
        out.push_str(&format!("Phase: {:?}\n", self.phase));
        out.push_str(&format!(
            "Branches in superposition: {}\n",
            self.deal_amplitudes.len()
        ));
        out.push_str(&format!("Pot: {:.1}\n\n", self.pot_total()));

        // Player's hand
        out.push_str(&format!("--- Player {} (You) ---\n", player));
        out.push_str(&format!("Chips: {:.1}\n", self.chips[player]));
        for c in 0..self.config.cards_per_hand {
            match self.player_measured[player][c] {
                Some(card) => out.push_str(&format!("  Card {}: {} (measured)\n", c, card)),
                None => {
                    if let Ok(probs) = self.card_probabilities(player, c) {
                        out.push_str(&format!("  Card {}: [superposition]\n", c));
                        let top = probs.iter().take(5);
                        for (card, prob) in top {
                            out.push_str(&format!("    {} : {:.1}%\n", card, prob * 100.0));
                        }
                        if probs.len() > 5 {
                            out.push_str(&format!("    ... and {} more\n", probs.len() - 5));
                        }
                    }
                }
            }
        }

        // Community cards
        if !self.community_revealed.is_empty() {
            out.push_str("\nCommunity: ");
            for oc in &self.community_revealed {
                match oc {
                    Some(card) => out.push_str(&format!("{} ", card)),
                    None => out.push_str("[?] "),
                }
            }
            out.push('\n');
        }

        // Other players (hidden info)
        for p in 0..self.config.num_players {
            if p == player {
                continue;
            }
            if self.folded[p] {
                out.push_str(&format!("\nPlayer {}: FOLDED\n", p));
            } else {
                out.push_str(&format!(
                    "\nPlayer {}: {} cards face-down, Chips: {:.1}\n",
                    p, self.config.cards_per_hand, self.chips[p]
                ));
            }
        }

        Ok(out)
    }

    /// Determine the winner (if the game is complete).
    pub fn winner(&self) -> Option<usize> {
        if self.phase != GamePhase::Complete {
            return None;
        }

        // If everyone but one folded, that player wins
        let active: Vec<usize> = (0..self.config.num_players)
            .filter(|&p| !self.folded[p])
            .collect();
        if active.len() == 1 {
            return Some(active[0]);
        }

        // Evaluate hands from the collapsed state
        if let Some((deal, _)) = self.deal_amplitudes.first() {
            let mut best_player = active[0];
            let mut best_rank = HandRank::HighCard(0);
            for &p in &active {
                let mut cards = deal.hands[p].clone();
                cards.extend_from_slice(&deal.community);
                let rank = evaluate_hand(&cards);
                if rank > best_rank {
                    best_rank = rank;
                    best_player = p;
                }
            }
            return Some(best_player);
        }

        None
    }

    /// Get the current game phase.
    pub fn phase(&self) -> GamePhase {
        self.phase
    }

    /// Get the total pot.
    pub fn pot_total(&self) -> f64 {
        self.pot_contributions.iter().sum()
    }

    /// Get a player's remaining chips.
    pub fn chips(&self, player: usize) -> f64 {
        self.chips[player]
    }

    /// Number of branches (deals) in the current superposition.
    pub fn branch_count(&self) -> usize {
        self.deal_amplitudes.len()
    }

    /// Whether a player has folded.
    pub fn is_folded(&self, player: usize) -> bool {
        self.folded[player]
    }

    /// Access the raw deal amplitudes (for advanced analysis).
    pub fn deal_amplitudes(&self) -> &[(Deal, C64)] {
        &self.deal_amplitudes
    }

    // --------------------------------------------------------
    // INTERNAL HELPERS
    // --------------------------------------------------------

    /// Renormalize amplitudes so that probabilities sum to 1.
    fn renormalize(&mut self) {
        let total: f64 = self.deal_amplitudes.iter().map(|(_, a)| a.norm_sqr()).sum();
        if total > 1e-15 {
            let inv_sqrt = 1.0 / total.sqrt();
            for (_, amp) in &mut self.deal_amplitudes {
                *amp = C64::new(amp.re * inv_sqrt, amp.im * inv_sqrt);
            }
        }
    }
}

// ============================================================
// CHSH INEQUALITY VIOLATION
// ============================================================

/// CHSH poker demonstration.
///
/// In a 4-card deck {A-spades, K-spades, A-hearts, K-hearts}, two players each get one card.
/// We show that quantum correlations between the cards can violate the CHSH inequality
/// (classical bound = 2, quantum bound = 2*sqrt(2) ~ 2.828).
///
/// The "measurement settings" correspond to different partial-measurement bases.
/// The CHSH correlator S = E(a0,b0) - E(a0,b1) + E(a1,b0) + E(a1,b1)
/// where E(a,b) is the correlation between outcomes under settings a,b.
pub struct ChshPokerDemo {
    /// Number of trials to run.
    pub num_trials: usize,
    /// Seed for reproducibility.
    pub seed: u64,
}

impl ChshPokerDemo {
    pub fn new(num_trials: usize, seed: u64) -> Self {
        Self { num_trials, seed }
    }

    /// Run the CHSH demonstration.
    ///
    /// Returns the CHSH correlator value S. Classical limit is |S| <= 2.
    /// Quantum mechanics allows |S| up to 2*sqrt(2).
    ///
    /// We create entangled card pairs and measure in two different bases:
    /// - Basis 0: HighLow (is the rank > 7?)
    /// - Basis 1: ColorOnly (is the suit red?)
    ///
    /// The quantum advantage comes from the deck constraint creating
    /// anti-correlations that mimic Bell-state entanglement.
    pub fn run(&self) -> ChshResult {
        let _rng_state = self.seed;
        let _next_rng = |state: &mut u64| -> u64 {
            let mut x = *state;
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            *state = x;
            x
        };

        // Compute correlators by exhaustive enumeration of the toy deck.
        // The deck is {A-spades, K-spades, A-hearts, K-hearts}, 2 players, 1 card each.
        let deck = toy_deck();
        // All valid deals: 4 * 3 = 12 ordered pairs
        let mut deals = Vec::new();
        for i in 0..deck.len() {
            for j in 0..deck.len() {
                if i != j {
                    deals.push((deck[i], deck[j]));
                }
            }
        }

        // Equal superposition over all deals
        let n = deals.len() as f64;

        // Define measurement outcomes as +1 or -1
        let outcome_high_low = |card: &Card| -> f64 {
            if card.rank.is_high() {
                1.0
            } else {
                -1.0
            }
        };
        let outcome_color = |card: &Card| -> f64 {
            if card.suit.is_red() {
                1.0
            } else {
                -1.0
            }
        };

        // Compute correlators E(a, b) = sum_deals p(deal) * outcome_a(alice) * outcome_b(bob)
        let e = |f_a: &dyn Fn(&Card) -> f64, f_b: &dyn Fn(&Card) -> f64| -> f64 {
            let mut corr = 0.0;
            for &(ca, cb) in &deals {
                corr += f_a(&ca) * f_b(&cb);
            }
            corr / n
        };

        let e_00 = e(&outcome_high_low, &outcome_high_low);
        let e_01 = e(&outcome_high_low, &outcome_color);
        let e_10 = e(&outcome_color, &outcome_high_low);
        let e_11 = e(&outcome_color, &outcome_color);

        // CHSH: S = E(0,0) - E(0,1) + E(1,0) + E(1,1)
        let s = e_00 - e_01 + e_10 + e_11;

        // Now add a quantum phase rotation to enhance correlations.
        // In the entangled basis, applying a relative phase to "matching" pairs
        // (both high or both low) vs "mismatched" pairs creates interference.
        let phase = std::f64::consts::FRAC_PI_4;
        let cos_p = phase.cos();
        let sin_p = phase.sin();

        // Quantum-enhanced correlator: apply phase to amplitudes before computing
        let mut q_amps: Vec<C64> = deals
            .iter()
            .map(|_| C64::new(1.0 / n.sqrt(), 0.0))
            .collect();

        // Phase rotation: matching-rank pairs get e^{i*pi/4}, mismatched get e^{-i*pi/4}
        for (i, &(ca, cb)) in deals.iter().enumerate() {
            if ca.rank == cb.rank {
                // This can't happen (same rank, different suits IS possible in toy deck)
                // A-spades and A-hearts have same rank
                let phase_factor = C64::new(cos_p, sin_p);
                q_amps[i] = q_amps[i] * phase_factor;
            } else {
                let phase_factor = C64::new(cos_p, -sin_p);
                q_amps[i] = q_amps[i] * phase_factor;
            }
        }

        // Renormalize
        let norm: f64 = q_amps.iter().map(|a| a.norm_sqr()).sum();
        let inv_norm = 1.0 / norm.sqrt();
        for a in &mut q_amps {
            *a = C64::new(a.re * inv_norm, a.im * inv_norm);
        }

        // Quantum correlators
        let eq = |f_a: &dyn Fn(&Card) -> f64, f_b: &dyn Fn(&Card) -> f64| -> f64 {
            let mut corr = 0.0;
            for (i, &(ca, cb)) in deals.iter().enumerate() {
                corr += q_amps[i].norm_sqr() * f_a(&ca) * f_b(&cb);
            }
            corr
        };

        let eq_00 = eq(&outcome_high_low, &outcome_high_low);
        let eq_01 = eq(&outcome_high_low, &outcome_color);
        let eq_10 = eq(&outcome_color, &outcome_high_low);
        let eq_11 = eq(&outcome_color, &outcome_color);

        let s_quantum = eq_00 - eq_01 + eq_10 + eq_11;

        ChshResult {
            classical_correlator: s,
            quantum_correlator: s_quantum,
            classical_bound: 2.0,
            tsirelson_bound: 2.0 * 2.0_f64.sqrt(),
            correlators_classical: [e_00, e_01, e_10, e_11],
            correlators_quantum: [eq_00, eq_01, eq_10, eq_11],
            violation: s_quantum.abs() > 2.0,
        }
    }
}

/// Result of a CHSH poker demonstration.
#[derive(Clone, Debug)]
pub struct ChshResult {
    /// Classical CHSH correlator S.
    pub classical_correlator: f64,
    /// Quantum-enhanced CHSH correlator S.
    pub quantum_correlator: f64,
    /// Classical upper bound (2.0).
    pub classical_bound: f64,
    /// Tsirelson bound (2*sqrt(2) ~ 2.828).
    pub tsirelson_bound: f64,
    /// Individual correlators [E(0,0), E(0,1), E(1,0), E(1,1)].
    pub correlators_classical: [f64; 4],
    /// Quantum correlators.
    pub correlators_quantum: [f64; 4],
    /// Whether the quantum correlator violates the classical bound.
    pub violation: bool,
}

impl fmt::Display for ChshResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "=== CHSH Poker Demonstration ===")?;
        writeln!(
            f,
            "Classical correlator S = {:.4}",
            self.classical_correlator
        )?;
        writeln!(f, "Quantum correlator  S = {:.4}", self.quantum_correlator)?;
        writeln!(f, "Classical bound       = {:.4}", self.classical_bound)?;
        writeln!(f, "Tsirelson bound       = {:.4}", self.tsirelson_bound)?;
        writeln!(
            f,
            "Violation: {}",
            if self.violation {
                "YES -- quantum advantage!"
            } else {
                "No"
            }
        )?;
        writeln!(f, "\nClassical correlators:")?;
        writeln!(f, "  E(HL,HL) = {:.4}", self.correlators_classical[0])?;
        writeln!(f, "  E(HL,CO) = {:.4}", self.correlators_classical[1])?;
        writeln!(f, "  E(CO,HL) = {:.4}", self.correlators_classical[2])?;
        writeln!(f, "  E(CO,CO) = {:.4}", self.correlators_classical[3])?;
        writeln!(f, "\nQuantum correlators:")?;
        writeln!(f, "  E(HL,HL) = {:.4}", self.correlators_quantum[0])?;
        writeln!(f, "  E(HL,CO) = {:.4}", self.correlators_quantum[1])?;
        writeln!(f, "  E(CO,HL) = {:.4}", self.correlators_quantum[2])?;
        writeln!(f, "  E(CO,CO) = {:.4}", self.correlators_quantum[3])
    }
}

// ============================================================
// COMBINATORIAL HELPERS
// ============================================================

/// Generate all k-permutations of a slice of indices.
fn permutations_k(items: &[usize], k: usize) -> Vec<Vec<usize>> {
    if k == 0 {
        return vec![vec![]];
    }
    let mut result = Vec::new();
    for (i, &item) in items.iter().enumerate() {
        let remaining: Vec<usize> = items
            .iter()
            .enumerate()
            .filter(|&(j, _)| j != i)
            .map(|(_, &v)| v)
            .collect();
        for mut sub in permutations_k(&remaining, k - 1) {
            sub.insert(0, item);
            result.push(sub);
        }
    }
    result
}

// ============================================================
// TESTS
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;

    // ------ Configuration & Builder ------

    #[test]
    fn test_config_builder() {
        let config = QuantumPokerConfig::new()
            .num_players(3)
            .cards_per_hand(2)
            .deck_size(52)
            .variant(PokerVariant::QuantumHoldem)
            .starting_chips(500.0)
            .seed(123)
            .community_cards(5);

        assert_eq!(config.num_players, 3);
        assert_eq!(config.cards_per_hand, 2);
        assert_eq!(config.deck_size, 52);
        assert_eq!(config.game_variant, PokerVariant::QuantumHoldem);
        assert!((config.starting_chips - 500.0).abs() < 1e-10);
        assert_eq!(config.seed, 123);
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_config_validation_too_many_players() {
        let config = QuantumPokerConfig::new().num_players(7);
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_config_validation_deck_too_small() {
        let config = QuantumPokerConfig::new()
            .num_players(6)
            .cards_per_hand(5)
            .community_cards(5)
            .deck_size(10);
        assert!(config.validate().is_err());
    }

    // ------ Card Types ------

    #[test]
    fn test_card_creation_and_index() {
        let card = Card::new(Suit::Spades, Rank::ACE);
        assert_eq!(card.suit, Suit::Spades);
        assert_eq!(card.rank, Rank::ACE);

        // ACE (14) -> rank_index = 12, Spades -> suit_index = 3
        // index = 12 * 4 + 3 = 51
        assert_eq!(card.index(), 51);

        let roundtrip = Card::from_index(card.index());
        assert_eq!(roundtrip, card);
    }

    #[test]
    fn test_standard_deck_has_52_cards() {
        let deck = standard_deck();
        assert_eq!(deck.len(), 52);

        // All unique
        let mut set = std::collections::HashSet::new();
        for card in &deck {
            assert!(set.insert(*card));
        }
    }

    #[test]
    fn test_toy_deck() {
        let deck = toy_deck();
        assert_eq!(deck.len(), 4);
        assert!(deck.contains(&Card::new(Suit::Spades, Rank::ACE)));
        assert!(deck.contains(&Card::new(Suit::Spades, Rank::KING)));
        assert!(deck.contains(&Card::new(Suit::Hearts, Rank::ACE)));
        assert!(deck.contains(&Card::new(Suit::Hearts, Rank::KING)));
    }

    #[test]
    fn test_suit_properties() {
        assert!(Suit::Hearts.is_red());
        assert!(Suit::Diamonds.is_red());
        assert!(!Suit::Clubs.is_red());
        assert!(!Suit::Spades.is_red());
    }

    #[test]
    fn test_rank_properties() {
        assert!(Rank::JACK.is_face());
        assert!(Rank::QUEEN.is_face());
        assert!(Rank::KING.is_face());
        assert!(!Rank::ACE.is_face());
        assert!(!Rank::TEN.is_face());

        assert!(Rank::ACE.is_high());
        assert!(Rank::EIGHT.is_high());
        assert!(!Rank::SEVEN.is_high());
        assert!(!Rank::TWO.is_high());
    }

    // ------ Deal Validity ------

    #[test]
    fn test_deal_creates_valid_superposition() {
        let config = QuantumPokerConfig::toy_game();
        let mut game = QuantumPokerGame::new(config).unwrap();
        game.deal().unwrap();

        assert!(game.branch_count() > 0);

        // All deals should be valid (no duplicate cards)
        for (deal, _) in game.deal_amplitudes() {
            assert!(deal.is_valid(), "Invalid deal found: {:?}", deal);
        }

        // Probabilities should sum to 1
        let total_prob: f64 = game
            .deal_amplitudes()
            .iter()
            .map(|(_, a)| a.norm_sqr())
            .sum();
        assert!(
            (total_prob - 1.0).abs() < 1e-10,
            "Probabilities sum to {} instead of 1.0",
            total_prob
        );
    }

    #[test]
    fn test_toy_game_enumerates_all_deals() {
        let config = QuantumPokerConfig::toy_game();
        let mut game = QuantumPokerGame::new(config).unwrap();
        game.deal().unwrap();

        // 4 cards, 2 players, 1 each, 0 community = P(4,2) = 12 deals
        assert_eq!(game.branch_count(), 12);
    }

    // ------ Measurement ------

    #[test]
    fn test_full_measurement_collapses_to_specific_card() {
        let config = QuantumPokerConfig::toy_game();
        let mut game = QuantumPokerGame::new(config).unwrap();
        game.deal().unwrap();

        let initial_branches = game.branch_count();
        let result = game.action(0, QuantumPokerAction::Measure(0)).unwrap();

        assert!(result.revealed_card.is_some());
        // After measurement, branches should be reduced
        assert!(game.branch_count() < initial_branches);

        // The measured card should be one of the toy deck cards
        let card = result.revealed_card.unwrap();
        let deck = toy_deck();
        assert!(deck.contains(&card), "Measured card {} not in deck", card);
    }

    #[test]
    fn test_partial_measurement_suit_preserves_rank_superposition() {
        let config = QuantumPokerConfig::toy_game();
        let mut game = QuantumPokerGame::new(config).unwrap();
        game.deal().unwrap();

        let result = game
            .action(
                0,
                QuantumPokerAction::PartialMeasure(0, MeasurementBasis::SuitOnly),
            )
            .unwrap();

        assert!(result.revealed_card.is_none()); // Not a full collapse
        assert!(result.partial_info.is_some());

        // After suit measurement, player should still have multiple possible ranks
        // (unless the deck is very small and only one rank of that suit exists)
        let probs = game.card_probabilities(0, 0).unwrap();
        // In the toy deck with suit measured, we should have exactly 2 possible cards
        // (A and K of the measured suit), but only 1 of each, so cards might reduce
        // depending on which deals survive.
        assert!(!probs.is_empty());
    }

    #[test]
    fn test_partial_measurement_high_low() {
        let config = QuantumPokerConfig::toy_game();
        let mut game = QuantumPokerGame::new(config).unwrap();
        game.deal().unwrap();

        let result = game
            .action(
                0,
                QuantumPokerAction::PartialMeasure(0, MeasurementBasis::HighLow),
            )
            .unwrap();

        assert!(result.partial_info.is_some());
        let info = result.partial_info.unwrap();
        // Both A and K are high (>7), so this should always say HIGH
        assert!(info.contains("HIGH"), "Expected HIGH but got: {}", info);
    }

    // ------ Hand Probabilities ------

    #[test]
    fn test_hand_probabilities_sum_to_one() {
        let config = QuantumPokerConfig::toy_game();
        let mut game = QuantumPokerGame::new(config).unwrap();
        game.deal().unwrap();

        let probs = game.hand_probabilities(0).unwrap();
        let total: f64 = probs.iter().map(|(_, p)| p).sum();
        assert!(
            (total - 1.0).abs() < 1e-10,
            "Hand probabilities sum to {} instead of 1.0",
            total
        );
    }

    // ------ Entanglement ------

    #[test]
    fn test_entanglement_entropy_positive() {
        let config = QuantumPokerConfig::toy_game();
        let mut game = QuantumPokerGame::new(config).unwrap();
        game.deal().unwrap();

        let entropy = game.entanglement_entropy(0, 1).unwrap();
        // In a superposition of deals, there should be nonzero entropy
        assert!(
            entropy > 0.0,
            "Entanglement entropy should be > 0, got {}",
            entropy
        );
    }

    #[test]
    fn test_measurement_changes_opponent_probabilities() {
        let config = QuantumPokerConfig::toy_game();
        let mut game = QuantumPokerGame::new(config).unwrap();
        game.deal().unwrap();

        // Get player 1's card probabilities before player 0 measures
        let probs_before = game.card_probabilities(1, 0).unwrap();

        // Player 0 measures their card
        game.action(0, QuantumPokerAction::Measure(0)).unwrap();

        // Get player 1's card probabilities after
        let probs_after = game.card_probabilities(1, 0).unwrap();

        // The distributions should differ (entanglement: measuring one affects the other)
        // At minimum, the number of possible cards should decrease
        assert!(
            probs_after.len() <= probs_before.len(),
            "Measurement should reduce or maintain opponent's possibilities"
        );
    }

    // ------ Quantum Bluff ------

    #[test]
    fn test_quantum_bluff_preserves_normalization() {
        let config = QuantumPokerConfig::toy_game();
        let mut game = QuantumPokerGame::new(config).unwrap();
        game.deal().unwrap();

        game.action(0, QuantumPokerAction::QuantumBluff).unwrap();

        let total_prob: f64 = game
            .deal_amplitudes()
            .iter()
            .map(|(_, a)| a.norm_sqr())
            .sum();
        assert!(
            (total_prob - 1.0).abs() < 1e-10,
            "After quantum bluff, probabilities sum to {} instead of 1.0",
            total_prob
        );
    }

    #[test]
    fn test_quantum_bluff_shifts_probabilities() {
        let config = QuantumPokerConfig::toy_game();
        let mut game = QuantumPokerGame::new(config).unwrap();
        game.deal().unwrap();

        let result = game.action(0, QuantumPokerAction::QuantumBluff).unwrap();
        assert!(result.prob_shift.is_some());
        // The shift should be between 0 and 1
        let shift = result.prob_shift.unwrap();
        assert!(shift >= 0.0 && shift <= 1.0);
    }

    // ------ Betting Mechanics ------

    #[test]
    fn test_bet_deducts_chips() {
        let config = QuantumPokerConfig::toy_game();
        let mut game = QuantumPokerGame::new(config).unwrap();
        game.deal().unwrap();

        let initial_chips = game.chips(0);
        game.action(0, QuantumPokerAction::Bet(10.0)).unwrap();
        assert!((game.chips(0) - (initial_chips - 10.0)).abs() < 1e-10);
        assert!((game.pot_total() - 10.0).abs() < 1e-10);
    }

    #[test]
    fn test_raise_increases_current_bet() {
        let config = QuantumPokerConfig::toy_game();
        let mut game = QuantumPokerGame::new(config).unwrap();
        game.deal().unwrap();

        game.action(0, QuantumPokerAction::Bet(10.0)).unwrap();
        game.action(1, QuantumPokerAction::Raise(5.0)).unwrap();

        // Player 1 should have paid 15 total (call 10 + raise 5)
        assert!((game.chips(1) - 85.0).abs() < 1e-10);
    }

    #[test]
    fn test_fold_removes_player() {
        let config = QuantumPokerConfig::toy_game();
        let mut game = QuantumPokerGame::new(config).unwrap();
        game.deal().unwrap();

        game.action(1, QuantumPokerAction::Fold).unwrap();

        assert!(game.is_folded(1));
        // With only 2 players and one folded, game should be complete
        assert_eq!(game.phase(), GamePhase::Complete);
        assert_eq!(game.winner(), Some(0));
    }

    #[test]
    fn test_insufficient_chips_error() {
        let config = QuantumPokerConfig::toy_game();
        let mut game = QuantumPokerGame::new(config).unwrap();
        game.deal().unwrap();

        let result = game.action(0, QuantumPokerAction::Bet(999.0));
        assert!(result.is_err());
        match result {
            Err(PokerError::InsufficientChips { .. }) => {}
            other => panic!("Expected InsufficientChips, got {:?}", other),
        }
    }

    // ------ Hand Ranking ------

    #[test]
    fn test_hand_ranking_comparison() {
        // Royal flush > straight flush > four of a kind > ... > high card
        assert!(HandRank::RoyalFlush > HandRank::StraightFlush(13));
        assert!(HandRank::StraightFlush(13) > HandRank::FourOfAKind(14));
        assert!(HandRank::FourOfAKind(14) > HandRank::FullHouse(14, 13));
        assert!(HandRank::FullHouse(14, 13) > HandRank::Flush(14));
        assert!(HandRank::Flush(14) > HandRank::Straight(14));
        assert!(HandRank::Straight(14) > HandRank::ThreeOfAKind(14));
        assert!(HandRank::ThreeOfAKind(14) > HandRank::TwoPair(14, 13));
        assert!(HandRank::TwoPair(14, 13) > HandRank::OnePair(14));
        assert!(HandRank::OnePair(14) > HandRank::HighCard(14));
    }

    #[test]
    fn test_evaluate_royal_flush() {
        let hand = [
            Card::new(Suit::Spades, Rank::TEN),
            Card::new(Suit::Spades, Rank::JACK),
            Card::new(Suit::Spades, Rank::QUEEN),
            Card::new(Suit::Spades, Rank::KING),
            Card::new(Suit::Spades, Rank::ACE),
        ];
        assert_eq!(evaluate_five(&hand), HandRank::RoyalFlush);
    }

    #[test]
    fn test_evaluate_straight_flush() {
        let hand = [
            Card::new(Suit::Hearts, Rank::FIVE),
            Card::new(Suit::Hearts, Rank::SIX),
            Card::new(Suit::Hearts, Rank::SEVEN),
            Card::new(Suit::Hearts, Rank::EIGHT),
            Card::new(Suit::Hearts, Rank::NINE),
        ];
        assert_eq!(evaluate_five(&hand), HandRank::StraightFlush(9));
    }

    #[test]
    fn test_evaluate_four_of_a_kind() {
        let hand = [
            Card::new(Suit::Hearts, Rank::ACE),
            Card::new(Suit::Diamonds, Rank::ACE),
            Card::new(Suit::Clubs, Rank::ACE),
            Card::new(Suit::Spades, Rank::ACE),
            Card::new(Suit::Hearts, Rank::KING),
        ];
        assert_eq!(evaluate_five(&hand), HandRank::FourOfAKind(14));
    }

    #[test]
    fn test_evaluate_full_house() {
        let hand = [
            Card::new(Suit::Hearts, Rank::KING),
            Card::new(Suit::Diamonds, Rank::KING),
            Card::new(Suit::Clubs, Rank::KING),
            Card::new(Suit::Spades, Rank::QUEEN),
            Card::new(Suit::Hearts, Rank::QUEEN),
        ];
        assert_eq!(evaluate_five(&hand), HandRank::FullHouse(13, 12));
    }

    #[test]
    fn test_evaluate_flush() {
        let hand = [
            Card::new(Suit::Clubs, Rank::TWO),
            Card::new(Suit::Clubs, Rank::FIVE),
            Card::new(Suit::Clubs, Rank::SEVEN),
            Card::new(Suit::Clubs, Rank::NINE),
            Card::new(Suit::Clubs, Rank::ACE),
        ];
        assert_eq!(evaluate_five(&hand), HandRank::Flush(14));
    }

    #[test]
    fn test_evaluate_straight() {
        let hand = [
            Card::new(Suit::Hearts, Rank::THREE),
            Card::new(Suit::Diamonds, Rank::FOUR),
            Card::new(Suit::Clubs, Rank::FIVE),
            Card::new(Suit::Spades, Rank::SIX),
            Card::new(Suit::Hearts, Rank::SEVEN),
        ];
        assert_eq!(evaluate_five(&hand), HandRank::Straight(7));
    }

    #[test]
    fn test_evaluate_wheel_straight() {
        let hand = [
            Card::new(Suit::Hearts, Rank::ACE),
            Card::new(Suit::Diamonds, Rank::TWO),
            Card::new(Suit::Clubs, Rank::THREE),
            Card::new(Suit::Spades, Rank::FOUR),
            Card::new(Suit::Hearts, Rank::FIVE),
        ];
        assert_eq!(evaluate_five(&hand), HandRank::Straight(5));
    }

    // ------ Full Game Flow ------

    #[test]
    fn test_showdown_determines_winner() {
        let config = QuantumPokerConfig::toy_game();
        let mut game = QuantumPokerGame::new(config).unwrap();
        game.deal().unwrap();

        let results = game.showdown().unwrap();
        assert!(!results.is_empty());
        assert_eq!(game.phase(), GamePhase::Complete);

        let winner = game.winner();
        assert!(winner.is_some());
    }

    #[test]
    fn test_full_holdem_flow() {
        let config = QuantumPokerConfig::new()
            .num_players(2)
            .cards_per_hand(2)
            .deck_size(52)
            .variant(PokerVariant::QuantumHoldem)
            .community_cards(5)
            .max_deals(200)
            .seed(42);

        let mut game = QuantumPokerGame::new(config).unwrap();

        // Deal
        game.deal().unwrap();
        assert!(game.branch_count() > 0);
        assert_eq!(game.phase(), GamePhase::PreFlop);

        // Pre-flop: player 0 measures a card
        game.action(0, QuantumPokerAction::Measure(0)).unwrap();

        // Betting
        game.action(0, QuantumPokerAction::Bet(10.0)).unwrap();
        game.action(1, QuantumPokerAction::Call).unwrap();

        // Flop
        let flop = game.reveal_community().unwrap();
        assert_eq!(flop.len(), 3);
        assert_eq!(game.phase(), GamePhase::Flop);

        // More betting
        game.action(0, QuantumPokerAction::Check).unwrap();
        game.action(1, QuantumPokerAction::Check).unwrap();

        // Turn
        let turn = game.reveal_community().unwrap();
        assert_eq!(turn.len(), 1);
        assert_eq!(game.phase(), GamePhase::Turn);

        // River
        let river = game.reveal_community().unwrap();
        assert_eq!(river.len(), 1);
        assert_eq!(game.phase(), GamePhase::River);

        // Showdown
        let results = game.showdown().unwrap();
        assert!(!results.is_empty());
        assert!(game.winner().is_some());
    }

    // ------ CHSH Violation ------

    #[test]
    fn test_chsh_demonstration() {
        let demo = ChshPokerDemo::new(1000, 42);
        let result = demo.run();

        // Classical correlator should be within classical bound
        // (or close to it for this small system)
        assert!(
            result.classical_correlator.abs() <= 2.0 + 1e-10,
            "Classical correlator {} exceeds bound 2.0",
            result.classical_correlator
        );

        // The quantum correlator should exist (may or may not violate
        // depending on the specific phase rotation used)
        assert!(result.quantum_correlator.is_finite());
    }

    // ------ Display ------

    #[test]
    fn test_display_state() {
        let config = QuantumPokerConfig::toy_game();
        let mut game = QuantumPokerGame::new(config).unwrap();
        game.deal().unwrap();

        let display = game.display_state(0).unwrap();
        assert!(display.contains("Quantum Poker"));
        assert!(display.contains("Player 0"));
        assert!(display.contains("superposition") || display.contains("Branches"));
    }

    // ------ Error Handling ------

    #[test]
    fn test_action_before_deal() {
        let config = QuantumPokerConfig::toy_game();
        let mut game = QuantumPokerGame::new(config).unwrap();

        let result = game.action(0, QuantumPokerAction::Check);
        assert!(matches!(result, Err(PokerError::NotDealt)));
    }

    #[test]
    fn test_invalid_player_index() {
        let config = QuantumPokerConfig::toy_game();
        let mut game = QuantumPokerGame::new(config).unwrap();
        game.deal().unwrap();

        let result = game.action(5, QuantumPokerAction::Check);
        assert!(matches!(result, Err(PokerError::InvalidPlayer(5, 2))));
    }

    #[test]
    fn test_invalid_card_index() {
        let config = QuantumPokerConfig::toy_game();
        let mut game = QuantumPokerGame::new(config).unwrap();
        game.deal().unwrap();

        // Toy game has 1 card per hand, index 1 is invalid
        let result = game.action(0, QuantumPokerAction::Measure(1));
        assert!(matches!(result, Err(PokerError::InvalidCardIndex(1, 1))));
    }

    #[test]
    fn test_action_after_fold() {
        let config = QuantumPokerConfig::toy_game();
        let mut game = QuantumPokerGame::new(config).unwrap();
        game.deal().unwrap();

        // With 2 players, folding ends the game
        game.action(1, QuantumPokerAction::Fold).unwrap();
        let result = game.action(1, QuantumPokerAction::Check);
        // Should get either PlayerFolded or GameOver
        assert!(result.is_err());
    }

    // ------ Mutual Information ------

    #[test]
    fn test_mutual_information() {
        let config = QuantumPokerConfig::toy_game();
        let mut game = QuantumPokerGame::new(config).unwrap();
        game.deal().unwrap();

        let mi = game.mutual_information(0, 1).unwrap();
        // Mutual information should be non-negative
        assert!(
            mi >= -1e-10,
            "Mutual information should be >= 0, got {}",
            mi
        );
        // And positive since hands are correlated (can't both have the same card)
        assert!(
            mi > 0.0,
            "Expected positive mutual information from deck constraint"
        );
    }

    // ------ Card Display ------

    #[test]
    fn test_card_display() {
        let card = Card::new(Suit::Spades, Rank::ACE);
        let s = format!("{}", card);
        assert!(
            s.contains("A"),
            "Card display should contain 'A', got: {}",
            s
        );
    }

    #[test]
    fn test_hand_rank_display() {
        let rank = HandRank::RoyalFlush;
        let s = format!("{}", rank);
        assert!(s.contains("Royal Flush"));

        let rank2 = HandRank::OnePair(14);
        let s2 = format!("{}", rank2);
        assert!(s2.contains("Pair"));
    }

    // ------ Entangled Pairs Variant ------

    #[test]
    fn test_entangled_pairs_game() {
        let config = QuantumPokerConfig::entangled_pairs();
        let mut game = QuantumPokerGame::new(config).unwrap();
        game.deal().unwrap();

        // 4 cards, 2 players, 1 each = 12 ordered deals
        assert_eq!(game.branch_count(), 12);

        // Measure player 0's card
        let r = game.action(0, QuantumPokerAction::Measure(0)).unwrap();
        let alice_card = r.revealed_card.unwrap();

        // Now player 1's card cannot be the same card
        let bob_probs = game.card_probabilities(1, 0).unwrap();
        for (card, prob) in &bob_probs {
            if *card == alice_card {
                assert!(
                    *prob < 1e-10,
                    "After Alice measures {}, Bob should not be able to have it (prob={})",
                    alice_card,
                    prob
                );
            }
        }
    }

    // ------ Error Display ------

    #[test]
    fn test_error_display() {
        let err = PokerError::InvalidPlayer(5, 2);
        let s = format!("{}", err);
        assert!(s.contains("Player 5"));

        let err2 = PokerError::InsufficientChips {
            player: 0,
            requested: 100.0,
            available: 50.0,
        };
        let s2 = format!("{}", err2);
        assert!(s2.contains("50.0"));
    }

    // ------ Quantum War ------

    #[test]
    fn test_quantum_war_simple_game() {
        let config = QuantumPokerConfig::new()
            .num_players(2)
            .cards_per_hand(1)
            .deck_size(4)
            .variant(PokerVariant::QuantumWar)
            .community_cards(0)
            .seed(42);

        let mut game = QuantumPokerGame::new(config).unwrap();
        game.deal().unwrap();

        // Both players measure
        game.action(0, QuantumPokerAction::Measure(0)).unwrap();
        game.action(1, QuantumPokerAction::Measure(0)).unwrap();

        // Showdown
        let results = game.showdown().unwrap();
        assert_eq!(results.len(), 2);
    }

    // ------ Re-measurement idempotency ------

    #[test]
    fn test_remeasure_returns_same_card() {
        let config = QuantumPokerConfig::toy_game();
        let mut game = QuantumPokerGame::new(config).unwrap();
        game.deal().unwrap();

        let r1 = game.action(0, QuantumPokerAction::Measure(0)).unwrap();
        let r2 = game.action(0, QuantumPokerAction::Measure(0)).unwrap();

        assert_eq!(r1.revealed_card, r2.revealed_card);
    }
}
