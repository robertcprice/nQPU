//! Domain applications: finance, logistics, games, NLP, art, and cognition.

// Game-style and interactive demos.
#[path = "games/quantum_chess.rs"]
pub mod quantum_chess;
#[path = "games/quantum_game.rs"]
pub mod quantum_game;
#[path = "games/quantum_poker.rs"]
pub mod quantum_poker;

// Decision, planning, and optimization-style domains.
#[path = "decision/quantum_climate.rs"]
pub mod quantum_climate;
#[path = "decision/quantum_finance.rs"]
pub mod quantum_finance;
#[path = "decision/quantum_logistics.rs"]
pub mod quantum_logistics;
#[path = "decision/selfish_routing.rs"]
pub mod selfish_routing;

// Creative and exploratory application modules.
#[path = "creative/quantum_art.rs"]
pub mod quantum_art;
#[path = "creative/quantum_cognition.rs"]
pub mod quantum_cognition;
#[path = "creative/quantum_nlp.rs"]
pub mod quantum_nlp;

#[cfg(feature = "experimental")]
#[path = "creative/creative_quantum.rs"]
pub mod creative_quantum;
