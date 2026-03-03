//! ArXiv Research Monitoring Engine for Quantum Computing
//!
//! This module provides a network-free ArXiv paper monitoring engine that
//! builds structured queries, parses Atom XML feed responses, scores papers
//! by relevance to quantum simulation research, and detects emerging trends.
//!
//! The monitor does not perform HTTP requests itself — it constructs query URLs
//! and parses XML responses, leaving actual network I/O to the caller.
//!
//! # Example
//!
//! ```rust
//! use nqpu_metal::arxiv_monitor::{ArxivMonitor, ArxivMonitorConfig};
//!
//! let config = ArxivMonitorConfig::builder()
//!     .add_category("quant-ph")
//!     .add_keyword("surface code", 0.9)
//!     .max_results(25)
//!     .days_lookback(7)
//!     .build();
//!
//! let monitor = ArxivMonitor::new(config);
//! let query_url = monitor.build_query();
//! // ... fetch the URL externally ...
//! // let papers = monitor.parse_atom_feed(&xml_response);
//! ```

use std::collections::HashMap;
use std::fmt;

// ============================================================
// ERRORS
// ============================================================

/// Errors that can occur during ArXiv monitoring operations.
#[derive(Debug, Clone)]
pub enum ArxivError {
    /// Malformed or unparseable XML content.
    ParseError(String),
    /// Invalid configuration parameter.
    ConfigError(String),
    /// Invalid date format or range.
    DateError(String),
    /// No papers matched the query criteria.
    EmptyResults,
}

impl fmt::Display for ArxivError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ArxivError::ParseError(msg) => write!(f, "XML parse error: {}", msg),
            ArxivError::ConfigError(msg) => write!(f, "Configuration error: {}", msg),
            ArxivError::DateError(msg) => write!(f, "Date error: {}", msg),
            ArxivError::EmptyResults => write!(f, "No papers matched the query"),
        }
    }
}

impl std::error::Error for ArxivError {}

// ============================================================
// SORT ORDER
// ============================================================

/// Sort order for ArXiv query results.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SortOrder {
    /// Sort by relevance to search terms.
    Relevance,
    /// Sort by original submission date.
    SubmittedDate,
    /// Sort by last updated date.
    LastUpdatedDate,
}

impl fmt::Display for SortOrder {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SortOrder::Relevance => write!(f, "relevance"),
            SortOrder::SubmittedDate => write!(f, "submittedDate"),
            SortOrder::LastUpdatedDate => write!(f, "lastUpdatedDate"),
        }
    }
}

/// Sort direction for query results.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SortDirection {
    Ascending,
    Descending,
}

impl fmt::Display for SortDirection {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SortDirection::Ascending => write!(f, "ascending"),
            SortDirection::Descending => write!(f, "descending"),
        }
    }
}

// ============================================================
// KEYWORD FILTER TARGET
// ============================================================

/// Which field a keyword filter applies to.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum KeywordField {
    /// Search in paper title.
    Title,
    /// Search in paper abstract.
    Abstract,
    /// Search in author names.
    Author,
    /// Search in all fields.
    All,
}

impl fmt::Display for KeywordField {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            KeywordField::Title => write!(f, "ti"),
            KeywordField::Abstract => write!(f, "abs"),
            KeywordField::Author => write!(f, "au"),
            KeywordField::All => write!(f, "all"),
        }
    }
}

// ============================================================
// ARXIV QUERY
// ============================================================

/// A structured query for the ArXiv API.
#[derive(Debug, Clone)]
pub struct ArxivQuery {
    /// ArXiv categories to search (e.g., "quant-ph", "cond-mat.str-el").
    pub categories: Vec<String>,
    /// Keyword filters: (field, keyword).
    pub keyword_filters: Vec<(KeywordField, String)>,
    /// Maximum number of results.
    pub max_results: usize,
    /// Starting index for pagination.
    pub start: usize,
    /// Sort order.
    pub sort_by: SortOrder,
    /// Sort direction.
    pub sort_order: SortDirection,
}

impl ArxivQuery {
    /// Create a new empty query.
    pub fn new() -> Self {
        Self {
            categories: Vec::new(),
            keyword_filters: Vec::new(),
            max_results: 50,
            start: 0,
            sort_by: SortOrder::SubmittedDate,
            sort_order: SortDirection::Descending,
        }
    }

    /// Add a category to the query.
    pub fn add_category(mut self, category: &str) -> Self {
        self.categories.push(category.to_string());
        self
    }

    /// Add a keyword filter.
    pub fn add_keyword(mut self, field: KeywordField, keyword: &str) -> Self {
        self.keyword_filters.push((field, keyword.to_string()));
        self
    }

    /// Set maximum results.
    pub fn max_results(mut self, n: usize) -> Self {
        self.max_results = n;
        self
    }

    /// Set sort order.
    pub fn sort_by(mut self, order: SortOrder) -> Self {
        self.sort_by = order;
        self
    }

    /// URL-encode a string for use in query parameters.
    fn url_encode(s: &str) -> String {
        let mut encoded = String::with_capacity(s.len() * 2);
        for ch in s.chars() {
            match ch {
                'a'..='z' | 'A'..='Z' | '0'..='9' | '-' | '_' | '.' | '~' => {
                    encoded.push(ch);
                }
                ' ' => encoded.push_str("%20"),
                '+' => encoded.push_str("%2B"),
                '&' => encoded.push_str("%26"),
                '=' => encoded.push_str("%3D"),
                ':' => encoded.push_str("%3A"),
                '/' => encoded.push_str("%2F"),
                '?' => encoded.push_str("%3F"),
                '#' => encoded.push_str("%23"),
                '"' => encoded.push_str("%22"),
                _ => {
                    let mut buf = [0u8; 4];
                    let s = ch.encode_utf8(&mut buf);
                    for byte in s.bytes() {
                        encoded.push_str(&format!("%{:02X}", byte));
                    }
                }
            }
        }
        encoded
    }

    /// Build the query URL for the ArXiv API.
    pub fn to_url(&self) -> String {
        let base = "http://export.arxiv.org/api/query";

        let mut search_parts: Vec<String> = Vec::new();

        // Build category filter
        if !self.categories.is_empty() {
            let cat_parts: Vec<String> = self
                .categories
                .iter()
                .map(|c| format!("cat:{}", Self::url_encode(c)))
                .collect();
            if cat_parts.len() == 1 {
                search_parts.push(cat_parts[0].clone());
            } else {
                let joined = cat_parts.join("+OR+");
                search_parts.push(format!("({})", joined));
            }
        }

        // Build keyword filters
        for (field, keyword) in &self.keyword_filters {
            search_parts.push(format!("{}:{}", field, Self::url_encode(keyword)));
        }

        let search_query = if search_parts.is_empty() {
            "all:quantum".to_string()
        } else {
            search_parts.join("+AND+")
        };

        format!(
            "{}?search_query={}&start={}&max_results={}&sortBy={}&sortOrder={}",
            base, search_query, self.start, self.max_results, self.sort_by, self.sort_order
        )
    }
}

impl Default for ArxivQuery {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Display for ArxivQuery {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.to_url())
    }
}

// ============================================================
// ARXIV PAPER
// ============================================================

/// Parsed metadata for an ArXiv paper.
#[derive(Debug, Clone)]
pub struct ArxivPaper {
    /// ArXiv identifier (e.g., "2401.12345").
    pub id: String,
    /// Paper title.
    pub title: String,
    /// List of author names.
    pub authors: Vec<String>,
    /// Full abstract text.
    pub abstract_text: String,
    /// All categories the paper belongs to.
    pub categories: Vec<String>,
    /// Primary category.
    pub primary_category: String,
    /// Publication date as ISO 8601 string.
    pub published: String,
    /// Last updated date as ISO 8601 string.
    pub updated: String,
    /// URL to the PDF.
    pub pdf_url: String,
    /// URL to the abstract page.
    pub abs_url: String,
}

impl ArxivPaper {
    /// Check if this paper appears in multiple quantum-relevant categories.
    pub fn is_crossover(&self) -> bool {
        let quantum_cats = [
            "quant-ph",
            "cond-mat.str-el",
            "cs.ET",
            "cond-mat.mes-hall",
            "physics.atom-ph",
            "cs.CR",
            "math-ph",
        ];
        let count = self
            .categories
            .iter()
            .filter(|c| quantum_cats.contains(&c.as_str()))
            .count();
        count >= 2
    }

    /// Extract the year from the published date string.
    pub fn year(&self) -> Option<u32> {
        if self.published.len() >= 4 {
            self.published[..4].parse().ok()
        } else {
            None
        }
    }

    /// Extract year-month-day as (u32, u32, u32) from published date.
    pub fn published_ymd(&self) -> Option<(u32, u32, u32)> {
        parse_date_ymd(&self.published)
    }
}

impl fmt::Display for ArxivPaper {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "[{}] {} ({}) -- {}",
            self.id,
            self.title,
            self.primary_category,
            if self.authors.len() <= 3 {
                self.authors.join(", ")
            } else {
                format!("{} et al.", self.authors[0])
            }
        )
    }
}

// ============================================================
// RESEARCH TREND
// ============================================================

/// A detected research trend from paper analysis.
#[derive(Debug, Clone)]
pub struct ResearchTrend {
    /// The topic or keyword being tracked.
    pub topic: String,
    /// Number of mentions in the most recent window.
    pub recent_count: usize,
    /// Number of mentions in the previous window (for comparison).
    pub previous_count: usize,
    /// Growth rate: (recent - previous) / max(previous, 1).
    pub growth_rate: f64,
    /// Whether this trend is considered "emerging" (growth > threshold).
    pub is_emerging: bool,
    /// Papers contributing to this trend.
    pub contributing_papers: Vec<String>,
}

impl ResearchTrend {
    /// Compute the growth rate from counts.
    pub fn compute_growth(recent: usize, previous: usize) -> f64 {
        let prev = if previous == 0 { 1 } else { previous };
        (recent as f64 - previous as f64) / prev as f64
    }
}

impl fmt::Display for ResearchTrend {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let status = if self.is_emerging {
            "EMERGING"
        } else {
            "stable"
        };
        write!(
            f,
            "{}: {} mentions (growth: {:.1}%, {}) -- {} papers",
            self.topic,
            self.recent_count,
            self.growth_rate * 100.0,
            status,
            self.contributing_papers.len()
        )
    }
}

// ============================================================
// RELEVANCE SCORER
// ============================================================

/// Scores papers by relevance to nQPU-Metal and quantum simulation research.
#[derive(Debug, Clone)]
pub struct RelevanceScorer {
    /// Keyword weights: keyword -> weight (0.0 to 1.0).
    pub keyword_weights: Vec<(String, f64)>,
    /// Category bonus scores.
    pub category_bonuses: HashMap<String, f64>,
    /// Recency boost factor (multiplied by days-since-published inverse).
    pub recency_factor: f64,
    /// Crossover category bonus.
    pub crossover_bonus: f64,
}

impl RelevanceScorer {
    /// Create a scorer with default quantum computing keyword weights.
    pub fn quantum_default() -> Self {
        let keyword_weights = vec![
            ("surface code".to_string(), 0.95),
            ("quantum error correction".to_string(), 0.90),
            ("stabilizer".to_string(), 0.85),
            ("tensor network".to_string(), 0.85),
            ("quantum simulation".to_string(), 0.80),
            ("quantum computing".to_string(), 0.70),
            ("fault tolerant".to_string(), 0.90),
            ("fault-tolerant".to_string(), 0.90),
            ("topological".to_string(), 0.75),
            ("clifford".to_string(), 0.80),
            ("magic state".to_string(), 0.85),
            ("lattice surgery".to_string(), 0.90),
            ("mps".to_string(), 0.70),
            ("matrix product state".to_string(), 0.80),
            ("dmrg".to_string(), 0.75),
            ("variational quantum".to_string(), 0.70),
            ("vqe".to_string(), 0.70),
            ("qaoa".to_string(), 0.70),
            ("quantum phase estimation".to_string(), 0.75),
            ("noise".to_string(), 0.50),
            ("decoherence".to_string(), 0.60),
            ("metal".to_string(), 0.30),
            ("gpu".to_string(), 0.50),
            ("gpu-accelerated".to_string(), 0.65),
            ("qubit".to_string(), 0.40),
            ("quantum".to_string(), 0.20),
            ("circuit cutting".to_string(), 0.85),
            ("quantum ldpc".to_string(), 0.90),
            ("floquet".to_string(), 0.80),
            ("bosonic code".to_string(), 0.80),
            ("cat qubit".to_string(), 0.75),
            ("superconducting".to_string(), 0.55),
            ("trapped ion".to_string(), 0.55),
            ("neutral atom".to_string(), 0.55),
            ("photonic".to_string(), 0.50),
        ];

        let mut category_bonuses = HashMap::new();
        category_bonuses.insert("quant-ph".to_string(), 0.30);
        category_bonuses.insert("cond-mat.str-el".to_string(), 0.20);
        category_bonuses.insert("cs.ET".to_string(), 0.15);
        category_bonuses.insert("cond-mat.mes-hall".to_string(), 0.10);
        category_bonuses.insert("physics.atom-ph".to_string(), 0.10);
        category_bonuses.insert("cs.CR".to_string(), 0.05);

        Self {
            keyword_weights,
            category_bonuses,
            recency_factor: 0.05,
            crossover_bonus: 0.10,
        }
    }

    /// Create a scorer from custom keyword weights.
    pub fn from_keywords(keywords: Vec<(String, f64)>) -> Self {
        Self {
            keyword_weights: keywords,
            category_bonuses: HashMap::new(),
            recency_factor: 0.05,
            crossover_bonus: 0.10,
        }
    }

    /// Score a single paper. Returns a value in [0.0, ~2.0] range.
    pub fn score(&self, paper: &ArxivPaper) -> f64 {
        let mut total = 0.0;

        // Keyword matching in title + abstract (case-insensitive)
        let title_lower = paper.title.to_lowercase();
        let abstract_lower = paper.abstract_text.to_lowercase();

        let mut matched_weight_sum = 0.0;
        let mut match_count = 0;

        for (keyword, weight) in &self.keyword_weights {
            let kw_lower = keyword.to_lowercase();
            let title_hit = title_lower.contains(&kw_lower);
            let abstract_hit = abstract_lower.contains(&kw_lower);

            if title_hit || abstract_hit {
                // Title matches are worth 1.5x abstract matches
                let multiplier = if title_hit { 1.5 } else { 1.0 };
                matched_weight_sum += weight * multiplier;
                match_count += 1;
            }
        }

        // Normalize keyword score: average of matched weights, scaled by coverage
        if match_count > 0 {
            let avg_weight = matched_weight_sum / match_count as f64;
            let coverage = (match_count as f64).min(10.0) / 10.0;
            total += avg_weight * (0.5 + 0.5 * coverage);
        }

        // Category bonus
        for cat in &paper.categories {
            if let Some(bonus) = self.category_bonuses.get(cat) {
                total += bonus;
            }
        }

        // Crossover bonus
        if paper.is_crossover() {
            total += self.crossover_bonus;
        }

        // Recency boost: baseline (actual date comparison needs external time)
        total += self.recency_factor;

        total
    }

    /// Score multiple papers and return them sorted by score descending.
    pub fn rank(&self, papers: &[ArxivPaper]) -> Vec<(ArxivPaper, f64)> {
        let mut scored: Vec<(ArxivPaper, f64)> = papers
            .iter()
            .map(|p| (p.clone(), self.score(p)))
            .collect();
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scored
    }
}

// ============================================================
// PAPER DIGEST
// ============================================================

/// A formatted summary of monitored papers.
#[derive(Debug, Clone)]
pub struct PaperDigest {
    /// Top papers ranked by relevance score.
    pub top_papers: Vec<(ArxivPaper, f64)>,
    /// Emerging research trends.
    pub emerging_trends: Vec<String>,
    /// Distribution of papers across categories.
    pub category_distribution: HashMap<String, usize>,
    /// Total number of papers analyzed.
    pub total_papers: usize,
}

impl fmt::Display for PaperDigest {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "=== ArXiv Research Digest ===")?;
        writeln!(f, "Total papers analyzed: {}", self.total_papers)?;
        writeln!(f)?;

        writeln!(f, "--- Top Papers ---")?;
        for (i, (paper, score)) in self.top_papers.iter().enumerate() {
            writeln!(f, "{}. [score: {:.3}] {}", i + 1, score, paper)?;
        }
        writeln!(f)?;

        if !self.emerging_trends.is_empty() {
            writeln!(f, "--- Emerging Trends ---")?;
            for trend in &self.emerging_trends {
                writeln!(f, "  * {}", trend)?;
            }
            writeln!(f)?;
        }

        writeln!(f, "--- Category Distribution ---")?;
        let mut cats: Vec<_> = self.category_distribution.iter().collect();
        cats.sort_by(|a, b| b.1.cmp(a.1));
        for (cat, count) in cats {
            writeln!(f, "  {}: {} papers", cat, count)?;
        }

        Ok(())
    }
}

// ============================================================
// ARXIV MONITOR CONFIG (BUILDER)
// ============================================================

/// Configuration for the ArXiv monitor.
#[derive(Debug, Clone)]
pub struct ArxivMonitorConfig {
    /// ArXiv categories to monitor.
    pub categories: Vec<String>,
    /// Keywords with relevance weights.
    pub keywords: Vec<(String, f64)>,
    /// Maximum results per query.
    pub max_results: usize,
    /// How many days back to search.
    pub days_lookback: usize,
    /// Minimum relevance score to include a paper.
    pub relevance_threshold: f64,
    /// Window in days for trend detection.
    pub trend_window_days: usize,
    /// Sort order for queries.
    pub sort_order: SortOrder,
    /// Trend growth threshold to consider "emerging".
    pub trend_growth_threshold: f64,
}

impl Default for ArxivMonitorConfig {
    fn default() -> Self {
        Self {
            categories: vec!["quant-ph".to_string()],
            keywords: vec![
                ("quantum error correction".to_string(), 0.9),
                ("surface code".to_string(), 0.9),
                ("quantum simulation".to_string(), 0.8),
                ("tensor network".to_string(), 0.8),
            ],
            max_results: 50,
            days_lookback: 7,
            relevance_threshold: 0.3,
            trend_window_days: 30,
            sort_order: SortOrder::SubmittedDate,
            trend_growth_threshold: 0.5,
        }
    }
}

/// Builder for [`ArxivMonitorConfig`].
pub struct ArxivMonitorConfigBuilder {
    config: ArxivMonitorConfig,
}

impl ArxivMonitorConfigBuilder {
    /// Start building a new config with defaults.
    pub fn new() -> Self {
        Self {
            config: ArxivMonitorConfig {
                categories: Vec::new(),
                keywords: Vec::new(),
                max_results: 50,
                days_lookback: 7,
                relevance_threshold: 0.3,
                trend_window_days: 30,
                sort_order: SortOrder::SubmittedDate,
                trend_growth_threshold: 0.5,
            },
        }
    }

    /// Add a category to monitor.
    pub fn add_category(mut self, cat: &str) -> Self {
        self.config.categories.push(cat.to_string());
        self
    }

    /// Set all categories at once.
    pub fn categories(mut self, cats: Vec<String>) -> Self {
        self.config.categories = cats;
        self
    }

    /// Add a keyword with its relevance weight.
    pub fn add_keyword(mut self, keyword: &str, weight: f64) -> Self {
        self.config.keywords.push((keyword.to_string(), weight));
        self
    }

    /// Set all keywords at once.
    pub fn keywords(mut self, kws: Vec<(String, f64)>) -> Self {
        self.config.keywords = kws;
        self
    }

    /// Set maximum results per query.
    pub fn max_results(mut self, n: usize) -> Self {
        self.config.max_results = n;
        self
    }

    /// Set how many days back to search.
    pub fn days_lookback(mut self, days: usize) -> Self {
        self.config.days_lookback = days;
        self
    }

    /// Set the relevance score threshold.
    pub fn relevance_threshold(mut self, threshold: f64) -> Self {
        self.config.relevance_threshold = threshold;
        self
    }

    /// Set the trend detection window in days.
    pub fn trend_window_days(mut self, days: usize) -> Self {
        self.config.trend_window_days = days;
        self
    }

    /// Set the sort order.
    pub fn sort_order(mut self, order: SortOrder) -> Self {
        self.config.sort_order = order;
        self
    }

    /// Set the growth threshold for emerging trend detection.
    pub fn trend_growth_threshold(mut self, threshold: f64) -> Self {
        self.config.trend_growth_threshold = threshold;
        self
    }

    /// Build the final configuration.
    pub fn build(self) -> ArxivMonitorConfig {
        self.config
    }
}

impl ArxivMonitorConfig {
    /// Start building a config with the builder pattern.
    pub fn builder() -> ArxivMonitorConfigBuilder {
        ArxivMonitorConfigBuilder::new()
    }

    /// Create a config optimized for quantum error correction research.
    pub fn qec_preset() -> Self {
        Self::builder()
            .add_category("quant-ph")
            .add_category("cond-mat.str-el")
            .add_keyword("surface code", 0.95)
            .add_keyword("quantum error correction", 0.95)
            .add_keyword("fault tolerant", 0.90)
            .add_keyword("stabilizer", 0.85)
            .add_keyword("lattice surgery", 0.90)
            .add_keyword("magic state", 0.85)
            .add_keyword("quantum ldpc", 0.90)
            .add_keyword("floquet code", 0.80)
            .add_keyword("decoder", 0.70)
            .max_results(100)
            .days_lookback(14)
            .relevance_threshold(0.4)
            .build()
    }

    /// Create a config for quantum simulation research.
    pub fn simulation_preset() -> Self {
        Self::builder()
            .add_category("quant-ph")
            .add_category("cond-mat.str-el")
            .add_category("physics.atom-ph")
            .add_keyword("quantum simulation", 0.90)
            .add_keyword("tensor network", 0.85)
            .add_keyword("matrix product state", 0.85)
            .add_keyword("dmrg", 0.80)
            .add_keyword("variational", 0.70)
            .add_keyword("hamiltonian simulation", 0.85)
            .add_keyword("trotter", 0.80)
            .add_keyword("gpu", 0.60)
            .max_results(75)
            .days_lookback(7)
            .relevance_threshold(0.3)
            .build()
    }
}

// ============================================================
// XML PARSING HELPERS
// ============================================================

/// Extract the text content between an opening and closing XML tag.
/// Returns `None` if the tag is not found.
fn extract_tag<'a>(xml: &'a str, tag: &str) -> Option<&'a str> {
    let open = format!("<{}", tag);
    let close = format!("</{}>", tag);

    let start_pos = xml.find(&open)?;
    // Find the end of the opening tag (handle attributes)
    let content_start = xml[start_pos..].find('>')? + start_pos + 1;
    let end_pos = xml[content_start..].find(&close)? + content_start;

    Some(xml[content_start..end_pos].trim())
}

/// Extract all occurrences of a tag's text content.
fn extract_all_tags<'a>(xml: &'a str, tag: &str) -> Vec<&'a str> {
    let open = format!("<{}", tag);
    let close = format!("</{}>", tag);
    let mut results = Vec::new();
    let mut search_from = 0;

    while search_from < xml.len() {
        let remaining = &xml[search_from..];
        if let Some(start_offset) = remaining.find(&open) {
            let abs_start = search_from + start_offset;
            let after_open = &xml[abs_start..];
            if let Some(gt_offset) = after_open.find('>') {
                let content_start = abs_start + gt_offset + 1;
                if let Some(end_offset) = xml[content_start..].find(&close) {
                    let content_end = content_start + end_offset;
                    results.push(xml[content_start..content_end].trim());
                    search_from = content_end + close.len();
                } else {
                    break;
                }
            } else {
                break;
            }
        } else {
            break;
        }
    }

    results
}

/// Extract an attribute value from a tag.
fn extract_attribute<'a>(tag_text: &'a str, attr: &str) -> Option<&'a str> {
    let pattern = format!("{}=\"", attr);
    let start = tag_text.find(&pattern)? + pattern.len();
    let end = tag_text[start..].find('"')? + start;
    Some(&tag_text[start..end])
}

/// Extract all `<entry>` blocks from an Atom feed.
fn extract_entries(xml: &str) -> Vec<&str> {
    let mut entries = Vec::new();
    let mut search_from = 0;

    while search_from < xml.len() {
        let remaining = &xml[search_from..];
        if let Some(start_offset) = remaining.find("<entry>") {
            let abs_start = search_from + start_offset;
            if let Some(end_offset) = xml[abs_start..].find("</entry>") {
                let entry_end = abs_start + end_offset + "</entry>".len();
                entries.push(&xml[abs_start..entry_end]);
                search_from = entry_end;
            } else {
                break;
            }
        } else {
            break;
        }
    }

    entries
}

/// Decode basic XML entities.
fn decode_xml_entities(s: &str) -> String {
    s.replace("&amp;", "&")
        .replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&quot;", "\"")
        .replace("&apos;", "'")
}

/// Parse a date string and extract (year, month, day).
fn parse_date_ymd(date_str: &str) -> Option<(u32, u32, u32)> {
    // Handles "2024-01-15T12:00:00Z" and "2024-01-15" formats
    if date_str.len() < 10 {
        return None;
    }
    let year: u32 = date_str[0..4].parse().ok()?;
    let month: u32 = date_str[5..7].parse().ok()?;
    let day: u32 = date_str[8..10].parse().ok()?;
    Some((year, month, day))
}

/// Approximate days between two dates (simplified, assumes 30-day months).
fn approx_days_between(earlier: (u32, u32, u32), later: (u32, u32, u32)) -> i32 {
    let d1 = earlier.0 as i32 * 365 + earlier.1 as i32 * 30 + earlier.2 as i32;
    let d2 = later.0 as i32 * 365 + later.1 as i32 * 30 + later.2 as i32;
    d2 - d1
}

// ============================================================
// ARXIV MONITOR
// ============================================================

/// The main ArXiv monitoring engine.
///
/// Builds queries, parses responses, scores papers, and detects trends.
/// Does not perform any network I/O -- that is left to the caller.
#[derive(Debug, Clone)]
pub struct ArxivMonitor {
    /// Monitor configuration.
    pub config: ArxivMonitorConfig,
    /// Relevance scorer instance.
    scorer: RelevanceScorer,
}

impl ArxivMonitor {
    /// Create a new monitor with the given configuration.
    pub fn new(config: ArxivMonitorConfig) -> Self {
        // Build scorer from config keywords + defaults
        let mut keyword_weights = config.keywords.clone();

        // Add default quantum computing terms if none provided
        if keyword_weights.is_empty() {
            keyword_weights = RelevanceScorer::quantum_default().keyword_weights;
        }

        let scorer = RelevanceScorer {
            keyword_weights,
            ..RelevanceScorer::quantum_default()
        };

        Self { config, scorer }
    }

    /// Build a query URL from the current configuration.
    pub fn build_query(&self) -> String {
        let mut query = ArxivQuery::new().max_results(self.config.max_results);

        for cat in &self.config.categories {
            query = query.add_category(cat);
        }

        // Add top-weighted keywords as title/abstract search terms
        let mut sorted_kws = self.config.keywords.clone();
        sorted_kws.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // Use up to 3 top keywords in the query (arXiv API has limits)
        for (kw, _weight) in sorted_kws.iter().take(3) {
            query = query.add_keyword(KeywordField::All, kw);
        }

        query.sort_by = self.config.sort_order;
        query.to_url()
    }

    /// Parse an ArXiv Atom XML feed response into a list of papers.
    pub fn parse_atom_feed(&self, xml: &str) -> Result<Vec<ArxivPaper>, ArxivError> {
        if xml.trim().is_empty() {
            return Err(ArxivError::ParseError("Empty XML input".to_string()));
        }

        let entries = extract_entries(xml);
        let mut papers = Vec::new();

        for entry in entries {
            match self.parse_entry(entry) {
                Ok(paper) => papers.push(paper),
                Err(_) => {
                    // Skip malformed entries but continue parsing
                    continue;
                }
            }
        }

        Ok(papers)
    }

    /// Parse a single `<entry>` XML block into an ArxivPaper.
    fn parse_entry(&self, entry: &str) -> Result<ArxivPaper, ArxivError> {
        let id_raw = extract_tag(entry, "id")
            .ok_or_else(|| ArxivError::ParseError("Missing <id> tag".to_string()))?;

        // Extract ArXiv ID from URL: http://arxiv.org/abs/2401.12345v1 -> 2401.12345
        let id = id_raw.rsplit('/').next().unwrap_or(id_raw).trim();
        // Strip version suffix (v1, v2, etc.)
        let id = if let Some(v_pos) = id.rfind('v') {
            if id[v_pos + 1..].chars().all(|c| c.is_ascii_digit()) && v_pos + 1 < id.len() {
                &id[..v_pos]
            } else {
                id
            }
        } else {
            id
        };

        let title = extract_tag(entry, "title")
            .map(|t| decode_xml_entities(t).replace('\n', " ").trim().to_string())
            .unwrap_or_default();

        let abstract_text = extract_tag(entry, "summary")
            .map(|t| decode_xml_entities(t).replace('\n', " ").trim().to_string())
            .unwrap_or_default();

        let published = extract_tag(entry, "published")
            .unwrap_or("")
            .to_string();

        let updated = extract_tag(entry, "updated")
            .unwrap_or("")
            .to_string();

        // Parse authors: <author><name>John Doe</name></author>
        let authors = extract_all_tags(entry, "name")
            .into_iter()
            .map(|n| decode_xml_entities(n).to_string())
            .collect();

        // Parse categories from <category term="quant-ph" ... />
        let mut categories = Vec::new();
        let mut primary_category = String::new();
        let mut search_pos = 0;
        while search_pos < entry.len() {
            if let Some(cat_start) = entry[search_pos..].find("<category") {
                let abs_start = search_pos + cat_start;
                if let Some(cat_end) = entry[abs_start..].find("/>") {
                    let tag_str = &entry[abs_start..abs_start + cat_end + 2];
                    if let Some(term) = extract_attribute(tag_str, "term") {
                        categories.push(term.to_string());
                    }
                    search_pos = abs_start + cat_end + 2;
                } else if let Some(cat_end) = entry[abs_start..].find('>') {
                    let tag_str = &entry[abs_start..abs_start + cat_end + 1];
                    if let Some(term) = extract_attribute(tag_str, "term") {
                        categories.push(term.to_string());
                    }
                    search_pos = abs_start + cat_end + 1;
                } else {
                    break;
                }
            } else {
                break;
            }
        }

        // Primary category: first one, or from arxiv:primary_category
        if let Some(prim_start) = entry.find("arxiv:primary_category") {
            if let Some(end) = entry[prim_start..].find("/>") {
                let tag_str = &entry[prim_start..prim_start + end + 2];
                if let Some(term) = extract_attribute(tag_str, "term") {
                    primary_category = term.to_string();
                }
            }
        }
        if primary_category.is_empty() {
            primary_category = categories.first().cloned().unwrap_or_default();
        }

        // Parse links for PDF and abstract URLs
        let mut pdf_url = String::new();
        let mut abs_url = String::new();
        let mut link_search = 0;
        while link_search < entry.len() {
            if let Some(link_start) = entry[link_search..].find("<link") {
                let abs_start = link_search + link_start;
                let link_end_pos = entry[abs_start..]
                    .find("/>")
                    .or_else(|| entry[abs_start..].find('>'))
                    .map(|p| abs_start + p + 2)
                    .unwrap_or(entry.len());
                let tag_str = &entry[abs_start..link_end_pos];

                if let Some(href) = extract_attribute(tag_str, "href") {
                    if tag_str.contains("title=\"pdf\"") || href.contains("/pdf/") {
                        pdf_url = href.to_string();
                    } else if abs_url.is_empty() {
                        abs_url = href.to_string();
                    }
                }
                link_search = link_end_pos;
            } else {
                break;
            }
        }

        // Fallback URL construction
        if pdf_url.is_empty() {
            pdf_url = format!("http://arxiv.org/pdf/{}", id);
        }
        if abs_url.is_empty() {
            abs_url = format!("http://arxiv.org/abs/{}", id);
        }

        Ok(ArxivPaper {
            id: id.to_string(),
            title,
            authors,
            abstract_text,
            categories,
            primary_category,
            published,
            updated,
            pdf_url,
            abs_url,
        })
    }

    /// Score papers by relevance and return them sorted (highest first).
    pub fn score_papers(&self, papers: &[ArxivPaper]) -> Vec<(ArxivPaper, f64)> {
        self.scorer.rank(papers)
    }

    /// Detect research trends from a collection of papers.
    ///
    /// This analyzes keyword frequency in titles and abstracts to identify
    /// topics that are accelerating in mentions.
    pub fn detect_trends(&self, papers: &[ArxivPaper]) -> Vec<ResearchTrend> {
        let trend_keywords: Vec<&str> = vec![
            "surface code",
            "quantum error correction",
            "stabilizer",
            "tensor network",
            "variational",
            "quantum simulation",
            "fault tolerant",
            "lattice surgery",
            "magic state",
            "quantum ldpc",
            "floquet",
            "bosonic",
            "neutral atom",
            "trapped ion",
            "superconducting",
            "machine learning",
            "classical shadow",
            "circuit cutting",
            "quantum advantage",
            "quantum supremacy",
        ];

        // Split papers into "recent" and "older" halves for trend comparison
        let midpoint = papers.len() / 2;
        let (older, recent) = if papers.len() >= 2 {
            (&papers[..midpoint], &papers[midpoint..])
        } else {
            (papers, papers)
        };

        let mut trends = Vec::new();

        for keyword in &trend_keywords {
            let kw_lower = keyword.to_lowercase();

            let recent_count = recent
                .iter()
                .filter(|p| {
                    p.title.to_lowercase().contains(&kw_lower)
                        || p.abstract_text.to_lowercase().contains(&kw_lower)
                })
                .count();

            let previous_count = older
                .iter()
                .filter(|p| {
                    p.title.to_lowercase().contains(&kw_lower)
                        || p.abstract_text.to_lowercase().contains(&kw_lower)
                })
                .count();

            let growth_rate = ResearchTrend::compute_growth(recent_count, previous_count);
            let is_emerging =
                growth_rate > self.config.trend_growth_threshold && recent_count >= 2;

            let contributing: Vec<String> = papers
                .iter()
                .filter(|p| {
                    p.title.to_lowercase().contains(&kw_lower)
                        || p.abstract_text.to_lowercase().contains(&kw_lower)
                })
                .map(|p| p.id.clone())
                .collect();

            if recent_count > 0 || previous_count > 0 {
                trends.push(ResearchTrend {
                    topic: keyword.to_string(),
                    recent_count,
                    previous_count,
                    growth_rate,
                    is_emerging,
                    contributing_papers: contributing,
                });
            }
        }

        // Sort: emerging first, then by growth rate
        trends.sort_by(|a, b| {
            b.is_emerging.cmp(&a.is_emerging).then_with(|| {
                b.growth_rate
                    .partial_cmp(&a.growth_rate)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
        });

        trends
    }

    /// Generate a comprehensive digest of analyzed papers.
    pub fn generate_digest(&self, papers: &[ArxivPaper]) -> PaperDigest {
        let scored = self.score_papers(papers);
        let trends = self.detect_trends(papers);

        // Category distribution
        let mut cat_dist: HashMap<String, usize> = HashMap::new();
        for paper in papers {
            for cat in &paper.categories {
                *cat_dist.entry(cat.clone()).or_insert(0) += 1;
            }
        }

        // Top papers (above threshold, up to 10)
        let top_papers: Vec<(ArxivPaper, f64)> = scored
            .into_iter()
            .filter(|(_, score)| *score >= self.config.relevance_threshold)
            .take(10)
            .collect();

        // Emerging trend names
        let emerging_trends: Vec<String> = trends
            .iter()
            .filter(|t| t.is_emerging)
            .map(|t| format!("{} (growth: {:.0}%)", t.topic, t.growth_rate * 100.0))
            .collect();

        PaperDigest {
            top_papers,
            emerging_trends,
            category_distribution: cat_dist,
            total_papers: papers.len(),
        }
    }

    /// Filter papers by relevance score threshold.
    pub fn filter_by_relevance(
        &self,
        papers: &[ArxivPaper],
        threshold: f64,
    ) -> Vec<(ArxivPaper, f64)> {
        self.scorer
            .rank(papers)
            .into_iter()
            .filter(|(_, score)| *score >= threshold)
            .collect()
    }

    /// Get papers that appear in multiple quantum-relevant categories.
    pub fn find_crossover_papers<'a>(&self, papers: &'a [ArxivPaper]) -> Vec<&'a ArxivPaper> {
        papers.iter().filter(|p| p.is_crossover()).collect()
    }
}

impl fmt::Display for ArxivMonitor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "ArxivMonitor(categories={:?}, keywords={}, max_results={}, lookback={}d)",
            self.config.categories,
            self.config.keywords.len(),
            self.config.max_results,
            self.config.days_lookback,
        )
    }
}

// ============================================================
// PREDEFINED KEYWORD SETS
// ============================================================

/// Predefined keyword sets for common quantum computing research areas.
pub struct QuantumKeywordSets;

impl QuantumKeywordSets {
    /// Keywords related to quantum error correction.
    pub fn error_correction() -> Vec<(String, f64)> {
        vec![
            ("surface code".to_string(), 0.95),
            ("quantum error correction".to_string(), 0.95),
            ("fault tolerant".to_string(), 0.90),
            ("stabilizer code".to_string(), 0.85),
            ("lattice surgery".to_string(), 0.90),
            ("magic state distillation".to_string(), 0.85),
            ("quantum ldpc".to_string(), 0.90),
            ("floquet code".to_string(), 0.80),
            ("color code".to_string(), 0.80),
            ("toric code".to_string(), 0.80),
            ("decoder".to_string(), 0.70),
            ("threshold".to_string(), 0.60),
        ]
    }

    /// Keywords related to quantum simulation and tensor networks.
    pub fn simulation() -> Vec<(String, f64)> {
        vec![
            ("quantum simulation".to_string(), 0.90),
            ("tensor network".to_string(), 0.85),
            ("matrix product state".to_string(), 0.85),
            ("dmrg".to_string(), 0.80),
            ("tebd".to_string(), 0.80),
            ("hamiltonian simulation".to_string(), 0.85),
            ("trotter".to_string(), 0.75),
            ("variational quantum eigensolver".to_string(), 0.80),
            ("quantum chemistry".to_string(), 0.75),
            ("many-body".to_string(), 0.70),
        ]
    }

    /// Keywords related to quantum hardware platforms.
    pub fn hardware() -> Vec<(String, f64)> {
        vec![
            ("superconducting qubit".to_string(), 0.80),
            ("trapped ion".to_string(), 0.80),
            ("neutral atom".to_string(), 0.80),
            ("photonic quantum".to_string(), 0.75),
            ("topological qubit".to_string(), 0.85),
            ("quantum dot".to_string(), 0.70),
            ("silicon spin".to_string(), 0.75),
            ("cat qubit".to_string(), 0.80),
            ("bosonic code".to_string(), 0.80),
            ("transmon".to_string(), 0.75),
        ]
    }

    /// Keywords related to quantum algorithms and applications.
    pub fn algorithms() -> Vec<(String, f64)> {
        vec![
            ("quantum algorithm".to_string(), 0.80),
            ("quantum advantage".to_string(), 0.85),
            ("quantum machine learning".to_string(), 0.75),
            ("qaoa".to_string(), 0.75),
            ("grover".to_string(), 0.65),
            ("shor".to_string(), 0.65),
            ("quantum walk".to_string(), 0.70),
            ("quantum optimization".to_string(), 0.75),
            ("quantum annealing".to_string(), 0.70),
            ("quantum phase estimation".to_string(), 0.80),
        ]
    }
}

// ============================================================
// TESTS
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;

    // ------- Helper: realistic ArXiv Atom feed XML -------

    fn sample_atom_feed() -> String {
        r#"<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <link href="http://arxiv.org/api/query?search_query=cat:quant-ph&amp;start=0&amp;max_results=10" rel="self" type="application/atom+xml"/>
  <title type="html">ArXiv Query: search_query=cat:quant-ph</title>
  <id>http://arxiv.org/api/query?search_query=cat:quant-ph</id>
  <updated>2024-01-20T00:00:00-05:00</updated>
  <opensearch:totalResults xmlns:opensearch="http://a9.com/-/spec/opensearch/1.1/">12345</opensearch:totalResults>
  <opensearch:startIndex xmlns:opensearch="http://a9.com/-/spec/opensearch/1.1/">0</opensearch:startIndex>
  <opensearch:itemsPerPage xmlns:opensearch="http://a9.com/-/spec/opensearch/1.1/">10</opensearch:itemsPerPage>
  <entry>
    <id>http://arxiv.org/abs/2401.12345v1</id>
    <updated>2024-01-18T18:00:00Z</updated>
    <published>2024-01-15T12:00:00Z</published>
    <title>Improved Surface Code Decoders Using Tensor Network Methods</title>
    <summary>We present a novel approach to decoding surface codes using tensor network
contraction methods. Our decoder achieves near-optimal performance with
polynomial runtime complexity. We demonstrate fault-tolerant quantum error
correction below the threshold on a distance-7 surface code.</summary>
    <author><name>Alice Quantum</name></author>
    <author><name>Bob Tensor</name></author>
    <author><name>Charlie Surface</name></author>
    <arxiv:primary_category xmlns:arxiv="http://arxiv.org/schemas/atom" term="quant-ph" scheme="http://arxiv.org/schemas/atom"/>
    <category term="quant-ph" scheme="http://arxiv.org/schemas/atom"/>
    <category term="cond-mat.str-el" scheme="http://arxiv.org/schemas/atom"/>
    <link href="http://arxiv.org/abs/2401.12345v1" rel="alternate" type="text/html"/>
    <link title="pdf" href="http://arxiv.org/pdf/2401.12345v1" rel="related" type="application/pdf"/>
  </entry>
  <entry>
    <id>http://arxiv.org/abs/2401.67890v2</id>
    <updated>2024-01-19T10:00:00Z</updated>
    <published>2024-01-10T08:00:00Z</published>
    <title>Quantum Machine Learning with Parameterized Circuits</title>
    <summary>We explore the use of parameterized quantum circuits for classification
tasks. Our variational quantum approach demonstrates competitive accuracy
on benchmark datasets using near-term quantum hardware.</summary>
    <author><name>Diana Variational</name></author>
    <author><name>Eve Algorithm</name></author>
    <arxiv:primary_category xmlns:arxiv="http://arxiv.org/schemas/atom" term="quant-ph" scheme="http://arxiv.org/schemas/atom"/>
    <category term="quant-ph" scheme="http://arxiv.org/schemas/atom"/>
    <category term="cs.LG" scheme="http://arxiv.org/schemas/atom"/>
    <link href="http://arxiv.org/abs/2401.67890v2" rel="alternate" type="text/html"/>
    <link title="pdf" href="http://arxiv.org/pdf/2401.67890v2" rel="related" type="application/pdf"/>
  </entry>
  <entry>
    <id>http://arxiv.org/abs/2401.11111v1</id>
    <updated>2024-01-20T14:00:00Z</updated>
    <published>2024-01-20T14:00:00Z</published>
    <title>Floquet Codes with Improved Thresholds via Lattice Surgery</title>
    <summary>We introduce a new family of Floquet codes that achieve higher thresholds
than previously known constructions. Using lattice surgery techniques, we
demonstrate fault-tolerant logical operations with a surface code overhead
reduction of 30%. Our stabilizer measurements leverage quantum LDPC structure.</summary>
    <author><name>Frank Floquet</name></author>
    <arxiv:primary_category xmlns:arxiv="http://arxiv.org/schemas/atom" term="quant-ph" scheme="http://arxiv.org/schemas/atom"/>
    <category term="quant-ph" scheme="http://arxiv.org/schemas/atom"/>
    <link href="http://arxiv.org/abs/2401.11111v1" rel="alternate" type="text/html"/>
    <link title="pdf" href="http://arxiv.org/pdf/2401.11111v1" rel="related" type="application/pdf"/>
  </entry>
  <entry>
    <id>http://arxiv.org/abs/2401.22222v1</id>
    <updated>2024-01-16T09:00:00Z</updated>
    <published>2024-01-16T09:00:00Z</published>
    <title>Classical Simulation of Noisy Quantum Circuits</title>
    <summary>We develop efficient classical simulation techniques for noisy quantum
circuits. By exploiting noise-induced structure, we show that certain
classes of noisy circuits can be simulated in polynomial time, challenging
claims of quantum advantage for near-term devices.</summary>
    <author><name>Grace Classical</name></author>
    <author><name>Hank Noise</name></author>
    <arxiv:primary_category xmlns:arxiv="http://arxiv.org/schemas/atom" term="quant-ph" scheme="http://arxiv.org/schemas/atom"/>
    <category term="quant-ph" scheme="http://arxiv.org/schemas/atom"/>
    <category term="cs.CC" scheme="http://arxiv.org/schemas/atom"/>
    <link href="http://arxiv.org/abs/2401.22222v1" rel="alternate" type="text/html"/>
    <link title="pdf" href="http://arxiv.org/pdf/2401.22222v1" rel="related" type="application/pdf"/>
  </entry>
</feed>"#
            .to_string()
    }

    fn make_test_paper(
        id: &str,
        title: &str,
        abstract_text: &str,
        categories: &[&str],
    ) -> ArxivPaper {
        ArxivPaper {
            id: id.to_string(),
            title: title.to_string(),
            authors: vec!["Test Author".to_string()],
            abstract_text: abstract_text.to_string(),
            categories: categories.iter().map(|s| s.to_string()).collect(),
            primary_category: categories.first().unwrap_or(&"quant-ph").to_string(),
            published: "2024-01-15T12:00:00Z".to_string(),
            updated: "2024-01-15T12:00:00Z".to_string(),
            pdf_url: format!("http://arxiv.org/pdf/{}", id),
            abs_url: format!("http://arxiv.org/abs/{}", id),
        }
    }

    // ============================================================
    // CONFIG BUILDER TESTS
    // ============================================================

    #[test]
    fn test_config_builder_defaults() {
        let config = ArxivMonitorConfig::builder().build();
        assert_eq!(config.max_results, 50);
        assert_eq!(config.days_lookback, 7);
        assert!((config.relevance_threshold - 0.3).abs() < f64::EPSILON);
        assert_eq!(config.trend_window_days, 30);
        assert!(config.categories.is_empty());
        assert!(config.keywords.is_empty());
    }

    #[test]
    fn test_config_builder_customization() {
        let config = ArxivMonitorConfig::builder()
            .add_category("quant-ph")
            .add_category("cond-mat.str-el")
            .add_keyword("surface code", 0.95)
            .add_keyword("tensor network", 0.85)
            .max_results(100)
            .days_lookback(14)
            .relevance_threshold(0.5)
            .trend_window_days(60)
            .sort_order(SortOrder::Relevance)
            .trend_growth_threshold(0.8)
            .build();

        assert_eq!(config.categories.len(), 2);
        assert_eq!(config.categories[0], "quant-ph");
        assert_eq!(config.categories[1], "cond-mat.str-el");
        assert_eq!(config.keywords.len(), 2);
        assert_eq!(config.max_results, 100);
        assert_eq!(config.days_lookback, 14);
        assert!((config.relevance_threshold - 0.5).abs() < f64::EPSILON);
        assert_eq!(config.trend_window_days, 60);
        assert_eq!(config.sort_order, SortOrder::Relevance);
        assert!((config.trend_growth_threshold - 0.8).abs() < f64::EPSILON);
    }

    #[test]
    fn test_config_qec_preset() {
        let config = ArxivMonitorConfig::qec_preset();
        assert!(config.categories.contains(&"quant-ph".to_string()));
        assert_eq!(config.max_results, 100);
        assert_eq!(config.days_lookback, 14);
        assert!(config.keywords.len() >= 5);
    }

    #[test]
    fn test_config_simulation_preset() {
        let config = ArxivMonitorConfig::simulation_preset();
        assert!(config.categories.contains(&"quant-ph".to_string()));
        assert!(config.categories.contains(&"cond-mat.str-el".to_string()));
        assert_eq!(config.max_results, 75);
    }

    // ============================================================
    // QUERY URL CONSTRUCTION TESTS
    // ============================================================

    #[test]
    fn test_query_single_category() {
        let query = ArxivQuery::new().add_category("quant-ph").max_results(10);
        let url = query.to_url();
        assert!(url.starts_with("http://export.arxiv.org/api/query?"));
        assert!(url.contains("cat:quant-ph"));
        assert!(url.contains("max_results=10"));
    }

    #[test]
    fn test_query_multiple_categories() {
        let query = ArxivQuery::new()
            .add_category("quant-ph")
            .add_category("cond-mat.str-el")
            .max_results(20);
        let url = query.to_url();
        assert!(url.contains("cat:quant-ph+OR+cat:cond-mat.str-el"));
        assert!(url.contains("max_results=20"));
    }

    #[test]
    fn test_query_with_keywords() {
        let query = ArxivQuery::new()
            .add_category("quant-ph")
            .add_keyword(KeywordField::Title, "surface code")
            .max_results(5);
        let url = query.to_url();
        assert!(url.contains("ti:surface%20code"));
        assert!(url.contains("+AND+"));
    }

    #[test]
    fn test_query_sort_order() {
        let mut query = ArxivQuery::new().add_category("quant-ph");
        query.sort_by = SortOrder::Relevance;
        let url = query.to_url();
        assert!(url.contains("sortBy=relevance"));

        query.sort_by = SortOrder::LastUpdatedDate;
        let url = query.to_url();
        assert!(url.contains("sortBy=lastUpdatedDate"));
    }

    #[test]
    fn test_query_url_encoding() {
        let query =
            ArxivQuery::new().add_keyword(KeywordField::Abstract, "quantum error correction");
        let url = query.to_url();
        assert!(url.contains("abs:quantum%20error%20correction"));
    }

    #[test]
    fn test_monitor_build_query() {
        let config = ArxivMonitorConfig::builder()
            .add_category("quant-ph")
            .add_keyword("surface code", 0.95)
            .add_keyword("tensor network", 0.85)
            .max_results(25)
            .build();
        let monitor = ArxivMonitor::new(config);
        let url = monitor.build_query();
        assert!(url.contains("max_results=25"));
        assert!(url.contains("quant-ph"));
    }

    // ============================================================
    // ATOM XML FEED PARSING TESTS
    // ============================================================

    #[test]
    fn test_parse_atom_feed_basic() {
        let config = ArxivMonitorConfig::default();
        let monitor = ArxivMonitor::new(config);
        let xml = sample_atom_feed();

        let papers = monitor.parse_atom_feed(&xml).unwrap();
        assert_eq!(papers.len(), 4);

        // First paper checks
        let p1 = &papers[0];
        assert_eq!(p1.id, "2401.12345");
        assert!(p1.title.contains("Surface Code Decoders"));
        assert_eq!(p1.authors.len(), 3);
        assert_eq!(p1.authors[0], "Alice Quantum");
        assert_eq!(p1.primary_category, "quant-ph");
        assert!(p1.categories.contains(&"quant-ph".to_string()));
        assert!(p1.categories.contains(&"cond-mat.str-el".to_string()));
        assert_eq!(p1.published, "2024-01-15T12:00:00Z");
        assert!(p1.pdf_url.contains("pdf"));
    }

    #[test]
    fn test_parse_paper_id_version_stripping() {
        let config = ArxivMonitorConfig::default();
        let monitor = ArxivMonitor::new(config);
        let xml = sample_atom_feed();

        let papers = monitor.parse_atom_feed(&xml).unwrap();
        // v1 and v2 suffixes should be stripped
        assert_eq!(papers[0].id, "2401.12345");
        assert_eq!(papers[1].id, "2401.67890");
    }

    #[test]
    fn test_parse_paper_authors() {
        let config = ArxivMonitorConfig::default();
        let monitor = ArxivMonitor::new(config);
        let xml = sample_atom_feed();
        let papers = monitor.parse_atom_feed(&xml).unwrap();

        // Paper 1: 3 authors
        assert_eq!(papers[0].authors.len(), 3);
        assert_eq!(papers[0].authors[2], "Charlie Surface");

        // Paper 2: 2 authors
        assert_eq!(papers[1].authors.len(), 2);

        // Paper 3: 1 author
        assert_eq!(papers[2].authors.len(), 1);
        assert_eq!(papers[2].authors[0], "Frank Floquet");
    }

    #[test]
    fn test_parse_paper_categories() {
        let config = ArxivMonitorConfig::default();
        let monitor = ArxivMonitor::new(config);
        let xml = sample_atom_feed();
        let papers = monitor.parse_atom_feed(&xml).unwrap();

        // Paper 1: quant-ph + cond-mat.str-el (crossover)
        assert!(papers[0].is_crossover());
        assert_eq!(papers[0].categories.len(), 2);

        // Paper 3: quant-ph only (not crossover)
        assert!(!papers[2].is_crossover());
    }

    #[test]
    fn test_parse_empty_feed() {
        let config = ArxivMonitorConfig::default();
        let monitor = ArxivMonitor::new(config);
        let xml = r#"<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <title>ArXiv Query</title>
</feed>"#;

        let papers = monitor.parse_atom_feed(xml).unwrap();
        assert!(papers.is_empty());
    }

    #[test]
    fn test_parse_empty_string() {
        let config = ArxivMonitorConfig::default();
        let monitor = ArxivMonitor::new(config);

        let result = monitor.parse_atom_feed("");
        assert!(result.is_err());
        match result {
            Err(ArxivError::ParseError(msg)) => assert!(msg.contains("Empty")),
            _ => panic!("Expected ParseError"),
        }
    }

    #[test]
    fn test_parse_malformed_entry_skipped() {
        let config = ArxivMonitorConfig::default();
        let monitor = ArxivMonitor::new(config);
        // Entry missing <id> tag should be skipped
        let xml = r#"<feed>
  <entry>
    <title>No ID paper</title>
    <summary>Missing id</summary>
  </entry>
  <entry>
    <id>http://arxiv.org/abs/2401.99999v1</id>
    <title>Good Paper</title>
    <summary>Has id</summary>
    <published>2024-01-20T00:00:00Z</published>
    <updated>2024-01-20T00:00:00Z</updated>
    <author><name>Test Author</name></author>
    <category term="quant-ph"/>
  </entry>
</feed>"#;

        let papers = monitor.parse_atom_feed(xml).unwrap();
        assert_eq!(papers.len(), 1);
        assert_eq!(papers[0].id, "2401.99999");
    }

    // ============================================================
    // RELEVANCE SCORING TESTS
    // ============================================================

    #[test]
    fn test_high_relevance_paper() {
        let scorer = RelevanceScorer::quantum_default();
        let paper = make_test_paper(
            "2401.00001",
            "Improved Surface Code Decoders Using Stabilizer Tensor Networks",
            "We present fault-tolerant quantum error correction using lattice surgery \
             on a surface code with magic state distillation.",
            &["quant-ph", "cond-mat.str-el"],
        );

        let score = scorer.score(&paper);
        // Should be high: many keyword hits, good categories, crossover
        assert!(score > 0.8, "High relevance paper score = {:.3}", score);
    }

    #[test]
    fn test_low_relevance_paper() {
        let scorer = RelevanceScorer::quantum_default();
        let paper = make_test_paper(
            "2401.00002",
            "Optimal Transport in Classical Fluid Dynamics",
            "We study the Navier-Stokes equations using variational methods \
             for incompressible fluid flow.",
            &["physics.flu-dyn"],
        );

        let score = scorer.score(&paper);
        // Should be low: no quantum keywords, irrelevant category
        assert!(score < 0.3, "Low relevance paper score = {:.3}", score);
    }

    #[test]
    fn test_medium_relevance_paper() {
        let scorer = RelevanceScorer::quantum_default();
        let paper = make_test_paper(
            "2401.00003",
            "Quantum Machine Learning with Noisy Qubits",
            "We explore variational quantum algorithms on noisy intermediate-scale \
             quantum devices for classification tasks.",
            &["quant-ph"],
        );

        let score = scorer.score(&paper);
        // Should be moderate: some keywords, right category
        assert!(
            score > 0.3 && score < 1.0,
            "Medium relevance = {:.3}",
            score
        );
    }

    #[test]
    fn test_scoring_rank_order() {
        let scorer = RelevanceScorer::quantum_default();
        let papers = vec![
            make_test_paper(
                "low",
                "Classical Optimization",
                "No quantum content here.",
                &["math.OC"],
            ),
            make_test_paper(
                "high",
                "Surface Code Fault-Tolerant Quantum Error Correction",
                "Stabilizer codes with lattice surgery and magic state distillation.",
                &["quant-ph", "cond-mat.str-el"],
            ),
            make_test_paper(
                "mid",
                "Quantum Noise Models",
                "Noise characterization for quantum computing.",
                &["quant-ph"],
            ),
        ];

        let ranked = scorer.rank(&papers);
        assert_eq!(ranked[0].0.id, "high");
        assert_eq!(ranked[2].0.id, "low");
    }

    // ============================================================
    // TREND DETECTION TESTS
    // ============================================================

    #[test]
    fn test_detect_emerging_trend() {
        let config = ArxivMonitorConfig::builder()
            .add_category("quant-ph")
            .trend_growth_threshold(0.3)
            .build();
        let monitor = ArxivMonitor::new(config);

        // Older papers: no floquet mentions
        let mut papers: Vec<ArxivPaper> = Vec::new();
        for i in 0..5 {
            papers.push(make_test_paper(
                &format!("old_{}", i),
                "Quantum Error Correction Review",
                "Standard surface code techniques for quantum computing.",
                &["quant-ph"],
            ));
        }
        // Recent papers: floquet mentions (emerging)
        for i in 0..5 {
            papers.push(make_test_paper(
                &format!("new_{}", i),
                &format!("Floquet Code Advances {}", i),
                "New floquet code constructions with improved thresholds.",
                &["quant-ph"],
            ));
        }

        let trends = monitor.detect_trends(&papers);
        let floquet_trend = trends.iter().find(|t| t.topic == "floquet");
        assert!(floquet_trend.is_some(), "Should detect floquet trend");
        let ft = floquet_trend.unwrap();
        assert!(ft.is_emerging, "Floquet should be emerging");
        assert!(ft.recent_count > ft.previous_count);
    }

    #[test]
    fn test_detect_stable_trend() {
        let config = ArxivMonitorConfig::builder()
            .add_category("quant-ph")
            .trend_growth_threshold(0.5)
            .build();
        let monitor = ArxivMonitor::new(config);

        // Equal mentions in both halves -> stable
        let mut papers: Vec<ArxivPaper> = Vec::new();
        for i in 0..10 {
            papers.push(make_test_paper(
                &format!("p_{}", i),
                "Surface Code Analysis",
                "Surface code decoder improvements.",
                &["quant-ph"],
            ));
        }

        let trends = monitor.detect_trends(&papers);
        let sc_trend = trends.iter().find(|t| t.topic == "surface code");
        assert!(sc_trend.is_some());
        assert!(
            !sc_trend.unwrap().is_emerging,
            "Equal distribution should be stable"
        );
    }

    // ============================================================
    // DIGEST GENERATION TESTS
    // ============================================================

    #[test]
    fn test_generate_digest() {
        let config = ArxivMonitorConfig::builder()
            .add_category("quant-ph")
            .add_keyword("surface code", 0.9)
            .relevance_threshold(0.0) // Include all
            .build();
        let monitor = ArxivMonitor::new(config);
        let xml = sample_atom_feed();
        let papers = monitor.parse_atom_feed(&xml).unwrap();

        let digest = monitor.generate_digest(&papers);
        assert_eq!(digest.total_papers, 4);
        assert!(!digest.top_papers.is_empty());
        assert!(!digest.category_distribution.is_empty());
        assert!(digest.category_distribution.contains_key("quant-ph"));
    }

    #[test]
    fn test_digest_display() {
        let config = ArxivMonitorConfig::builder()
            .add_category("quant-ph")
            .relevance_threshold(0.0)
            .build();
        let monitor = ArxivMonitor::new(config);
        let xml = sample_atom_feed();
        let papers = monitor.parse_atom_feed(&xml).unwrap();

        let digest = monitor.generate_digest(&papers);
        let display = format!("{}", digest);
        assert!(display.contains("ArXiv Research Digest"));
        assert!(display.contains("Total papers analyzed:"));
        assert!(display.contains("Category Distribution"));
    }

    // ============================================================
    // CATEGORY FILTERING TESTS
    // ============================================================

    #[test]
    fn test_filter_by_relevance_threshold() {
        let config = ArxivMonitorConfig::builder()
            .add_category("quant-ph")
            .add_keyword("surface code", 0.9)
            .build();
        let monitor = ArxivMonitor::new(config);

        let papers = vec![
            make_test_paper(
                "high",
                "Surface Code Fault-Tolerant QEC",
                "Surface code with stabilizer and lattice surgery.",
                &["quant-ph"],
            ),
            make_test_paper(
                "low",
                "Cooking Recipes Online",
                "No quantum content whatsoever.",
                &["cs.IR"],
            ),
        ];

        let filtered = monitor.filter_by_relevance(&papers, 0.5);
        assert!(!filtered.is_empty());
        // The high-relevance paper should pass; the cooking one should not
        assert!(
            filtered.iter().all(|(p, _)| p.id == "high"),
            "Only the high-relevance paper should pass threshold 0.5"
        );
    }

    #[test]
    fn test_find_crossover_papers() {
        let config = ArxivMonitorConfig::default();
        let monitor = ArxivMonitor::new(config);

        let papers = vec![
            make_test_paper("cross", "Cross", "Cross", &["quant-ph", "cond-mat.str-el"]),
            make_test_paper("single", "Single", "Single", &["quant-ph"]),
        ];

        let crossovers = monitor.find_crossover_papers(&papers);
        assert_eq!(crossovers.len(), 1);
        assert_eq!(crossovers[0].id, "cross");
    }

    // ============================================================
    // DATE HANDLING TESTS
    // ============================================================

    #[test]
    fn test_paper_year_extraction() {
        let paper = make_test_paper("test", "Test", "Test", &["quant-ph"]);
        assert_eq!(paper.year(), Some(2024));
    }

    #[test]
    fn test_paper_published_ymd() {
        let paper = make_test_paper("test", "Test", "Test", &["quant-ph"]);
        assert_eq!(paper.published_ymd(), Some((2024, 1, 15)));
    }

    #[test]
    fn test_approx_days_between_same_month() {
        let d1 = (2024, 1, 1);
        let d2 = (2024, 1, 15);
        let days = approx_days_between(d1, d2);
        assert_eq!(days, 14);
    }

    #[test]
    fn test_approx_days_between_different_month() {
        let d1 = (2024, 1, 1);
        let d3 = (2024, 2, 1);
        let days2 = approx_days_between(d1, d3);
        assert_eq!(days2, 30); // Using 30-day months
    }

    #[test]
    fn test_date_parsing_short_string() {
        assert_eq!(parse_date_ymd("short"), None);
        assert_eq!(parse_date_ymd(""), None);
    }

    // ============================================================
    // DISPLAY TRAIT TESTS
    // ============================================================

    #[test]
    fn test_paper_display_few_authors() {
        let paper = make_test_paper("2401.00001", "Test Paper", "Abstract", &["quant-ph"]);
        let display = format!("{}", paper);
        assert!(display.contains("2401.00001"));
        assert!(display.contains("Test Paper"));
        assert!(display.contains("Test Author"));
    }

    #[test]
    fn test_paper_display_many_authors() {
        let mut paper = make_test_paper("2401.00001", "Test Paper", "Abstract", &["quant-ph"]);
        paper.authors = vec![
            "Alice".to_string(),
            "Bob".to_string(),
            "Charlie".to_string(),
            "Diana".to_string(),
        ];
        let display = format!("{}", paper);
        assert!(display.contains("Alice et al."));
    }

    #[test]
    fn test_error_display() {
        let e = ArxivError::ParseError("bad xml".to_string());
        assert_eq!(format!("{}", e), "XML parse error: bad xml");

        let e = ArxivError::ConfigError("bad config".to_string());
        assert_eq!(format!("{}", e), "Configuration error: bad config");

        let e = ArxivError::DateError("bad date".to_string());
        assert_eq!(format!("{}", e), "Date error: bad date");

        let e = ArxivError::EmptyResults;
        assert_eq!(format!("{}", e), "No papers matched the query");
    }

    #[test]
    fn test_monitor_display() {
        let config = ArxivMonitorConfig::builder()
            .add_category("quant-ph")
            .add_keyword("test", 0.5)
            .max_results(25)
            .days_lookback(14)
            .build();
        let monitor = ArxivMonitor::new(config);
        let display = format!("{}", monitor);
        assert!(display.contains("ArxivMonitor"));
        assert!(display.contains("quant-ph"));
        assert!(display.contains("25"));
        assert!(display.contains("14d"));
    }

    #[test]
    fn test_sort_order_display() {
        assert_eq!(format!("{}", SortOrder::Relevance), "relevance");
        assert_eq!(format!("{}", SortOrder::SubmittedDate), "submittedDate");
        assert_eq!(
            format!("{}", SortOrder::LastUpdatedDate),
            "lastUpdatedDate"
        );
    }

    #[test]
    fn test_trend_display() {
        let trend = ResearchTrend {
            topic: "surface code".to_string(),
            recent_count: 15,
            previous_count: 5,
            growth_rate: 2.0,
            is_emerging: true,
            contributing_papers: vec!["p1".to_string(), "p2".to_string()],
        };
        let display = format!("{}", trend);
        assert!(display.contains("surface code"));
        assert!(display.contains("EMERGING"));
        assert!(display.contains("200.0%"));
    }

    // ============================================================
    // KEYWORD SET TESTS
    // ============================================================

    #[test]
    fn test_predefined_keyword_sets() {
        let qec = QuantumKeywordSets::error_correction();
        assert!(qec.len() >= 8);
        assert!(qec.iter().any(|(k, _)| k == "surface code"));

        let sim = QuantumKeywordSets::simulation();
        assert!(sim.len() >= 5);
        assert!(sim.iter().any(|(k, _)| k == "tensor network"));

        let hw = QuantumKeywordSets::hardware();
        assert!(hw.len() >= 5);
        assert!(hw.iter().any(|(k, _)| k == "trapped ion"));

        let alg = QuantumKeywordSets::algorithms();
        assert!(alg.len() >= 5);
        assert!(alg.iter().any(|(k, _)| k == "qaoa"));
    }

    // ============================================================
    // GROWTH RATE CALCULATION
    // ============================================================

    #[test]
    fn test_growth_rate_computation() {
        // Doubling: 10 from 5 -> growth 1.0 (100%)
        assert!((ResearchTrend::compute_growth(10, 5) - 1.0).abs() < f64::EPSILON);

        // No change: 5 from 5 -> growth 0.0
        assert!((ResearchTrend::compute_growth(5, 5) - 0.0).abs() < f64::EPSILON);

        // From zero: 5 from 0 -> growth 5.0 (treats zero as 1)
        assert!((ResearchTrend::compute_growth(5, 0) - 5.0).abs() < f64::EPSILON);

        // Decline: 2 from 10 -> growth -0.8 (-80%)
        assert!((ResearchTrend::compute_growth(2, 10) - (-0.8)).abs() < f64::EPSILON);
    }

    // ============================================================
    // XML HELPER TESTS
    // ============================================================

    #[test]
    fn test_xml_entity_decoding() {
        assert_eq!(decode_xml_entities("a &amp; b"), "a & b");
        assert_eq!(decode_xml_entities("&lt;tag&gt;"), "<tag>");
        assert_eq!(decode_xml_entities("&quot;hi&quot;"), "\"hi\"");
    }

    #[test]
    fn test_extract_tag_simple() {
        let xml = "<title>Hello World</title>";
        assert_eq!(extract_tag(xml, "title"), Some("Hello World"));
    }

    #[test]
    fn test_extract_tag_with_attributes() {
        let xml = r#"<title type="html">Hello</title>"#;
        assert_eq!(extract_tag(xml, "title"), Some("Hello"));
    }

    #[test]
    fn test_extract_tag_missing() {
        let xml = "<other>data</other>";
        assert_eq!(extract_tag(xml, "title"), None);
    }

    #[test]
    fn test_extract_all_tags() {
        let xml = "<root><name>Alice</name><name>Bob</name><name>Charlie</name></root>";
        let names = extract_all_tags(xml, "name");
        assert_eq!(names, vec!["Alice", "Bob", "Charlie"]);
    }
}
