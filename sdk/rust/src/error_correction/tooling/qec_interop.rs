//! QEC interoperability helpers (Stim-like detector model export).

use crate::dynamic_surface_code::DynamicSurfaceCode;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DetectorNode {
    pub id: usize,
    pub round: usize,
    pub x: usize,
    pub y: usize,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ErrorTerm {
    pub probability: f64,
    pub detectors: Vec<usize>,
    pub observables: Vec<usize>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct StimLikeDetectorModel {
    pub distance: usize,
    pub rounds: usize,
    pub detectors: Vec<DetectorNode>,
    pub error_terms: Vec<ErrorTerm>,
    pub logical_observable: Vec<usize>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct DetectorModelConfig {
    pub rounds: usize,
    pub data_error_rate: f64,
    pub measurement_error_rate: f64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct MatchingGraphEdge {
    pub detector_a: usize,
    pub detector_b: Option<usize>,
    pub weight: f64,
    pub probability: f64,
    pub observables: Vec<usize>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct MatchingGraph {
    pub num_detectors: usize,
    pub edges: Vec<MatchingGraphEdge>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct MatchingGraphConfig {
    /// Skip terms with probability below this threshold.
    pub min_probability: f64,
    /// Clamp probability into [eps, 1-eps] before weight conversion.
    pub clamp_epsilon: f64,
    /// If true, approximate hyperedges using pairwise edges from the first detector.
    pub expand_hyperedges: bool,
}

impl Default for MatchingGraphConfig {
    fn default() -> Self {
        Self {
            min_probability: 0.0,
            clamp_epsilon: 1e-12,
            expand_hyperedges: false,
        }
    }
}

impl Default for DetectorModelConfig {
    fn default() -> Self {
        Self {
            rounds: 4,
            data_error_rate: 1e-3,
            measurement_error_rate: 5e-4,
        }
    }
}

impl StimLikeDetectorModel {
    /// Render a simple Stim-like text form for interop and debugging.
    ///
    /// This intentionally uses a lightweight, readable representation and is
    /// not a complete Stim language emitter.
    pub fn to_text(&self) -> String {
        let mut out = String::new();
        out.push_str("# nQPU-Metal Stim-like detector model\n");
        out.push_str(&format!(
            "# distance={} rounds={} detectors={} terms={}\n",
            self.distance,
            self.rounds,
            self.detectors.len(),
            self.error_terms.len()
        ));

        for d in &self.detectors {
            out.push_str(&format!(
                "detector D{} r={} x={} y={}\n",
                d.id, d.round, d.x, d.y
            ));
        }

        for term in &self.error_terms {
            let mut toks = term
                .detectors
                .iter()
                .map(|d| format!("D{}", d))
                .collect::<Vec<_>>();
            toks.extend(term.observables.iter().map(|o| format!("L{}", o)));
            out.push_str(&format!(
                "error({:.8}) {}\n",
                term.probability,
                toks.join(" ")
            ));
        }

        if !self.logical_observable.is_empty() {
            let obs = self
                .logical_observable
                .iter()
                .map(|d| format!("D{}", d))
                .collect::<Vec<_>>()
                .join(" ");
            out.push_str(&format!("logical_observable L0 {}\n", obs));
        }

        out
    }

    /// Parse a model previously exported with `to_text()`.
    pub fn from_text(text: &str) -> Result<Self, String> {
        let mut distance = None;
        let mut rounds = None;
        let mut detectors: Vec<DetectorNode> = Vec::new();
        let mut error_terms: Vec<ErrorTerm> = Vec::new();
        let mut logical_observable: Vec<usize> = Vec::new();

        for (line_no, raw) in text.lines().enumerate() {
            let line = raw.trim();
            if line.is_empty() {
                continue;
            }

            if line.starts_with('#') {
                if line.contains("distance=") && line.contains("rounds=") {
                    for tok in line.split_whitespace() {
                        if let Some(v) = tok.strip_prefix("distance=") {
                            distance = v.parse::<usize>().ok();
                        } else if let Some(v) = tok.strip_prefix("rounds=") {
                            rounds = v.parse::<usize>().ok();
                        }
                    }
                }
                continue;
            }

            if let Some(rest) = line.strip_prefix("detector ") {
                let mut id = None;
                let mut round = None;
                let mut x = None;
                let mut y = None;

                for tok in rest.split_whitespace() {
                    if let Some(v) = tok.strip_prefix('D') {
                        id =
                            Some(v.parse::<usize>().map_err(|_| {
                                format!("invalid detector id on line {}", line_no + 1)
                            })?);
                    } else if let Some(v) = tok.strip_prefix("r=") {
                        round = Some(
                            v.parse::<usize>()
                                .map_err(|_| format!("invalid round on line {}", line_no + 1))?,
                        );
                    } else if let Some(v) = tok.strip_prefix("x=") {
                        x = Some(
                            v.parse::<usize>()
                                .map_err(|_| format!("invalid x coord on line {}", line_no + 1))?,
                        );
                    } else if let Some(v) = tok.strip_prefix("y=") {
                        y = Some(
                            v.parse::<usize>()
                                .map_err(|_| format!("invalid y coord on line {}", line_no + 1))?,
                        );
                    }
                }

                detectors.push(DetectorNode {
                    id: id.ok_or_else(|| format!("missing detector id on line {}", line_no + 1))?,
                    round: round
                        .ok_or_else(|| format!("missing detector round on line {}", line_no + 1))?,
                    x: x.ok_or_else(|| format!("missing detector x on line {}", line_no + 1))?,
                    y: y.ok_or_else(|| format!("missing detector y on line {}", line_no + 1))?,
                });
                continue;
            }

            if line.starts_with("error(") {
                let close = line
                    .find(')')
                    .ok_or_else(|| format!("malformed error term on line {}", line_no + 1))?;
                let prob_str = &line["error(".len()..close];
                let prob = prob_str
                    .parse::<f64>()
                    .map_err(|_| format!("invalid probability on line {}", line_no + 1))?;
                let mut dets = Vec::new();
                let mut obs = Vec::new();
                let rest = line[close + 1..].trim();
                for tok in rest.split_whitespace() {
                    if tok.starts_with('D') {
                        dets.push(
                            parse_detector_ref(tok)
                                .map_err(|e| format!("{} on line {}", e, line_no + 1))?,
                        );
                    } else if tok.starts_with('L') {
                        obs.push(
                            parse_observable_ref(tok)
                                .map_err(|e| format!("{} on line {}", e, line_no + 1))?,
                        );
                    } else {
                        return Err(format!(
                            "unsupported token '{}' in error term on line {}",
                            tok,
                            line_no + 1
                        ));
                    }
                }
                error_terms.push(ErrorTerm {
                    probability: prob,
                    detectors: dets,
                    observables: obs,
                });
                continue;
            }

            if let Some(rest) = line.strip_prefix("logical_observable ") {
                for tok in rest.split_whitespace().skip(1) {
                    logical_observable.push(
                        parse_detector_ref(tok)
                            .map_err(|e| format!("{} on line {}", e, line_no + 1))?,
                    );
                }
                continue;
            }

            return Err(format!("unrecognized model line {}: {}", line_no + 1, line));
        }

        if detectors.is_empty() {
            let max_id = error_terms
                .iter()
                .flat_map(|e| e.detectors.iter().copied())
                .chain(logical_observable.iter().copied())
                .max();
            if let Some(m) = max_id {
                for id in 0..=m {
                    detectors.push(DetectorNode {
                        id,
                        round: 0,
                        x: 0,
                        y: 0,
                    });
                }
            }
        }

        let inferred_rounds = detectors.iter().map(|d| d.round).max().map_or(0, |r| r + 1);
        Ok(StimLikeDetectorModel {
            distance: distance.unwrap_or(0),
            rounds: rounds.unwrap_or(inferred_rounds),
            detectors,
            error_terms,
            logical_observable,
        })
    }

    /// Convert detector error model terms into a MWPM-style matching graph.
    ///
    /// Edges touching one detector are emitted as boundary edges (`detector_b=None`).
    /// Two-detector terms are emitted directly.
    /// Hyperedges (3+ detectors) require `expand_hyperedges=true` for a star approximation.
    pub fn to_matching_graph(&self, config: &MatchingGraphConfig) -> Result<MatchingGraph, String> {
        if config.clamp_epsilon <= 0.0 || config.clamp_epsilon >= 0.5 {
            return Err("clamp_epsilon must be in (0, 0.5)".to_string());
        }
        if !(0.0..=1.0).contains(&config.min_probability) {
            return Err("min_probability must be in [0, 1]".to_string());
        }

        let num_detectors = self
            .detectors
            .iter()
            .map(|d| d.id)
            .max()
            .map_or(0, |m| m + 1);

        let mut edges = Vec::new();
        for term in &self.error_terms {
            if term.probability < config.min_probability {
                continue;
            }
            if !(0.0..=1.0).contains(&term.probability) {
                return Err(format!(
                    "invalid error probability {}, expected [0,1]",
                    term.probability
                ));
            }

            let mut dets = term.detectors.clone();
            dets.sort_unstable();
            dets.dedup();

            if dets.is_empty() {
                continue;
            }
            if dets.iter().any(|&d| d >= num_detectors) {
                return Err("error term references unknown detector id".to_string());
            }

            let weight = log_odds_weight(term.probability, config.clamp_epsilon);
            if dets.len() == 1 {
                edges.push(MatchingGraphEdge {
                    detector_a: dets[0],
                    detector_b: None,
                    weight,
                    probability: term.probability,
                    observables: term.observables.clone(),
                });
            } else if dets.len() == 2 {
                edges.push(MatchingGraphEdge {
                    detector_a: dets[0],
                    detector_b: Some(dets[1]),
                    weight,
                    probability: term.probability,
                    observables: term.observables.clone(),
                });
            } else if config.expand_hyperedges {
                let anchor = dets[0];
                for &other in &dets[1..] {
                    edges.push(MatchingGraphEdge {
                        detector_a: anchor,
                        detector_b: Some(other),
                        weight,
                        probability: term.probability,
                        observables: term.observables.clone(),
                    });
                }
            } else {
                return Err(format!(
                    "hyperedge term with {} detectors requires expand_hyperedges=true",
                    dets.len()
                ));
            }
        }

        Ok(MatchingGraph {
            num_detectors,
            edges,
        })
    }
}

/// Build a Stim-like detector model for a distance-`d` surface code.
pub fn build_stim_like_surface_code_model(
    distance: usize,
    config: &DetectorModelConfig,
) -> Result<StimLikeDetectorModel, String> {
    if distance < 3 || distance.is_multiple_of(2) {
        return Err("distance must be odd and >= 3".to_string());
    }
    if config.rounds == 0 {
        return Err("rounds must be >= 1".to_string());
    }
    if !(0.0..=1.0).contains(&config.data_error_rate)
        || !(0.0..=1.0).contains(&config.measurement_error_rate)
    {
        return Err("error rates must be in [0, 1]".to_string());
    }

    let checks_per_side = distance - 1;
    let checks_per_round = checks_per_side * checks_per_side;

    let mut detectors = Vec::with_capacity(config.rounds * checks_per_round);
    for round in 0..config.rounds {
        for y in 0..checks_per_side {
            for x in 0..checks_per_side {
                detectors.push(DetectorNode {
                    id: detectors.len(),
                    round,
                    x,
                    y,
                });
            }
        }
    }

    let mut error_terms = Vec::new();

    // Data-error contribution on each detector in each round.
    for d in &detectors {
        error_terms.push(ErrorTerm {
            probability: config.data_error_rate,
            detectors: vec![d.id],
            observables: vec![],
        });
    }

    // Measurement-error style temporal links between same-position checks.
    for round in 0..config.rounds.saturating_sub(1) {
        for y in 0..checks_per_side {
            for x in 0..checks_per_side {
                let a = idx(round, x, y, checks_per_side);
                let b = idx(round + 1, x, y, checks_per_side);
                error_terms.push(ErrorTerm {
                    probability: config.measurement_error_rate,
                    detectors: vec![a, b],
                    observables: vec![],
                });
            }
        }
    }

    // Simple boundary logical-observable proxy: last-round top row.
    let mut logical_observable = Vec::new();
    let last_round = config.rounds - 1;
    for x in 0..checks_per_side {
        logical_observable.push(idx(last_round, x, 0, checks_per_side));
    }

    Ok(StimLikeDetectorModel {
        distance,
        rounds: config.rounds,
        detectors,
        error_terms,
        logical_observable,
    })
}

/// Build a Stim-like detector model from dynamic-surface-code runtime params.
pub fn build_stim_like_from_dynamic_code(
    code: &DynamicSurfaceCode,
    rounds: usize,
) -> Result<StimLikeDetectorModel, String> {
    let cfg = DetectorModelConfig {
        rounds,
        data_error_rate: code.physical_error_rate,
        measurement_error_rate: (code.physical_error_rate * 0.5).min(1.0),
    };
    build_stim_like_surface_code_model(code.distance, &cfg)
}

/// Parse Stim-like detector model text into structured representation.
pub fn parse_stim_like_detector_model(text: &str) -> Result<StimLikeDetectorModel, String> {
    StimLikeDetectorModel::from_text(text)
}

/// Convert a detector model into a MWPM-style matching graph.
pub fn build_matching_graph(
    model: &StimLikeDetectorModel,
    config: &MatchingGraphConfig,
) -> Result<MatchingGraph, String> {
    model.to_matching_graph(config)
}

fn idx(round: usize, x: usize, y: usize, checks_per_side: usize) -> usize {
    round * checks_per_side * checks_per_side + y * checks_per_side + x
}

fn parse_detector_ref(tok: &str) -> Result<usize, String> {
    tok.strip_prefix('D')
        .ok_or_else(|| format!("expected detector token, got '{}'", tok))
        .and_then(|v| {
            v.parse::<usize>()
                .map_err(|_| format!("invalid detector id '{}'", tok))
        })
}

fn parse_observable_ref(tok: &str) -> Result<usize, String> {
    tok.strip_prefix('L')
        .ok_or_else(|| format!("expected observable token, got '{}'", tok))
        .and_then(|v| {
            v.parse::<usize>()
                .map_err(|_| format!("invalid observable id '{}'", tok))
        })
}

fn log_odds_weight(probability: f64, epsilon: f64) -> f64 {
    let p = probability.clamp(epsilon, 1.0 - epsilon);
    ((1.0 - p) / p).ln()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_model_detector_count() {
        let cfg = DetectorModelConfig {
            rounds: 2,
            data_error_rate: 1e-3,
            measurement_error_rate: 5e-4,
        };
        let model = build_stim_like_surface_code_model(5, &cfg).expect("model");
        // (d-1)^2 * rounds = 4*4*2 = 32
        assert_eq!(model.detectors.len(), 32);
    }

    #[test]
    fn test_text_export_contains_terms() {
        let model =
            build_stim_like_surface_code_model(3, &DetectorModelConfig::default()).expect("model");
        let text = model.to_text();
        assert!(text.contains("error("));
        assert!(text.contains("logical_observable L0"));
    }

    #[test]
    fn test_build_from_dynamic_code() {
        let code = DynamicSurfaceCode::new(7).with_error_rate(0.02);
        let model = build_stim_like_from_dynamic_code(&code, 3).expect("dynamic model");
        assert_eq!(model.distance, 7);
        assert_eq!(model.rounds, 3);
    }

    #[test]
    fn test_roundtrip_text_parse() {
        let src =
            build_stim_like_surface_code_model(5, &DetectorModelConfig::default()).expect("model");
        let txt = src.to_text();
        let parsed = parse_stim_like_detector_model(&txt).expect("parse");
        assert_eq!(parsed.distance, src.distance);
        assert_eq!(parsed.rounds, src.rounds);
        assert_eq!(parsed.detectors.len(), src.detectors.len());
        assert_eq!(parsed.error_terms.len(), src.error_terms.len());
    }

    #[test]
    fn test_parse_rejects_bad_line() {
        let text = "garbage line";
        let err = parse_stim_like_detector_model(text).expect_err("expected parse error");
        assert!(err.contains("unrecognized"));
    }

    #[test]
    fn test_parse_error_observable_tokens() {
        let text = "\
# nQPU-Metal Stim-like detector model
# distance=3 rounds=1 detectors=1 terms=1
detector D0 r=0 x=0 y=0
error(0.01000000) D0 L0 L2
logical_observable L0 D0
";
        let parsed = parse_stim_like_detector_model(text).expect("parse");
        assert_eq!(parsed.error_terms.len(), 1);
        assert_eq!(parsed.error_terms[0].detectors, vec![0]);
        assert_eq!(parsed.error_terms[0].observables, vec![0, 2]);
    }

    #[test]
    fn test_matching_graph_boundary_and_pair_edges() {
        let model = StimLikeDetectorModel {
            distance: 3,
            rounds: 1,
            detectors: vec![
                DetectorNode {
                    id: 0,
                    round: 0,
                    x: 0,
                    y: 0,
                },
                DetectorNode {
                    id: 1,
                    round: 0,
                    x: 1,
                    y: 0,
                },
            ],
            error_terms: vec![
                ErrorTerm {
                    probability: 0.01,
                    detectors: vec![0],
                    observables: vec![],
                },
                ErrorTerm {
                    probability: 0.02,
                    detectors: vec![0, 1],
                    observables: vec![0],
                },
            ],
            logical_observable: vec![0],
        };

        let g = model
            .to_matching_graph(&MatchingGraphConfig::default())
            .expect("matching graph");
        assert_eq!(g.num_detectors, 2);
        assert_eq!(g.edges.len(), 2);
        assert_eq!(g.edges[0].detector_b, None);
        assert_eq!(g.edges[1].detector_b, Some(1));
    }

    #[test]
    fn test_matching_graph_rejects_hyperedge_without_expand() {
        let model = StimLikeDetectorModel {
            distance: 3,
            rounds: 1,
            detectors: vec![
                DetectorNode {
                    id: 0,
                    round: 0,
                    x: 0,
                    y: 0,
                },
                DetectorNode {
                    id: 1,
                    round: 0,
                    x: 1,
                    y: 0,
                },
                DetectorNode {
                    id: 2,
                    round: 0,
                    x: 2,
                    y: 0,
                },
            ],
            error_terms: vec![ErrorTerm {
                probability: 0.01,
                detectors: vec![0, 1, 2],
                observables: vec![],
            }],
            logical_observable: vec![],
        };

        let err = model
            .to_matching_graph(&MatchingGraphConfig::default())
            .expect_err("hyperedge should error");
        assert!(err.contains("hyperedge"));
    }

    #[test]
    fn test_matching_graph_expands_hyperedge() {
        let model = StimLikeDetectorModel {
            distance: 3,
            rounds: 1,
            detectors: vec![
                DetectorNode {
                    id: 0,
                    round: 0,
                    x: 0,
                    y: 0,
                },
                DetectorNode {
                    id: 1,
                    round: 0,
                    x: 1,
                    y: 0,
                },
                DetectorNode {
                    id: 2,
                    round: 0,
                    x: 2,
                    y: 0,
                },
            ],
            error_terms: vec![ErrorTerm {
                probability: 0.01,
                detectors: vec![0, 1, 2],
                observables: vec![1],
            }],
            logical_observable: vec![],
        };

        let cfg = MatchingGraphConfig {
            expand_hyperedges: true,
            ..MatchingGraphConfig::default()
        };
        let g = model.to_matching_graph(&cfg).expect("matching graph");
        assert_eq!(g.edges.len(), 2);
        assert_eq!(g.edges[0].detector_a, 0);
        assert_eq!(g.edges[0].detector_b, Some(1));
        assert_eq!(g.edges[1].detector_b, Some(2));
        assert_eq!(g.edges[0].observables, vec![1]);
    }
}
