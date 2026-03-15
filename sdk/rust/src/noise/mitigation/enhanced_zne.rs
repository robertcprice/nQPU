//! Enhanced zero-noise extrapolation (ZNE).

use crate::error_mitigation::{fold_gates_global, fold_gates_local};
use crate::gates::Gate;

/// Circuit folding strategy.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FoldingStrategy {
    Global,
    Local,
}

/// Extrapolation model.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ExtrapolationModel {
    Linear,
    Richardson,
    Exponential,
}

/// A noise-scaled observation point.
#[derive(Clone, Copy, Debug)]
pub struct ZnePoint {
    pub scale: f64,
    pub value: f64,
}

/// Enhanced ZNE runner.
#[derive(Clone, Debug)]
pub struct EnhancedZne {
    pub odd_scales: Vec<usize>,
    pub folding: FoldingStrategy,
    pub model: ExtrapolationModel,
}

impl Default for EnhancedZne {
    fn default() -> Self {
        Self {
            odd_scales: vec![1, 3, 5],
            folding: FoldingStrategy::Global,
            model: ExtrapolationModel::Richardson,
        }
    }
}

impl EnhancedZne {
    pub fn run<F>(&self, gates: &[Gate], mut evaluator: F) -> Result<(f64, Vec<ZnePoint>), String>
    where
        F: FnMut(&[Gate]) -> Result<f64, String>,
    {
        if self.odd_scales.is_empty() {
            return Err("odd_scales cannot be empty".to_string());
        }

        let mut points = Vec::with_capacity(self.odd_scales.len());
        for &s in &self.odd_scales {
            if s == 0 {
                return Err("scale factor must be >= 1".to_string());
            }

            let folded = match self.folding {
                FoldingStrategy::Global => fold_gates_global(gates, s),
                FoldingStrategy::Local => fold_gates_local(gates, s),
            };
            let value = evaluator(&folded)?;
            points.push(ZnePoint {
                scale: s as f64,
                value,
            });
        }

        let mitigated = match self.model {
            ExtrapolationModel::Linear => linear_extrapolate_zero(
                &points.iter().map(|p| p.scale).collect::<Vec<_>>(),
                &points.iter().map(|p| p.value).collect::<Vec<_>>(),
            ),
            ExtrapolationModel::Richardson => richardson_extrapolate_zero(
                &points.iter().map(|p| p.scale).collect::<Vec<_>>(),
                &points.iter().map(|p| p.value).collect::<Vec<_>>(),
            ),
            ExtrapolationModel::Exponential => exponential_extrapolate_zero(
                &points.iter().map(|p| p.scale).collect::<Vec<_>>(),
                &points.iter().map(|p| p.value).collect::<Vec<_>>(),
            ),
        };

        Ok((mitigated, points))
    }
}

/// Least-squares linear fit y = a x + b, returning b at x=0.
pub fn linear_extrapolate_zero(scales: &[f64], values: &[f64]) -> f64 {
    assert_eq!(scales.len(), values.len());
    if scales.is_empty() {
        return 0.0;
    }
    if scales.len() == 1 {
        return values[0];
    }

    let n = scales.len() as f64;
    let sx: f64 = scales.iter().sum();
    let sy: f64 = values.iter().sum();
    let sxx: f64 = scales.iter().map(|x| x * x).sum();
    let sxy: f64 = scales.iter().zip(values.iter()).map(|(x, y)| x * y).sum();

    let denom = n * sxx - sx * sx;
    if denom.abs() < 1e-12 {
        return values[0];
    }

    let a = (n * sxy - sx * sy) / denom;
    let b = (sy - a * sx) / n;
    b
}

/// Richardson extrapolation at x=0 for arbitrary distinct scales.
pub fn richardson_extrapolate_zero(scales: &[f64], values: &[f64]) -> f64 {
    assert_eq!(scales.len(), values.len());
    if scales.is_empty() {
        return 0.0;
    }
    if scales.len() == 1 {
        return values[0];
    }

    let mut estimate = 0.0;
    for i in 0..scales.len() {
        let mut w = 1.0;
        for j in 0..scales.len() {
            if i != j {
                let denom = scales[i] - scales[j];
                if denom.abs() < 1e-12 {
                    return linear_extrapolate_zero(scales, values);
                }
                w *= (-scales[j]) / denom;
            }
        }
        estimate += w * values[i];
    }
    estimate
}

/// Exponential fit y(x) = a * exp(b x) with optional shift if y <= 0 appears.
/// Returns y(0) = a.
pub fn exponential_extrapolate_zero(scales: &[f64], values: &[f64]) -> f64 {
    assert_eq!(scales.len(), values.len());
    if scales.is_empty() {
        return 0.0;
    }
    if scales.len() == 1 {
        return values[0];
    }

    let min_y = values
        .iter()
        .fold(f64::INFINITY, |acc, &v| if v < acc { v } else { acc });
    let shift = if min_y <= 0.0 { 1.0 - min_y } else { 0.0 };

    let mut xs = Vec::new();
    let mut ys = Vec::new();
    for (&x, &y) in scales.iter().zip(values.iter()) {
        let yp = y + shift;
        if yp > 0.0 {
            xs.push(x);
            ys.push(yp.ln());
        }
    }

    if xs.len() < 2 {
        return linear_extrapolate_zero(scales, values);
    }

    let n = xs.len() as f64;
    let sx: f64 = xs.iter().sum();
    let sy: f64 = ys.iter().sum();
    let sxx: f64 = xs.iter().map(|x| x * x).sum();
    let sxy: f64 = xs.iter().zip(ys.iter()).map(|(x, y)| x * y).sum();

    let denom = n * sxx - sx * sx;
    if denom.abs() < 1e-12 {
        return linear_extrapolate_zero(scales, values);
    }

    let b = (n * sxy - sx * sy) / denom;
    let ln_a = (sy - b * sx) / n;
    ln_a.exp() - shift
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_linear_extrapolation() {
        let x = [1.0, 3.0, 5.0];
        let y = [0.9, 0.7, 0.5]; // y = 1.0 - 0.1x
        let z = linear_extrapolate_zero(&x, &y);
        assert!((z - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_richardson_matches_quadratic_zero_limit() {
        let x = [1.0, 3.0, 5.0];
        // f(x)=1 + 0.2x + 0.05x^2 => f(0)=1
        let y = [1.25, 2.05, 3.25];
        let z = richardson_extrapolate_zero(&x, &y);
        assert!((z - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_exponential_extrapolation() {
        let x = [1.0, 3.0, 5.0];
        let y = [0.904837, 0.740818, 0.606531]; // exp(-0.1x)
        let z = exponential_extrapolate_zero(&x, &y);
        assert!((z - 1.0).abs() < 0.02);
    }

    #[test]
    fn test_enhanced_zne_runner() {
        let gates = vec![Gate::h(0), Gate::cnot(0, 1), Gate::rz(1, 0.3)];
        let zne = EnhancedZne::default();

        let (mitigated, points) = zne
            .run(&gates, |folded| {
                let noise = 0.01 * folded.len() as f64;
                Ok((1.0 - noise).max(0.0))
            })
            .expect("zne run");

        assert_eq!(points.len(), 3);
        assert!(mitigated > 0.8);
    }
}
