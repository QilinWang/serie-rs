use crate::draw_protocol::DrawRef;
use crate::numeric_frame::{NumericColumn, NumericFrame, NumericFrameError};
use crate::stage_noise::{NoiseRecipe, NoisyBundle, SourceDescriptor};

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// bundle transitions (stage fn change)
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// partition(bundle, config): NoisyBundle -> PartitionedBundle
// PartitionConfig/PartitionStats methods: internal helpers (no bundle boundary change)

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// partition
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// partition structs

/// Partition fractions — where each split starts and ends in [0, 1].
/// Python: PartitionSpec
#[derive(Debug, Clone, PartialEq)]
pub struct PartitionConfig {
    pub t0: f64, // train start
    pub t1: f64, // train end
    pub v0: f64, // val start
    pub v1: f64, // val end
    pub s0: f64, // test start
    pub s1: f64, // test end
}

impl PartitionConfig {
    pub fn new(t0: f64, t1: f64, v0: f64, v1: f64, s0: f64, s1: f64) -> Self {
        // Python: __attrs_post_init__ validation
        assert!(0.0 <= t0 && t0 < t1 && t1 <= 1.0, "Train bounds invalid: {t0}-{t1}");
        assert!(t1 <= v0 && v0 < v1 && v1 <= 1.0, "Val bounds invalid: {v0}-{v1}");
        assert!(v1 <= s0 && s0 < s1 && s1 <= 1.0, "Test bounds invalid: {s0}-{s1}");
        Self { t0, t1, v0, v1, s0, s1 }
    }

    // ── Presets ──

    pub fn p70_10_20() -> Self {
        Self::new(0.0, 0.7, 0.7, 0.8, 0.8, 1.0)
    }
    pub fn p70_20_10() -> Self {
        Self::new(0.0, 0.7, 0.7, 0.9, 0.9, 1.0)
    }
    pub fn p60_20_20() -> Self {
        Self::new(0.0, 0.6, 0.6, 0.8, 0.8, 1.0)
    }
    pub fn p50_25_25() -> Self {
        Self::new(0.0, 0.5, 0.5, 0.75, 0.75, 1.0)
    }
    pub fn p60_15_25() -> Self {
        Self::new(0.0, 0.6, 0.6, 0.75, 0.75, 1.0)
    }
    pub fn p60_10_30() -> Self {
        Self::new(0.0, 0.6, 0.6, 0.7, 0.7, 1.0)
    }
    pub fn p50_10_40() -> Self {
        Self::new(0.0, 0.5, 0.5, 0.6, 0.6, 1.0)
    }
    pub fn p70_gap_10_20() -> Self {
        Self::new(0.0, 0.7, 0.71, 0.81, 0.82, 1.0)
    }

    /// Convert a fraction pair to row indices given total row count.
    /// Python: def idx(self, a, b, n) -> (int, int)
    fn to_indices(&self, a: f64, b: f64, n: usize) -> (usize, usize) {
        ((n as f64 * a) as usize, (n as f64 * b) as usize)
    }

    /// Compute all six indices (train_start, train_end, val_start, val_end, test_start, test_end).
    /// Python: the body of SubStagePartition.apply
    pub fn partition_indices(&self, n_rows: usize) -> PartitionIndices {
        let (t0, t1) = self.to_indices(self.t0, self.t1, n_rows);
        let (v0, v1) = self.to_indices(self.v0, self.v1, n_rows);
        let (s0, s1) = self.to_indices(self.s0, self.s1, n_rows);
        PartitionIndices { t0, t1, v0, v1, s0, s1 }
    }
}

/// Materialized partition boundaries — row indices into the data.
/// Python: db.indices = (t0_idx, t1_idx, v0_idx, v1_idx, s0_idx, s1_idx)
#[derive(Debug, Clone, PartialEq)]
pub struct PartitionIndices {
    pub t0: usize, // train start row
    pub t1: usize, // train end row
    pub v0: usize, // val start row
    pub v1: usize, // val end row
    pub s0: usize, // test start row
    pub s1: usize, // test end row
}

impl PartitionIndices {
    pub fn train_count(&self) -> usize {
        self.t1 - self.t0
    }
    pub fn val_count(&self) -> usize {
        self.v1 - self.v0
    }
    pub fn test_count(&self) -> usize {
        self.s1 - self.s0
    }
}

#[derive(Debug, Clone)]
pub struct PartitionedBundle {
    pub source_descriptor: SourceDescriptor,
    pub frame: NumericFrame,
    pub feature_names: Vec<String>,
    pub channels: usize,
    pub sigma: f64,
    pub noise_seed: u64,
    pub noise_draw_ref: Option<DrawRef>,
    pub noise_recipe: NoiseRecipe,
    pub config: PartitionConfig,
    pub indices: PartitionIndices,
    pub train_stats: PartitionStats,
    pub val_stats: PartitionStats,
    pub test_stats: PartitionStats,
}

// partition details
/// What to subtract and divide by during normalization.
/// Python: NormalizationMethod in fr_specs.py
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum NormMethod {
    Standard,     // (mean, std)
    RobustIqr,    // (median, iqr)
    RobustMad,    // (median, mad * 1.4826)
    CenterMean,   // (mean, 1.0)
    CenterMedian, // (median, 1.0)
    None,         // (0.0, 1.0)
}

/// Per-channel statistics for one data split.
/// Computed directly from a NumericFrame slice — no external query engine.
#[derive(Debug, Clone, PartialEq)]
pub struct PartitionStats {
    pub count: usize,
    pub mean: Vec<f64>,   // [D]
    pub std: Vec<f64>,    // [D]
    pub median: Vec<f64>, // [D]
    pub mad: Vec<f64>,    // [D] — median absolute deviation
    pub min: Vec<f64>,    // [D]
    pub max: Vec<f64>,    // [D]
    pub q01: Vec<f64>,    // [D]
    pub q05: Vec<f64>,    // [D]
    pub q25: Vec<f64>,    // [D]
    pub q75: Vec<f64>,    // [D]
    pub q95: Vec<f64>,    // [D]
    pub q99: Vec<f64>,    // [D]
    // ── New: computed during from_frame_slice, used by health/drift ──
    pub iqr: Vec<f64>,       // q75 - q25, per channel
    pub zero_frac: Vec<f64>, // fraction of zeros per channel
}

impl PartitionStats {
    /// after stats is compute, further analysis function
    pub fn dim(&self) -> usize {
        self.mean.len()
    }

    // ── Health: derived from single split's stats ──

    /// Channels with heavy tails: (q99-q95)/iqr or (q05-q01)/iqr > threshold
    pub fn heavy_tail_channels(&self, thr: f64) -> Vec<usize> {
        (0..self.dim())
            .filter(|&i| {
                let denom = self.iqr[i].max(1e-8);
                let right = (self.q99[i] - self.q95[i]) / denom;
                let left = (self.q05[i] - self.q01[i]) / denom;
                right > thr || left > thr
            })
            .collect()
    }

    /// Channels with near-zero variance
    pub fn const_channels(&self) -> Vec<usize> {
        (0..self.dim()).filter(|&i| self.std[i] < 1e-6).collect()
    }

    /// Channels where fraction of zeros exceeds threshold
    pub fn sparse_channels(&self, thr: f64) -> Vec<usize> {
        (0..self.dim()).filter(|&i| self.zero_frac[i] > thr).collect()
    }

    // ── Drift: compare this split against a reference (train) ──

    /// Mean shift normalized by reference std: avg(|self.mean - ref.mean| / ref.std)
    pub fn mean_shift_from(&self, reference: &PartitionStats) -> f64 {
        let dim = self.dim() as f64;
        self.mean
            .iter()
            .zip(&reference.mean)
            .zip(&reference.std)
            .map(|((self_mean, ref_mean), ref_std)| (self_mean - ref_mean).abs() / ref_std.max(1e-8))
            .sum::<f64>()
            / dim
    }

    /// Median of (self.std / reference.std) across channels.
    /// Values near 1.0 = similar variance. Far from 1.0 = drift.
    /// Python: float(np.median(v.std / np.maximum(t.std, 1e-8)))
    pub fn std_ratio_from(&self, reference: &PartitionStats) -> f64 {
        // self = val or test stats, reference = train stats
        // For each channel: how much bigger/smaller is this split's std vs train's?
        let mut ratios: Vec<f64> = self
            .std
            .iter()
            .zip(&reference.std)
            .map(|(self_std, ref_std)| {
                // e.g. val_std=2.0, train_std=1.0 → ratio=2.0 (val is 2x noisier)
                // ref_std.max(1e-8) prevents division by zero for constant channels
                self_std / ref_std.max(1e-8)
            })
            .collect();

        // Sort for median computation.
        // partial_cmp because f64 has NaN (which can't be ordered).
        // .unwrap() is safe here — we clamped denom above so no NaN produced.
        ratios.sort_by(|a, b| a.partial_cmp(b).unwrap());

        // Median: middle value if odd count, average of two middle if even.
        // Python: np.median(ratios)
        let n = ratios.len();
        if n % 2 == 0 {
            (ratios[n / 2 - 1] + ratios[n / 2]) / 2.0
        } else {
            ratios[n / 2]
        }
    }

    /// Return (center, scale) per channel for the given normalization method.
    /// Python: PartitionStats.center_and_spread
    pub fn center_and_spread(&self, method: NormMethod) -> (Vec<f64>, Vec<f64>) {
        let ones = vec![1.0; self.dim()];
        let zeros = vec![0.0; self.dim()];

        match method {
            NormMethod::Standard => (self.mean.clone(), self.std.clone()),
            NormMethod::RobustIqr => (self.median.clone(), self.iqr.clone()),
            NormMethod::RobustMad => {
                // MAD * 1.4826 ≈ std for normal distributions
                let scaled: Vec<f64> = self.mad.iter().map(|m| m * 1.4826).collect();
                (self.median.clone(), scaled)
            }
            NormMethod::CenterMean => (self.mean.clone(), ones),
            NormMethod::CenterMedian => (self.median.clone(), ones),
            NormMethod::None => (zeros, ones),
        }
    }

    /// Compute stats for rows [start..end] of the NumericFrame.
    /// Python: SubStageComputePartitionStats._get_stats
    pub fn from_frame_slice(frame: &NumericFrame, start: usize, end: usize) -> Result<Self, NumericFrameError> {
        let slice = frame.slice(start, end - start)?;
        let count = slice.height();
        let columns = frame.column_names();
        let mut mean = Vec::with_capacity(columns.len());
        let mut std = Vec::with_capacity(columns.len());
        let mut median = Vec::with_capacity(columns.len());
        let mut mad = Vec::with_capacity(columns.len());
        let mut min = Vec::with_capacity(columns.len());
        let mut max = Vec::with_capacity(columns.len());
        let mut q01 = Vec::with_capacity(columns.len());
        let mut q05 = Vec::with_capacity(columns.len());
        let mut q25 = Vec::with_capacity(columns.len());
        let mut q75 = Vec::with_capacity(columns.len());
        let mut q95 = Vec::with_capacity(columns.len());
        let mut q99 = Vec::with_capacity(columns.len());
        let mut zero_frac = Vec::with_capacity(columns.len());

        // NumericColumn owns the eager aggregations used by every split. Keeping
        // them together makes the statistical contract explicit and testable.
        for column_name in &columns {
            let values = slice.column(column_name.as_str())?.numeric()?;
            let column_median = values.median().unwrap_or(0.0);
            mean.push(values.mean().unwrap_or(0.0));
            std.push(values.sample_std().unwrap_or(0.0));
            median.push(column_median);
            min.push(values.min().unwrap_or(0.0));
            max.push(values.max().unwrap_or(0.0));
            q01.push(values.quantile_linear(0.01).unwrap_or(0.0));
            q05.push(values.quantile_linear(0.05).unwrap_or(0.0));
            q25.push(values.quantile_linear(0.25).unwrap_or(0.0));
            q75.push(values.quantile_linear(0.75).unwrap_or(0.0));
            q95.push(values.quantile_linear(0.95).unwrap_or(0.0));
            q99.push(values.quantile_linear(0.99).unwrap_or(0.0));

            let deviations: Vec<f64> = values
                .into_iter()
                .flatten()
                .map(|value| (value - column_median).abs())
                .collect();
            mad.push(NumericColumn::new(String::new(), deviations).median().unwrap_or(0.0));

            let denominator = values.len() as f64;
            let zero_count = values
                .into_iter()
                .filter(|value| matches!(value, Some(value) if value.abs() < 1e-10))
                .count() as f64;
            zero_frac.push(if denominator == 0.0 {
                0.0
            } else {
                zero_count / denominator
            });
        }

        let iqr: Vec<f64> = q75.iter().zip(&q25).map(|(a, b)| a - b).collect();

        Ok(Self {
            count,
            mean,
            std,
            median,
            mad,
            min,
            max,
            q01,
            q05,
            q25,
            q75,
            q95,
            q99,
            iqr,
            zero_frac,
        })
    }
}

pub fn partition(bundle: NoisyBundle, config: PartitionConfig) -> Result<PartitionedBundle, NumericFrameError> {
    let n_rows = bundle.frame.height();
    let indices = config.partition_indices(n_rows);

    let train_stats = PartitionStats::from_frame_slice(&bundle.frame, indices.t0, indices.t1)?;
    let val_stats = PartitionStats::from_frame_slice(&bundle.frame, indices.v0, indices.v1)?;
    let test_stats = PartitionStats::from_frame_slice(&bundle.frame, indices.s0, indices.s1)?;

    Ok(PartitionedBundle {
        source_descriptor: bundle.source_descriptor,
        frame: bundle.frame,
        feature_names: bundle.feature_names,
        channels: bundle.channels,
        sigma: bundle.sigma,
        noise_seed: bundle.noise_seed,
        noise_draw_ref: bundle.noise_draw_ref,
        noise_recipe: bundle.recipe,
        config,
        indices,
        train_stats,
        val_stats,
        test_stats,
    })
}

/// Tests for the partition stage: split boundaries, statistics, and leakage prevention.
///
/// Why these tests exist:
/// - Train statistics must come from the train slice ONLY. If val or test data
///   leaks into the mean/std computation, the model sees future information
///   during normalization — this is the most common data leakage bug in time
///   series forecasting and silently inflates all metrics.
/// - Preset indices must cover the full series with no gaps and no overlaps.
///   A gap means some data is silently dropped; an overlap means some data
///   appears in both train and test, inflating accuracy.
/// - Health signals (constant channels, zero fractions, heavy tails) detect
///   pathological data before training — catching these early saves hours of
///   wasted GPU time on degenerate inputs.
/// - center_and_spread tests pin the normalization formulas: Standard uses
///   mean/std, Robust uses median/MAD*1.4826, None is identity (0, 1).
#[cfg(test)]
mod tests {
    use super::*;
    use crate::stage_clean::KnownMeta;
    use crate::stage_noise::{NoiseRecipe, SourceDescriptor};

    fn make_tiny_frame() -> NumericFrame {
        let n = 100;
        let hufl: Vec<f64> = (0..n).map(|i| 5.0 + 3.0 * (i as f64 * 0.1).sin()).collect();

        let mut hull: Vec<f64> = (0..n).map(|i| 8.0 + 2.0 * (i as f64 * 0.05).cos()).collect();
        for v in hull.iter_mut().take(25).skip(20) {
            *v = 0.0;
        }

        let mut mufl: Vec<f64> = (0..n).map(|i| 12.0 + 4.0 * (i as f64 * 0.08).sin()).collect();
        for v in mufl.iter_mut().take(53).skip(50) {
            *v = 88.297_996_52;
        }

        let ot: Vec<f64> = (0..n).map(|i| 20.0 + 6.0 * (i as f64 * 0.03).sin()).collect();

        NumericFrame::new(
            n,
            vec![
                NumericColumn::new("HUFL".into(), &hufl),
                NumericColumn::new("HULL".into(), &hull),
                NumericColumn::new("MUFL".into(), &mufl),
                NumericColumn::new("OT".into(), &ot),
            ],
        )
        .unwrap()
    }

    fn make_health_frame() -> NumericFrame {
        let n = 200;
        let ch0: Vec<f64> = (0..n).map(|i| 5.0 + (i as f64 * 0.1).sin()).collect();
        let ch1: Vec<f64> = vec![3.14; n];
        let mut ch2: Vec<f64> = vec![0.0; n];
        for i in (0..n).step_by(5) {
            ch2[i] = 1.0 + (i as f64 * 0.01);
        }
        let mut ch3: Vec<f64> = (0..n).map(|i| i as f64 / n as f64).collect();
        for (i, v) in ch3.iter_mut().enumerate().take(10) {
            *v = -100.0 - i as f64;
        }
        for (i, v) in ch3.iter_mut().enumerate().skip(n - 10) {
            *v = 100.0 + i as f64;
        }

        NumericFrame::new(
            n,
            vec![
                NumericColumn::new("normal".into(), &ch0),
                NumericColumn::new("constant".into(), &ch1),
                NumericColumn::new("sparse".into(), &ch2),
                NumericColumn::new("heavy".into(), &ch3),
            ],
        )
        .unwrap()
    }

    fn noisy_bundle_from(frame: NumericFrame) -> NoisyBundle {
        let feature_names = frame.column_names();
        let channels = frame.width();
        NoisyBundle {
            source_descriptor: SourceDescriptor::Known {
                known_meta: KnownMeta::etth1(),
                source_tag: "etth1__drop_and_impute".to_string(),
            },
            frame,
            feature_names,
            channels,
            sigma: 0.1,
            noise_seed: 42,
            noise_draw_ref: None,
            recipe: NoiseRecipe::Uniform,
        }
    }

    #[test]
    fn partition_p70_10_20_on_1000() {
        let idx = PartitionConfig::p70_10_20().partition_indices(1000);
        assert_eq!((idx.t0, idx.t1), (0, 700));
        assert_eq!((idx.v0, idx.v1), (700, 800));
        assert_eq!((idx.s0, idx.s1), (800, 1000));
        assert_eq!(idx.train_count() + idx.val_count() + idx.test_count(), 1000);
    }

    #[test]
    fn partition_gap_preset_has_gaps() {
        let idx = PartitionConfig::p70_gap_10_20().partition_indices(10_000);
        assert!(idx.v0 > idx.t1);
        assert!(idx.s0 > idx.v1);
    }

    #[test]
    #[should_panic(expected = "Train bounds")]
    fn partition_rejects_bad_train_bounds() {
        let _ = PartitionConfig::new(0.5, 0.3, 0.5, 0.7, 0.7, 1.0);
    }

    #[test]
    #[should_panic(expected = "Val bounds")]
    fn partition_rejects_overlap_val_bounds() {
        let _ = PartitionConfig::new(0.0, 0.7, 0.5, 0.8, 0.8, 1.0);
    }

    #[test]
    #[should_panic(expected = "Test bounds")]
    fn partition_rejects_test_over_one() {
        let _ = PartitionConfig::new(0.0, 0.5, 0.5, 0.7, 0.7, 1.1);
    }

    #[test]
    fn stats_basic_shapes() {
        let s = PartitionStats::from_frame_slice(&make_tiny_frame(), 0, 70).unwrap();
        assert_eq!(s.count, 70);
        assert_eq!(s.dim(), 4);
        assert!(s.std.iter().all(|&v| v > 0.0));
        assert!(s.iqr.iter().all(|&v| v >= 0.0));
    }

    #[test]
    fn stats_detects_zero_fraction() {
        let s = PartitionStats::from_frame_slice(&make_tiny_frame(), 0, 70).unwrap();
        assert!(s.zero_frac[1] > 0.0);
    }

    #[test]
    fn stats_different_ranges_have_different_means() {
        let full = PartitionStats::from_frame_slice(&make_tiny_frame(), 0, 100).unwrap();
        let half = PartitionStats::from_frame_slice(&make_tiny_frame(), 0, 50).unwrap();
        assert!((full.mean[0] - half.mean[0]).abs() > 0.01);
    }

    #[test]
    fn center_and_spread_standard_matches_mean_std() {
        let s = PartitionStats::from_frame_slice(&make_tiny_frame(), 0, 100).unwrap();
        let (c, sc) = s.center_and_spread(NormMethod::Standard);
        assert_eq!(c, s.mean);
        assert_eq!(sc, s.std);
    }

    #[test]
    fn center_and_spread_none_is_identity() {
        let s = PartitionStats::from_frame_slice(&make_tiny_frame(), 0, 100).unwrap();
        let (c, sc) = s.center_and_spread(NormMethod::None);
        assert!(c.iter().all(|&v| v == 0.0));
        assert!(sc.iter().all(|&v| v == 1.0));
    }

    #[test]
    fn center_and_spread_robust_mad_scales_by_14826() {
        let s = PartitionStats::from_frame_slice(&make_tiny_frame(), 0, 100).unwrap();
        let (_, sc) = s.center_and_spread(NormMethod::RobustMad);
        for (scaled, raw) in sc.iter().zip(s.mad.iter()) {
            assert!((*scaled - *raw * 1.4826).abs() < 1e-10);
        }
    }

    #[test]
    fn health_signals_detect_pathologies() {
        let s = PartitionStats::from_frame_slice(&make_health_frame(), 0, 200).unwrap();
        assert!(s.const_channels().contains(&1));
        assert!(s.sparse_channels(0.3).contains(&2));
        assert!(s.heavy_tail_channels(0.1).contains(&3));
    }

    #[test]
    fn drift_self_comparison_is_zero_and_one() {
        let s = PartitionStats::from_frame_slice(&make_tiny_frame(), 0, 70).unwrap();
        assert!((s.mean_shift_from(&s)).abs() < 1e-10);
        assert!((s.std_ratio_from(&s) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn partition_stage_populates_indices_and_stats() {
        let bundle = noisy_bundle_from(make_tiny_frame());
        let out = partition(bundle, PartitionConfig::p70_10_20()).unwrap();
        assert_eq!(
            out.indices.train_count() + out.indices.val_count() + out.indices.test_count(),
            out.frame.height()
        );
        assert_eq!(out.train_stats.dim(), out.channels);
        assert_eq!(out.val_stats.dim(), out.channels);
        assert_eq!(out.test_stats.dim(), out.channels);
    }

    // ── Gap tests (mirroring Python test_contracts.py) ────────────────

    #[test]
    fn train_stats_use_train_slice_only() {
        // Train-only stats must differ from full-series stats when the data
        // has different distributions across splits. This is the no-leakage contract.
        let bundle = noisy_bundle_from(make_tiny_frame());
        let out = partition(bundle, PartitionConfig::p70_10_20()).unwrap();
        let full = PartitionStats::from_frame_slice(&out.frame, 0, out.frame.height()).unwrap();
        // Train mean and full mean should differ (sinusoidal data drifts across splits)
        let drift = (out.train_stats.mean[0] - full.mean[0]).abs();
        assert!(
            drift > 0.001,
            "train stats should differ from full stats, drift={drift}"
        );
    }

    #[test]
    fn all_preset_indices_cover_full_series() {
        let presets = [
            PartitionConfig::p70_10_20(),
            PartitionConfig::p70_20_10(),
            PartitionConfig::p60_20_20(),
            PartitionConfig::p50_25_25(),
            PartitionConfig::p60_15_25(),
            PartitionConfig::p60_10_30(),
            PartitionConfig::p50_10_40(),
        ];
        for config in &presets {
            let idx = config.partition_indices(1000);
            let total = idx.train_count() + idx.val_count() + idx.test_count();
            assert_eq!(total, 1000, "preset {:?} doesn't cover full series", config);
        }
    }

    #[test]
    fn partition_indices_are_non_overlapping() {
        let idx = PartitionConfig::p70_20_10().partition_indices(1000);
        assert!(idx.t1 <= idx.v0, "train and val overlap");
        assert!(idx.v1 <= idx.s0, "val and test overlap");
    }

    #[test]
    fn gap_partition_has_non_adjacent_splits() {
        let idx = PartitionConfig::p70_gap_10_20().partition_indices(10_000);
        assert!(idx.v0 > idx.t1, "gap partition should have gap between train and val");
        assert!(idx.s0 > idx.v1, "gap partition should have gap between val and test");
        // Total covered is less than 10_000 because of gaps
        let covered = idx.train_count() + idx.val_count() + idx.test_count();
        assert!(covered < 10_000);
    }

    #[test]
    fn feature_names_survive_partition() {
        let bundle = noisy_bundle_from(make_tiny_frame());
        let names_before: Vec<String> = bundle.feature_names.iter().map(|n| n.to_string()).collect();
        let out = partition(bundle, PartitionConfig::p70_10_20()).unwrap();
        let names_after: Vec<String> = out.feature_names.iter().map(|n| n.to_string()).collect();
        assert_eq!(names_before, names_after);
    }
}
