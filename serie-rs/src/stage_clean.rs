#[cfg(feature = "pipeline")]
use crate::numeric_frame::{NumericColumn, NumericFrame, NumericFrameError};
#[cfg(feature = "pipeline")]
use crate::stage_load::{LoadConfig, LoadedBundle};

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// bundle transitions (stage fn change)
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// clean(bundle, config): LoadedBundle -> CleanedBundle
// detect_sentinel_runs + known-meta helpers: internal details

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// clean
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// clean structs
#[derive(Debug, Clone)]
#[cfg(feature = "pipeline")]
pub struct CleanedBundle {
    pub known_meta: KnownMeta,
    pub source_tag: String,
    pub frame: NumericFrame,
    pub feature_names: Vec<String>,
    pub channels: usize,
}

// clean stage fn
#[cfg(feature = "pipeline")]
pub fn clean(bundle: LoadedBundle, config: &LoadConfig) -> Result<CleanedBundle, NumericFrameError> {
    // Stage 2: scan sentinels, drop bad columns, impute survivors
    let mut frame = bundle.frame;
    s02_02_clean_columns(&mut frame, config)?;

    let _feature_names = frame.column_names();
    println!("   Features after clean: {:?}", _feature_names);

    Ok(CleanedBundle {
        feature_names: _feature_names,
        channels: frame.width(),
        known_meta: bundle.known_meta,
        source_tag: bundle.source_tag,
        frame,
    })
}

// clean details
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// clean columns
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
/// Scan sentinels once. Drop bad columns, impute the survivors.
/// Both flags in LoadConfig: skip_drop_bad, skip_impute.
#[cfg(feature = "pipeline")]
pub fn s02_02_clean_columns(frame: &mut NumericFrame, config: &LoadConfig) -> Result<(), NumericFrameError> {
    if config.skip_drop_bad && config.skip_impute {
        return Ok(());
    }

    let columns: Vec<String> = frame.column_names();
    let n_rows = frame.height();
    let threshold_steps = (config.drop_col_hour_threshold as usize) * (config.known_meta.steps_per_hour() as usize);
    let mut dropped: Vec<String> = Vec::new();
    let mut sentinel_masks: Vec<(String, Vec<bool>)> = Vec::new();

    // ── Single scan: compute drop statistics on the RAW frame ──
    // CHG (2026-07-02, parity fix): Python's order is stats → drop → mask →
    // impute. Keeping the masks beside their column names preserves that order
    // in the eager numeric pass.
    for c in &columns {
        let sentinels = config.known_meta.sentinels_for(c);
        if sentinels.is_empty() {
            continue;
        }

        let values: Vec<Option<f64>> = frame.column(c.as_str())?.numeric()?.into_iter().collect();
        let (sentinel_flags, run_lengths) = detect_sentinel_runs(&values, &sentinels, config.value_tol);
        let sentinel_fraction = if n_rows == 0 {
            0.0
        } else {
            sentinel_flags.iter().filter(|flag| **flag).count() as f64 / n_rows as f64
        };
        let max_run = run_lengths.iter().copied().max().unwrap_or(0) as usize;

        if !config.skip_drop_bad && (sentinel_fraction >= config.drop_col_threshold || max_run >= threshold_steps) {
            println!(
                "🧊 Dropping column: {} (frac={:.3}, max_run={})",
                c, sentinel_fraction, max_run
            );
            dropped.push(c.to_string());
        } else if !config.skip_impute {
            sentinel_masks.push((c.clone(), sentinel_flags));
        }
    }

    // ── Drop bad columns ──
    if !config.skip_drop_bad {
        if !dropped.is_empty() {
            println!("🧊 Dropping {} columns total", dropped.len());
            *frame = frame.drop_columns(&dropped);
        } else {
            println!("No col dropping due to sentinels!")
        }
    }

    // ── Mask and impute the SURVIVING columns ──
    if !config.skip_impute {
        let mut imputed = 0usize;
        for (column_name, sentinel_flags) in sentinel_masks {
            let mut values: Vec<Option<f64>> = frame.column(column_name.as_str())?.numeric()?.into_iter().collect();
            for (value, is_sentinel) in values.iter_mut().zip(sentinel_flags) {
                if is_sentinel {
                    *value = None;
                }
            }
            impute_values(&mut values, config.known_meta.impute_method, column_name.as_str())?;
            let imputed_column = NumericColumn::new(column_name.clone(), values);
            frame.replace(column_name.as_str(), imputed_column)?;
            imputed += 1;
        }
        println!("🩹 Imputed short runs on {} columns.", imputed);
    }

    Ok(())
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// sentinel detection
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#[cfg(feature = "pipeline")]
fn detect_sentinel_runs(values: &[Option<f64>], sentinels: &[f64], value_tol: f64) -> (Vec<bool>, Vec<u32>) {
    let flags: Vec<bool> = values
        .iter()
        .map(|value| value.is_some_and(|value| sentinels.iter().any(|sentinel| (value - sentinel).abs() <= value_tol)))
        .collect();
    let mut run_lengths = vec![0u32; flags.len()];
    let mut start = 0usize;
    while start < flags.len() {
        if !flags[start] {
            start += 1;
            continue;
        }
        let mut end = start + 1;
        while end < flags.len() && flags[end] {
            end += 1;
        }
        run_lengths[start..end].fill((end - start) as u32);
        start = end;
    }
    (flags, run_lengths)
}

#[cfg(feature = "pipeline")]
fn impute_values(values: &mut [Option<f64>], method: ImputeMethod, column_name: &str) -> Result<(), NumericFrameError> {
    let present: Vec<usize> = values
        .iter()
        .enumerate()
        .filter_map(|(index, value)| value.map(|_| index))
        .collect();
    let Some(&first) = present.first() else {
        return Err(NumericFrameError::new(format!(
            "known-data column {column_name:?} has no non-null value to impute from"
        )));
    };
    let last = *present.last().expect("first present index exists");
    let first_value = values[first].expect("present index has a value");
    values[..first].fill(Some(first_value));
    for pair in present.windows(2) {
        let left = pair[0];
        let right = pair[1];
        if right == left + 1 {
            continue;
        }
        let left_value = values[left].expect("present index has a value");
        let right_value = values[right].expect("present index has a value");
        for index in (left + 1)..right {
            values[index] = Some(match method {
                ImputeMethod::Linear => {
                    let step = (right_value - left_value) / (right - left) as f64;
                    left_value + step * (index - left) as f64
                }
                ImputeMethod::Nearest => {
                    if index - left <= right - index {
                        left_value
                    } else {
                        right_value
                    }
                }
            });
        }
    }
    let last_value = values[last].expect("present index has a value");
    values[(last + 1)..].fill(Some(last_value));
    Ok(())
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// known data id
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
/// Default fraction of sentinel observations that removes a known-data column.
pub const DEFAULT_DROP_COL_THRESHOLD: f64 = 0.2;
/// Default absolute tolerance for matching one declared sentinel value.
pub const DEFAULT_SENTINEL_VALUE_TOL: f64 = 1e-6;
/// Default longest sentinel run, in hours, that removes a column.
pub const DEFAULT_DROP_COL_HOUR_THRESHOLD: u32 = 168;

// ── The dataset ID enum ──
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum KnownDataId {
    EttH1,
    EttH2,
    EttM1,
    EttM2,
    Weather,
}

impl KnownDataId {
    pub fn tag(&self) -> &'static str {
        match self {
            Self::EttH1 => "etth1",
            Self::EttH2 => "etth2",
            Self::EttM1 => "ettm1",
            Self::EttM2 => "ettm2",
            Self::Weather => "weather",
        }
    }
}

// ── The per-column sentinel policy (this replaces exempt_cols) ──
// Remember: the whole point was to kill the blacklist+exemption pattern.
// Now each column can say "I have no sentinels" directly.
#[derive(Debug, Clone, PartialEq)]
pub enum SentinelPolicy {
    /// Use the dataset-wide universal_sentinels list
    Inherit,
    /// This column has its own suspicious values
    Custom(Vec<f64>),
    /// This column is clean — skip sentinel detection entirely
    /// (this is what exempt_cols used to mean)
    None,
}
use std::collections::HashMap;
use std::path::PathBuf;

// ── Impute method: same idea, just an enum ──
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ImputeMethod {
    Nearest,
    Linear,
}

pub enum Sentinel {
    Universal(Vec<f64>),
    NumericColumn(String, Vec<f64>),
}

// ── KnownMeta itself ──
// In Rust, a frozen attrs class is just a struct.
// No "frozen" keyword needed — Rust structs are immutable by default
// unless you hold a &mut reference.
#[derive(Debug, Clone, PartialEq)]
pub struct KnownMeta {
    pub dataset_id: KnownDataId,
    pub path: PathBuf,
    sampling_seconds: u32,           // REQUIRE: > 0
    pub date_column: Option<String>, // None = no date column to drop

    // Sentinel config
    universal_sentinels: Vec<f64>,
    // per-column overrides: missing key → Inherit (use universal)
    // present key → Custom([...]) or None (exempt)
    column_sentinels: HashMap<String, SentinelPolicy>,

    impute_method: ImputeMethod,
}

#[allow(dead_code)]
impl KnownMeta {
    // This is your @property. In Rust, it's just a method.
    // &self means "I'm borrowing this struct read-only."
    // -> u32 means "I return an unsigned 32-bit integer."
    pub fn steps_per_hour(&self) -> u32 {
        3600 / self.sampling_seconds
    }
    /// Interpolation method for imputing masked sentinel runs (Python
    /// `KnownMeta.impute_method`). Getter because the field is module-private;
    /// the dgp_bank known-data generator reuses this cleaning metadata.
    pub fn impute_method(&self) -> ImputeMethod {
        self.impute_method
    }
    // This is your @classmethod constructor. In Rust it's called
    // an "associated function" — no &self parameter, called as
    // KnownMeta::etth1() instead of some_instance.etth1().
    pub fn etth1() -> Self {
        // Self = KnownMeta (the type we're impl'ing)
        Self {
            dataset_id: KnownDataId::EttH1,
            path: PathBuf::from("./datasets/ett/ETTh1.csv"),
            sampling_seconds: 3600,
            universal_sentinels: vec![0.0],
            column_sentinels: HashMap::new(), // no overrides
            date_column: Some("date".into()), // .into() converts &str → String
            impute_method: ImputeMethod::Linear,
        }
    }
    pub fn etth2() -> Self {
        let mut col_sentinels = HashMap::new();
        // LULL and MUFL have their own suspicious values ON TOP of universal
        col_sentinels.insert("LULL".into(), SentinelPolicy::Custom(vec![-31.46199989, -28.15699959]));
        col_sentinels.insert("MUFL".into(), SentinelPolicy::Custom(vec![88.29799652]));

        Self {
            dataset_id: KnownDataId::EttH2,
            path: PathBuf::from("./datasets/ett/ETTh2.csv"),
            sampling_seconds: 3600,
            universal_sentinels: vec![0.0],
            column_sentinels: col_sentinels,
            date_column: Some("date".into()),
            impute_method: ImputeMethod::Linear,
        }
    }
    pub fn ettm1() -> Self {
        Self {
            dataset_id: KnownDataId::EttM1,
            path: PathBuf::from("./datasets/ett/ETTm1.csv"),
            sampling_seconds: 900,
            universal_sentinels: vec![0.0],
            column_sentinels: HashMap::new(), // no overrides
            date_column: Some("date".into()), // .into() converts &str → String
            impute_method: ImputeMethod::Linear,
        }
    }
    pub fn ettm2() -> Self {
        let mut col_sentinels = HashMap::new();
        // LULL and MUFL have their own suspicious values ON TOP of universal
        col_sentinels.insert("LULL".into(), SentinelPolicy::Custom(vec![-31.46199989, -28.15699959]));
        col_sentinels.insert("MUFL".into(), SentinelPolicy::Custom(vec![88.29799652]));
        Self {
            dataset_id: KnownDataId::EttM2,
            path: PathBuf::from("./datasets/ett/ETTm2.csv"),
            sampling_seconds: 900,
            universal_sentinels: vec![0.0],
            column_sentinels: col_sentinels,
            date_column: Some("date".into()),
            impute_method: ImputeMethod::Linear,
        }
    }
    pub fn weather() -> Self {
        Self {
            dataset_id: KnownDataId::Weather,
            path: PathBuf::from("./datasets/weather/weather.csv"),
            sampling_seconds: 600,
            universal_sentinels: vec![0.0, -9999.0],
            column_sentinels: HashMap::new(), // no overrides
            date_column: Some("date".into()), // .into() converts &str → String
            impute_method: ImputeMethod::Linear,
        }
    }

    // Helper: what sentinels apply to a given column?
    // This is the logic that was split across three Python stages.
    pub fn sentinels_for(&self, column: &str) -> Vec<f64> {
        match self.column_sentinels.get(column) {
            Some(SentinelPolicy::Custom(extra)) => {
                let mut all = self.universal_sentinels.clone();
                all.extend(extra.iter().copied());
                all
            }
            Some(SentinelPolicy::None) => vec![],
            Some(SentinelPolicy::Inherit) | None => self.universal_sentinels.clone(),
        }
    }
}

/// Tests for the cleaning stage: sentinel detection, imputation, and column policy.
///
/// Why these tests exist:
/// - Known datasets (ETT, HuGaDB) contain sentinel values (zeros, -999) that
///   must be detected and imputed before noise injection, otherwise the noise
///   stage injects noise on top of garbage values and the train statistics
///   are polluted by sentinels.
/// - The cleaning stage must be a no-op when both flags are disabled — adding
///   a cleaning stage to a pipeline that doesn't need it must not change the data.
/// - Non-sentinel columns must pass through unchanged (isolation property).
#[cfg(all(test, feature = "pipeline"))]
mod tests {
    use super::*;
    use std::collections::HashMap;
    use std::path::PathBuf;

    fn make_tiny_ett() -> NumericFrame {
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

    #[test]
    fn sentinels_custom_merges_universal() {
        let s = KnownMeta::etth2().sentinels_for("MUFL");
        assert_eq!(s.len(), 2);
        assert!((s[0] - 0.0).abs() < 1e-10);
        assert!((s[1] - 88.297_996_52).abs() < 1e-6);
    }

    #[test]
    fn sentinels_none_returns_empty() {
        let mut column_sentinels = HashMap::new();
        column_sentinels.insert("rain".to_string(), SentinelPolicy::None);
        let meta = KnownMeta {
            dataset_id: KnownDataId::Weather,
            path: PathBuf::from("./test.csv"),
            sampling_seconds: 600,
            universal_sentinels: vec![0.0, -9999.0],
            column_sentinels,
            date_column: None,
            impute_method: ImputeMethod::Linear,
        };

        assert!(meta.sentinels_for("rain").is_empty());
    }

    #[test]
    fn detect_runs_finds_zero_blocks() {
        let data: Vec<f64> = vec![5.0, 0.0, 0.0, 0.0, 3.0, 7.0, 0.0, 0.0, 9.0];
        let meta = KnownMeta::etth1();
        let values: Vec<Option<f64>> = data.into_iter().map(Some).collect();
        let (_, runlen) = detect_sentinel_runs(&values, &meta.sentinels_for("x"), 1e-6);

        assert_eq!(runlen[0], 0);
        assert_eq!(runlen[1], 3);
        assert_eq!(runlen[2], 3);
        assert_eq!(runlen[3], 3);
        assert_eq!(runlen[6], 2);
        assert_eq!(runlen[7], 2);
    }

    #[test]
    fn detect_runs_multi_sentinel_column() {
        let data: Vec<f64> = vec![5.0, 0.0, 0.0, 88.297_996_52, 88.297_996_52, 3.0];
        let meta = KnownMeta::etth2();
        let values: Vec<Option<f64>> = data.into_iter().map(Some).collect();
        let (cond, _) = detect_sentinel_runs(&values, &meta.sentinels_for("MUFL"), 1e-6);

        assert!(!cond[0]);
        assert!(cond[1]);
        assert!(cond[3]);
        assert!(!cond[5]);
    }

    #[test]
    fn clean_imputes_hull_zero_run() {
        let mut frame = make_tiny_ett();
        let cfg = LoadConfig::new(KnownMeta::etth1());

        let before = frame.column("HULL").unwrap().numeric().unwrap().get(22).unwrap();
        assert!((before - 0.0).abs() < 1e-10);

        s02_02_clean_columns(&mut frame, &cfg).unwrap();

        let after = frame.column("HULL").unwrap().numeric().unwrap().get(22).unwrap();
        assert!(after.abs() > 0.01);
    }

    #[test]
    fn clean_drops_bad_column_even_with_imputation_enabled() {
        // Regression for the 2026-07-02 parity fix: masking used to run
        // before the drop stats, nulling the sentinels the stats were about
        // to count — so with imputation on (the default), the drop arm could
        // never fire. Stats must evaluate on the raw frame (Python's
        // StageDropBadColumns → StageImputeShortRuns order).
        let mut frame = make_tiny_ett();
        let n = frame.height();
        // Make HULL bad beyond doubt: 30% zeros ≥ the default frac threshold.
        let bad: Vec<f64> = (0..n)
            .map(|i| if i < 30 { 0.0 } else { 8.0 + (i as f64) * 0.01 })
            .collect();
        frame.replace("HULL", NumericColumn::new("HULL".into(), &bad)).unwrap();

        let cfg = LoadConfig::new(KnownMeta::etth1());
        assert!(
            !cfg.skip_impute,
            "default config must keep imputation on for this regression"
        );
        s02_02_clean_columns(&mut frame, &cfg).unwrap();

        assert!(
            frame.column("HULL").is_err(),
            "column with 30% sentinels must drop even when imputation is enabled"
        );
        // Survivors still get imputed: MUFL's short sentinel run is filled.
        let mufl = frame.column("MUFL").unwrap().numeric().unwrap();
        assert!(
            mufl.into_iter().all(|v| v.is_some()),
            "survivors must have no nulls after impute"
        );
    }

    #[test]
    fn clean_skip_both_is_noop() {
        let mut frame = make_tiny_ett();
        let before_shape = frame.shape();
        let cfg = LoadConfig {
            skip_impute: true,
            skip_drop_bad: true,
            ..LoadConfig::new(KnownMeta::etth1())
        };

        s02_02_clean_columns(&mut frame, &cfg).unwrap();
        assert_eq!(frame.shape(), before_shape);
    }

    #[test]
    fn clean_preserves_clean_column_values() {
        let mut frame = make_tiny_ett();
        let before: Vec<f64> = frame
            .column("HUFL")
            .unwrap()
            .numeric()
            .unwrap()
            .into_iter()
            .filter_map(|v| v)
            .collect();

        s02_02_clean_columns(&mut frame, &LoadConfig::new(KnownMeta::etth1())).unwrap();

        let after: Vec<f64> = frame
            .column("HUFL")
            .unwrap()
            .numeric()
            .unwrap()
            .into_iter()
            .filter_map(|v| v)
            .collect();

        for (a, b) in before.iter().zip(after.iter()) {
            assert!((a - b).abs() < 1e-12);
        }
    }
}
