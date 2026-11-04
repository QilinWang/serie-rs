// The local numeric frame and CSV reader are the only tabular surfaces used by
// this legacy known-data pipeline.
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// bundle transitions (stage fn change)
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// load(config): LoadConfig -> LoadedBundle
// load_ready_bundle(load_config, noise_config, partition_config, ready_config): LoadConfig -> ReadyBundle
// s02_01/s02_03 helpers: internal details (no extra bundle boundary)
//
// flow (full known-data pipeline helper)
//   LoadConfig
//   -> load
//   -> clean
//   -> inject_noise
//   -> partition
//   -> ready
//   -> ReadyBundle
//
// flow (train-std-scaled noise helper)
//   LoadConfig
//   -> load
//   -> clean
//   -> partition clean data to compute train std
//   -> inject_noise with NoiseRecipe::Scaled { channel_std: train_std }
//   -> partition noisy data
//   -> ready
//   -> ReadyBundle

use crate::numeric_frame::{NumericColumn, NumericFrame, NumericFrameError};
use crate::stage_clean::{
    DEFAULT_DROP_COL_HOUR_THRESHOLD as KNOWN_DEFAULT_DROP_COL_HOUR_THRESHOLD,
    DEFAULT_DROP_COL_THRESHOLD as KNOWN_DEFAULT_DROP_COL_THRESHOLD,
    DEFAULT_SENTINEL_VALUE_TOL as KNOWN_DEFAULT_SENTINEL_VALUE_TOL, KnownMeta, clean,
};
use crate::stage_noise::{NoiseConfig, NoiseRecipe, inject_noise};
use crate::stage_partition::{PartitionConfig, partition};
use crate::stage_ready::{ReadyBundle, ReadyConfig, ready};

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// load
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// load structs

#[derive(Debug, Clone, PartialEq)]
pub struct LoadConfig {
    pub known_meta: KnownMeta,
    pub remove_cols: Vec<String>,
    pub remove_col_idx: Vec<usize>,
    pub skip_impute: bool,
    pub skip_drop_bad: bool,
    pub drop_col_threshold: f64,
    pub value_tol: f64,
    pub drop_col_hour_threshold: u32,
}

impl LoadConfig {
    const DEFAULT_DROP_COL_THRESHOLD: f64 = KNOWN_DEFAULT_DROP_COL_THRESHOLD;
    const DEFAULT_VALUE_TOL: f64 = KNOWN_DEFAULT_SENTINEL_VALUE_TOL;
    const DEFAULT_DROP_COL_HOUR_THRESHOLD: u32 = KNOWN_DEFAULT_DROP_COL_HOUR_THRESHOLD;

    pub fn new(known_meta: KnownMeta) -> Self {
        Self {
            known_meta,
            remove_cols: vec![],
            remove_col_idx: vec![],
            skip_impute: false,
            skip_drop_bad: false,
            drop_col_threshold: Self::DEFAULT_DROP_COL_THRESHOLD,
            value_tol: Self::DEFAULT_VALUE_TOL,
            drop_col_hour_threshold: Self::DEFAULT_DROP_COL_HOUR_THRESHOLD, // 24 * 7, 168 hours, one week
        }
    }

    pub fn cleaning_mode_tag(&self) -> &'static str {
        match (self.skip_drop_bad, self.skip_impute) {
            (true, true) => "no_clean",
            (false, true) => "drop_only",
            (true, false) => "impute_only",
            (false, false) => "drop_and_impute",
        }
    }

    pub fn tag(&self) -> String {
        let mut parts = vec![
            self.known_meta.dataset_id.tag().to_string(),
            self.cleaning_mode_tag().to_string(),
        ];

        if !self.remove_cols.is_empty() {
            let mut removed_columns = self.remove_cols.clone();
            removed_columns.sort();
            let removed_columns = removed_columns
                .iter()
                .map(|name| make_tag_safe_token(name))
                .collect::<Vec<_>>()
                .join("+");
            parts.push(format!("rmcols_{}", removed_columns));
        }

        if !self.remove_col_idx.is_empty() {
            let mut removed_column_indices = self.remove_col_idx.clone();
            removed_column_indices.sort_unstable();
            let removed_column_indices = removed_column_indices
                .iter()
                .map(|idx| idx.to_string())
                .collect::<Vec<_>>()
                .join("+");
            parts.push(format!("rmidx_{}", removed_column_indices));
        }

        if !self.skip_drop_bad && (self.drop_col_threshold - Self::DEFAULT_DROP_COL_THRESHOLD).abs() > f64::EPSILON {
            parts.push(format!("dropthr_{}", format_tag_float(self.drop_col_threshold)));
        }

        if (!self.skip_drop_bad || !self.skip_impute) && (self.value_tol - Self::DEFAULT_VALUE_TOL).abs() > f64::EPSILON
        {
            parts.push(format!("tol_{}", format_tag_float(self.value_tol)));
        }

        if !self.skip_drop_bad && self.drop_col_hour_threshold != Self::DEFAULT_DROP_COL_HOUR_THRESHOLD {
            parts.push(format!("runh_{}", self.drop_col_hour_threshold));
        }

        parts.join("__")
    }
}

fn make_tag_safe_token(raw: &str) -> String {
    raw.chars()
        .map(|ch| if ch.is_ascii_alphanumeric() { ch } else { '_' })
        .collect()
}

fn format_tag_float(value: f64) -> String {
    let mut formatted = format!("{value:.12}");
    while formatted.contains('.') && formatted.ends_with('0') {
        formatted.pop();
    }
    if formatted.ends_with('.') {
        formatted.pop();
    }
    formatted
}

pub struct LoadedBundle {
    pub known_meta: KnownMeta,
    pub source_tag: String,
    pub frame: NumericFrame,
    pub feature_names: Vec<String>,
    pub channels: usize,
}

// load stage fn
pub fn load(config: LoadConfig) -> Result<LoadedBundle, NumericFrameError> {
    // Stage 1: load CSV and drop the date column
    let mut frame = s02_01_load_csv(&config)?;

    // Stage 1a: remove specified columns by name or index
    if !config.remove_cols.is_empty() || !config.remove_col_idx.is_empty() {
        s02_03_remove_columns(&mut frame, &config)?;
    }

    let _feature_names = frame.column_names();

    // Python: print(f"   Features: {feature_names}")
    // {:?} is Rust's "debug print" — prints the Vec with brackets and commas.
    // Like Python's repr(): ['HUFL', 'HULL', 'MUFL', ...]
    // {} would be "display print" (like str()), but Vec doesn't implement Display.
    println!("   Features: {:?}", _feature_names);

    Ok(LoadedBundle {
        feature_names: _feature_names,
        channels: frame.width(),
        source_tag: config.tag(),
        known_meta: config.known_meta,
        frame,
    })
}

// known-data full-pipeline stage helper
pub fn load_ready_bundle(
    load_config: LoadConfig,
    noise_config: &NoiseConfig,
    partition_config: PartitionConfig,
    ready_config: ReadyConfig,
) -> Result<ReadyBundle, NumericFrameError> {
    let loaded_bundle = load(load_config.clone())?;
    let cleaned_bundle = clean(loaded_bundle, &load_config)?;
    let noisy_bundle = inject_noise(cleaned_bundle, noise_config)?;
    let partitioned_bundle = partition(noisy_bundle, partition_config)?;
    ready(partitioned_bundle, ready_config)
}

pub fn load_ready_bundle_train_std_noise(
    load_config: LoadConfig,
    sigma: f64,
    noise_seed: u64,
    partition_config: PartitionConfig,
    ready_config: ReadyConfig,
) -> Result<ReadyBundle, NumericFrameError> {
    let loaded_bundle = load(load_config.clone())?;
    let cleaned_bundle = clean(loaded_bundle, &load_config)?;
    let clean_like = inject_noise(
        cleaned_bundle.clone(),
        &NoiseConfig {
            sigma: 0.0,
            seed: 0,
            draw_ref: None,
            recipe: NoiseRecipe::Uniform,
            skip_noise: true,
        },
    )?;
    let clean_partitioned = partition(clean_like, partition_config.clone())?;
    let noisy_config = NoiseConfig {
        sigma,
        seed: noise_seed,
        draw_ref: None,
        recipe: NoiseRecipe::Scaled {
            channel_std: clean_partitioned.train_stats.std,
        },
        skip_noise: false,
    };
    let noisy_bundle = inject_noise(cleaned_bundle, &noisy_config)?;
    let noisy_partitioned = partition(noisy_bundle, partition_config)?;
    ready(noisy_partitioned, ready_config)
}

// load details
pub fn s02_01_load_csv(config: &LoadConfig) -> Result<NumericFrame, NumericFrameError> {
    println!("-> Loading known: {:?}", config.known_meta.dataset_id);

    let csv_path = &config.known_meta.path;
    let date_col = config.known_meta.date_column.as_deref();

    println!("   Reading CSV: {}", csv_path.display());

    let mut reader = csv::Reader::from_path(csv_path).map_err(|error| {
        NumericFrameError::new(format!("cannot open known-data CSV {}: {error}", csv_path.display()))
    })?;
    let headers = reader
        .headers()
        .map_err(|error| NumericFrameError::new(format!("cannot read CSV header: {error}")))?
        .clone();
    let selected: Vec<(usize, String)> = headers
        .iter()
        .enumerate()
        .filter(|(_, name)| date_col != Some(*name))
        .map(|(index, name)| (index, name.to_string()))
        .collect();
    if selected.is_empty() {
        return Err(NumericFrameError::new("known-data CSV has no numeric feature columns"));
    }
    let mut columns = vec![Vec::<Option<f64>>::new(); selected.len()];
    for record in reader.records() {
        let record = record.map_err(|error| NumericFrameError::new(format!("cannot read CSV row: {error}")))?;
        for (column, (source_index, name)) in columns.iter_mut().zip(&selected) {
            let raw = record.get(*source_index).unwrap_or_default().trim();
            let value = if raw.is_empty() {
                None
            } else {
                Some(raw.parse::<f64>().map_err(|error| {
                    NumericFrameError::new(format!(
                        "CSV column {name:?} contains non-numeric value {raw:?}: {error}"
                    ))
                })?)
            };
            column.push(value);
        }
    }
    let height = columns.first().map_or(0, Vec::len);
    let frame = NumericFrame::new(
        height,
        selected
            .into_iter()
            .zip(columns)
            .map(|((_, name), values)| NumericColumn::new(name, values))
            .collect(),
    )?;
    frame.drop_null_rows()
}

pub fn s02_03_remove_columns(frame: &mut NumericFrame, config: &LoadConfig) -> Result<(), NumericFrameError> {
    if !config.remove_cols.is_empty() || !config.remove_col_idx.is_empty() {
        // Snapshot all column names BEFORE any drops.
        // We need this because indices refer to the original layout.
        // After dropping column 3, what was column 5 is now column 4 — chaos.
        let all_names = frame.column_names();

        // Clone the index list because we need to sort it but config is borrowed (&).
        // Can't sort a borrowed slice — sorting mutates.
        let mut sorted_idx = config.remove_col_idx.clone();

        // Sort descending: [2, 7, 4] → [7, 4, 2].
        // Drop highest index first so lower indices stay valid.
        // |a, b| b.cmp(a) is "compare b to a" — reversed, so descending.
        // sort_unstable_by = sort without preserving order of equal elements (faster).
        sorted_idx.sort_unstable_by(|a, b| b.cmp(a));

        // Walk through indices highest to lowest
        for idx in sorted_idx {
            // Bounds check — skip if index is out of range
            if idx < all_names.len() {
                // all_names[idx] looks up the name we snapshotted.
                // &all_names[idx] borrows it as &str for drop_in_place.
                // ? = crash out if column somehow doesn't exist.
                // let _ = discard the returned Series (we don't need it).
                let _ = frame.drop_column(&all_names[idx])?;
            }
        }

        // Now drop by name — indices already handled, no shifting concern.
        // &config.remove_cols borrows the Vec, for col_name iterates &String.
        for col_name in &config.remove_cols {
            let _ = frame.drop_column(col_name)?;
        }
    }

    Ok(())
}

/// Tests for the load stage: config defaults, tag generation, and column removal.
///
/// Why these tests exist:
/// - Tag generation determines the artifact directory name and DB identity.
///   A tag that includes irrelevant knobs (e.g. drop settings when drop is
///   disabled) creates spurious hash differences between identical experiments.
/// - NumericColumn removal must apply index-based removal before name-based removal
///   to avoid index shifting bugs — removing column 2 by index then column
///   "HUFL" by name must not accidentally remove a different column.
/// - The full-pipeline integration test verifies that load → clean → noise →
///   partition → ready composes without panics on a real known dataset config.
#[cfg(test)]
mod tests {
    use super::*;
    use crate::stage_noise::NoiseConfig;
    use crate::stage_partition::PartitionConfig;
    use crate::stage_ready::ReadyConfig;
    use std::time::{SystemTime, UNIX_EPOCH};

    #[test]
    fn load_config_defaults() {
        let c = LoadConfig::new(KnownMeta::etth1());
        assert!(c.remove_cols.is_empty());
        assert!(c.remove_col_idx.is_empty());
        assert!(!c.skip_impute);
        assert!(!c.skip_drop_bad);
        assert!((c.drop_col_threshold - 0.2).abs() < 1e-10);
        assert!((c.value_tol - 1e-6).abs() < 1e-15);
        assert_eq!(c.drop_col_hour_threshold, 168);
    }

    #[test]
    fn load_config_cleaning_mode_tags_match_flag_combinations() {
        let base = LoadConfig::new(KnownMeta::etth1());
        assert_eq!(base.cleaning_mode_tag(), "drop_and_impute");
        assert_eq!(base.tag(), "etth1__drop_and_impute");

        let no_clean = LoadConfig {
            skip_drop_bad: true,
            skip_impute: true,
            ..LoadConfig::new(KnownMeta::etth1())
        };
        assert_eq!(no_clean.cleaning_mode_tag(), "no_clean");
        assert_eq!(no_clean.tag(), "etth1__no_clean");

        let drop_only = LoadConfig {
            skip_drop_bad: false,
            skip_impute: true,
            ..LoadConfig::new(KnownMeta::etth1())
        };
        assert_eq!(drop_only.cleaning_mode_tag(), "drop_only");

        let impute_only = LoadConfig {
            skip_drop_bad: true,
            skip_impute: false,
            ..LoadConfig::new(KnownMeta::etth1())
        };
        assert_eq!(impute_only.cleaning_mode_tag(), "impute_only");
    }

    #[test]
    fn load_config_tag_appends_only_non_default_data_affecting_knobs() {
        let config = LoadConfig {
            remove_cols: vec!["OT".to_string(), "HUFL".to_string()],
            remove_col_idx: vec![7, 2],
            drop_col_threshold: 0.35,
            value_tol: 1e-4,
            drop_col_hour_threshold: 24,
            ..LoadConfig::new(KnownMeta::etth1())
        };

        assert_eq!(
            config.tag(),
            "etth1__drop_and_impute__rmcols_HUFL+OT__rmidx_2+7__dropthr_0.35__tol_0.0001__runh_24"
        );
    }

    #[test]
    fn load_config_tag_skips_irrelevant_drop_knobs_when_drop_is_disabled() {
        let config = LoadConfig {
            skip_drop_bad: true,
            skip_impute: false,
            drop_col_threshold: 0.35,
            value_tol: 1e-4,
            drop_col_hour_threshold: 24,
            ..LoadConfig::new(KnownMeta::etth1())
        };

        assert_eq!(config.tag(), "etth1__impute_only__tol_0.0001");
    }

    #[test]
    fn remove_columns_by_index_then_name() {
        let mut frame = NumericFrame::new(
            2,
            vec![
                NumericColumn::new("a".into(), &[1.0f64, 2.0]),
                NumericColumn::new("b".into(), &[3.0f64, 4.0]),
                NumericColumn::new("c".into(), &[5.0f64, 6.0]),
                NumericColumn::new("d".into(), &[7.0f64, 8.0]),
            ],
        )
        .unwrap();

        let config = LoadConfig {
            remove_cols: vec!["a".to_string()],
            remove_col_idx: vec![2], // remove original "c" first
            ..LoadConfig::new(KnownMeta::etth1())
        };

        s02_03_remove_columns(&mut frame, &config).unwrap();
        let names = frame.column_names();
        assert_eq!(names, vec!["b", "d"]);
    }

    #[test]
    fn remove_columns_ignores_out_of_range_indices() {
        let mut frame = NumericFrame::new(
            2,
            vec![
                NumericColumn::new("a".into(), &[1.0f64, 2.0]),
                NumericColumn::new("b".into(), &[3.0f64, 4.0]),
            ],
        )
        .unwrap();

        let config = LoadConfig {
            remove_cols: vec![],
            remove_col_idx: vec![999],
            ..LoadConfig::new(KnownMeta::etth1())
        };

        s02_03_remove_columns(&mut frame, &config).unwrap();
        assert_eq!(frame.column_names(), vec!["a", "b"]);
    }

    #[test]
    fn load_ready_bundle_runs_full_known_pipeline() {
        let unique_suffix = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos();
        let csv_path = std::env::temp_dir().join(format!(
            "serie_rs_stage_load_ready_bundle_{}_{}.csv",
            std::process::id(),
            unique_suffix
        ));

        let mut csv_text = String::from("date,HUFL,HULL,OT\n");
        for i in 0..50 {
            csv_text.push_str(&format!(
                "2020-01-01 00:{:02}:00,{:.3},{:.3},{:.3}\n",
                i,
                1.0 + i as f64 * 0.1,
                2.0 + i as f64 * 0.2,
                3.0 + i as f64 * 0.3,
            ));
        }
        std::fs::write(&csv_path, csv_text).unwrap();

        let mut known_meta = KnownMeta::etth1();
        known_meta.path = csv_path.clone();

        let mut load_config = LoadConfig::new(known_meta);
        load_config.skip_drop_bad = true;
        load_config.skip_impute = true;
        let expected_tag = load_config.tag();

        let noise_config = NoiseConfig {
            skip_noise: true,
            ..NoiseConfig::new(0.0, 11)
        };

        let ready_bundle = load_ready_bundle(
            load_config,
            &noise_config,
            PartitionConfig::p70_10_20(),
            ReadyConfig::new(2, 1),
        )
        .unwrap();

        assert_eq!(ready_bundle.channels, 3);
        assert!(ready_bundle.train.n_windows > 0);
        assert!(ready_bundle.val.n_windows > 0);
        assert!(ready_bundle.test.n_windows > 0);
        assert_eq!(ready_bundle.source_descriptor.tag(), expected_tag);

        std::fs::remove_file(csv_path).unwrap();
    }
}
