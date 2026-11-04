use crate::draw_protocol::DrawRef;
use crate::numeric_frame::{NumericColumn, NumericFrame, NumericFrameError};
use crate::stage_clean::{CleanedBundle, KnownMeta};
use crate::stage_synthetic::{SyntheticBundle, TrajectorySpec};

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// bundle transitions (stage fn change)
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// inject_noise(bundle, noise_config): CleanedBundle | SyntheticBundle -> NoisyBundle
// NoiseProfile/NoiseRecipe methods  : internal helpers (no bundle boundary change)

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// noise
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// noise structs

#[derive(Debug, Clone, PartialEq)]
pub struct NoiseConfig {
    pub sigma: f64,
    pub seed: u64,
    pub draw_ref: Option<DrawRef>,
    pub recipe: NoiseRecipe,
    pub skip_noise: bool, // true = pass through clean, sigma=0 baseline
}

impl NoiseConfig {
    pub fn new(sigma: f64, seed: u64) -> Self {
        Self {
            sigma,
            seed,
            draw_ref: None,
            recipe: NoiseRecipe::Uniform,
            skip_noise: false,
        }
    }

    pub fn from_noise_draw(sigma: f64, draw_ref: DrawRef) -> Self {
        Self {
            sigma,
            seed: draw_ref.draw_seed,
            draw_ref: Some(draw_ref),
            recipe: NoiseRecipe::Uniform,
            skip_noise: false,
        }
    }

    /// Protocol B: clean train/val, noisy test.
    pub fn clean_train(sigma: f64, seed: u64) -> Self {
        Self {
            sigma,
            seed,
            draw_ref: None,
            recipe: NoiseRecipe::PerPhase {
                train_frac: 0.7,
                val_frac: 0.2,
                test_frac: 0.1,
                w_train: 0.0,
                w_val: 0.0,
                w_test: 1.0,
            },
            skip_noise: false,
        }
    }
}

pub enum NoiseStageInputBundle {
    KnownClean(CleanedBundle),
    Synthetic(SyntheticBundle),
}

impl From<CleanedBundle> for NoiseStageInputBundle {
    fn from(bundle: CleanedBundle) -> Self {
        Self::KnownClean(bundle)
    }
}

impl From<SyntheticBundle> for NoiseStageInputBundle {
    fn from(bundle: SyntheticBundle) -> Self {
        Self::Synthetic(bundle)
    }
}

#[derive(Debug, Clone)]
pub enum SourceDescriptor {
    Known { known_meta: KnownMeta, source_tag: String },
    Synthetic(TrajectorySpec),
}

impl SourceDescriptor {
    pub fn tag(&self) -> String {
        match self {
            Self::Known { source_tag, .. } => source_tag.clone(),
            Self::Synthetic(spec) => spec.tag(),
        }
    }
}

pub struct NoisyBundle {
    pub source_descriptor: SourceDescriptor,
    pub frame: NumericFrame,
    pub feature_names: Vec<String>,
    pub channels: usize,
    // These fields record what was done — the data's identity
    pub sigma: f64,
    pub noise_seed: u64,
    pub noise_draw_ref: Option<DrawRef>,
    pub recipe: NoiseRecipe,
}

// noise stage fn
pub fn inject_noise(
    bundle: impl Into<NoiseStageInputBundle>,
    noise_config: &NoiseConfig,
) -> Result<NoisyBundle, NumericFrameError> {
    let bundle = bundle.into();
    let (source_descriptor, mut frame) = match bundle {
        NoiseStageInputBundle::KnownClean(bundle) => (
            SourceDescriptor::Known {
                known_meta: bundle.known_meta,
                source_tag: bundle.source_tag,
            },
            bundle.frame,
        ),
        NoiseStageInputBundle::Synthetic(bundle) => (SourceDescriptor::Synthetic(bundle.config), bundle.frame),
    };

    if !noise_config.skip_noise && noise_config.sigma > 0.0 {
        // Generate noise matrix: same shape as frame, N(0, sigma)
        let n_rows = frame.height();
        let columns = frame.column_names();
        let dim = columns.len();

        // Materialize the noise profile: [T, D] weight matrix
        let profile = NoiseProfile::from_recipe(&noise_config.recipe, n_rows, dim);

        // One RNG, one seed, reproducible
        use rand::SeedableRng;
        use rand_distr::{Distribution, StandardNormal};
        use rand_pcg::Pcg64;

        let mut rng = Pcg64::seed_from_u64(noise_config.seed);

        // realized = clean + (sigma * sigma_map[t,d]) * unit_noise[t,d]
        let noisy_cols: Vec<NumericColumn> = columns
            .iter()
            .enumerate()
            .map(|(d, c)| {
                let series = frame.column(c).expect("column must exist");
                let f64_ca = series.numeric().expect("column must be f64");

                let noisy: Vec<Option<f64>> = f64_ca
                    .into_iter()
                    .enumerate()
                    .map(|(t, val)| {
                        val.map(|v| {
                            let unit: f64 = StandardNormal.sample(&mut rng);
                            v + noise_config.sigma * profile.read_weight(t, d) * unit
                        })
                    })
                    .collect();

                NumericColumn::new(c.clone(), noisy.as_slice())
            })
            .collect();

        frame = NumericFrame::new(n_rows, noisy_cols)?;
    }

    Ok(NoisyBundle {
        feature_names: frame.column_names(),
        channels: frame.width(),
        source_descriptor,
        frame,
        sigma: noise_config.sigma,
        noise_seed: noise_config.seed,
        noise_draw_ref: noise_config.draw_ref,
        recipe: noise_config.recipe.clone(),
    })
}

#[derive(Debug, Clone, PartialEq)]
pub enum NoiseRecipe {
    /// All splits at titration sigma. sigma_map = all ones.
    Uniform,
    /// Scale noise to channel stds. sigma=1 → 1 std-dev per channel.
    Scaled { channel_std: Vec<f64> },
    /// Three levels aligned to partition boundaries.
    PerPhase {
        train_frac: f64,
        val_frac: f64,
        test_frac: f64,
        w_train: f64,
        w_val: f64,
        w_test: f64,
    },
}

impl NoiseRecipe {
    pub fn tag(&self) -> &'static str {
        match self {
            Self::Uniform => "uniform",
            Self::Scaled { .. } => "scaled",
            Self::PerPhase { .. } => "perphase",
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct NoiseProfile {
    pub sigma_map: Vec<f64>, // [T * D], row-major
    pub rows: usize,         // T — number of timesteps
    pub dim: usize,          // D — number of channels
}

impl NoisyBundle {
    pub fn tag(&self) -> String {
        let mut tag = self.source_descriptor.tag();
        if self.sigma > 0.0 {
            tag.push_str(&format!(
                "__noise_{}__{}__seed_{}",
                format_tag_float(self.sigma),
                self.recipe.tag(),
                self.noise_seed
            ));
        }
        tag
    }
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

impl NoiseProfile {
    // ── The one entry point from config ──

    /// Build from a recipe + known dimensions.
    /// This replaces both NoiseProfileSpec and its resolve()/materialize().
    pub fn from_recipe(recipe: &NoiseRecipe, rows: usize, dim: usize) -> Self {
        match recipe {
            NoiseRecipe::Uniform => Self::uniform(rows, dim),
            NoiseRecipe::Scaled { channel_std } => {
                assert_eq!(
                    channel_std.len(),
                    dim,
                    "channel_std length {} != dim {}",
                    channel_std.len(),
                    dim
                );
                Self::per_channel(channel_std, rows)
            }
            NoiseRecipe::PerPhase {
                train_frac,
                val_frac,
                w_train,
                w_val,
                w_test,
                ..
            } => Self::per_phase(rows, dim, *train_frac, *val_frac, *w_train, *w_val, *w_test),
        }
    }

    // ── Direct constructors (still available for tests, one-offs) ──

    /// All-ones: sigma values ARE the effective sigmas.
    /// This is the default — recovers standard additive noise.
    /// Python: NoiseProfile.uniform(T, D)
    pub fn uniform(rows: usize, dim: usize) -> Self {
        Self {
            sigma_map: vec![1.0; rows * dim],
            rows,
            dim,
        }
    }

    /// Different relative intensity per channel, constant in time.
    /// weights: [D] — e.g. [1.0, 0.5, 2.0] means ch0 full, ch1 half, ch2 double.
    ///
    /// If weights = [1.0, 0.5, 2.0] and rows = 3, sigma_map becomes:
    /// [1.0, 0.5, 2.0,   1.0, 0.5, 2.0,   1.0, 0.5, 2.0]
    ///  ─── row 0 ────   ─── row 1 ────   ─── row 2 ────
    ///
    /// Same pattern every row — noise shape is constant in time,
    /// only varies across channels.
    pub fn per_channel(weights: &[f64], rows: usize) -> Self {
        // &[f64] is a borrowed slice — like a numpy view into a list of floats.
        // We don't own it, just read from it.
        let dim = weights.len();

        // (0..rows) — a range iterator: 0, 1, 2, ..., rows-1
        //             Python: range(rows)
        //
        // .flat_map(|_| ...) — for each row, produce D(dim) values, flatten into one stream.
        //   |_| means "I don't care about the row index" (underscore = ignore).
        //   Python equivalent: [w for _ in range(rows) for w in weights]
        //
        // weights.iter() — iterate over &f64 references (borrowed, no copy)
        // .copied()      — copy the f64 value out of the reference (&f64 → f64)
        //                   f64 is 8 bytes, Copy is trivial.
        //                   Without .copied(), you'd collect Vec<&f64> instead of Vec<f64>.
        //
        // .collect()      — drain the iterator into a Vec<f64>.
        //                   Rust knows the target type from the `let sigma_map: Vec<f64>` annotation.
        let sigma_map: Vec<f64> = (0..rows).flat_map(|_| weights.iter().copied()).collect();

        Self { sigma_map, rows, dim }
    }

    /// Scale noise proportional to each channel's empirical std.
    /// sigma=1.0 then means "1 std-dev of noise per channel".
    /// Python: NoiseProfile.from_channel_std(channel_std, T)
    pub fn from_channel_std(channel_std: &[f64], rows: usize) -> Self {
        Self::per_channel(channel_std, rows)
    }

    /// Three constant levels aligned to partition boundaries.
    /// e.g. (0, 0, 1) = clean train, clean val, noisy test.
    /// Python: NoiseProfile.per_phase(T, D, partition_spec, w_train, w_val, w_test)
    pub fn per_phase(
        rows: usize,
        dim: usize,
        train_frac: f64,
        val_frac: f64,
        w_train: f64,
        w_val: f64,
        w_test: f64,
    ) -> Self {
        let t1 = (rows as f64 * train_frac) as usize;
        let v1 = (rows as f64 * (train_frac + val_frac)) as usize;
        let mut sigma_map = vec![0.0; rows * dim];
        for t in 0..rows {
            let weight = if t < t1 {
                w_train
            } else if t < v1 {
                w_val
            } else {
                w_test
            };
            for d in 0..dim {
                sigma_map[t * dim + d] = weight;
            }
        }
        Self { sigma_map, rows, dim }
    }

    /// Fully custom — you built the [T, D] map yourself.
    /// Python: NoiseProfile.from_map(sigma_map)
    pub fn from_map(sigma_map: Vec<f64>, rows: usize, dim: usize) -> Self {
        assert_eq!(
            sigma_map.len(),
            rows * dim,
            "sigma_map length {} != rows={} * dim={}",
            sigma_map.len(),
            rows,
            dim,
        );
        Self { sigma_map, rows, dim }
    }

    /// Examine the inside
    /// Read the weight at (t, d).
    pub fn read_weight(&self, t: usize, d: usize) -> f64 {
        self.sigma_map[self.idx(t, d)]
    }
    // ── Index helper ──
    // sigma_map[(t, d)] = sigma_map[t * dim + d]
    fn idx(&self, t: usize, d: usize) -> usize {
        t * self.dim + d
    }
}

/// Tests for the noise injection stage: profiles, recipes, CRN, and provenance.
///
/// Why these tests exist:
/// - Common Random Numbers (CRN): same seed + different sigma must produce noise
///   draws that differ only in amplitude, not in the draw sequence itself. This
///   is the foundation of noise titration — without CRN, comparing sigma=0.1 vs
///   sigma=1.0 is confounded by different noise realizations.
/// - Sigma-zero identity: injecting noise at sigma=0 must return bit-identical
///   data. If it doesn't, the "clean" baseline in a titration sweep is not
///   actually clean.
/// - Seed determinism: same seed → identical noise. Without this, experiments
///   are not reproducible.
/// - Provenance: the noise draw reference (seed_base, draw_ordinal, draw_seed)
///   must survive injection so that downstream stages can audit which draw
///   produced which result.
/// - Tag minimality: sigma=0 should produce a minimal tag (no noise suffix),
///   while sigma>0 should include the noise identity in the tag.
#[cfg(test)]
mod tests {
    use super::*;
    use crate::stage_synthetic::{ODESystem, TrajectorySpec};

    fn make_tiny_frame() -> NumericFrame {
        let n = 100;
        let hufl: Vec<f64> = (0..n).map(|i| 5.0 + 3.0 * (i as f64 * 0.1).sin()).collect();
        let hull: Vec<f64> = (0..n).map(|i| 8.0 + 2.0 * (i as f64 * 0.05).cos()).collect();
        let mufl: Vec<f64> = (0..n).map(|i| 12.0 + 4.0 * (i as f64 * 0.08).sin()).collect();
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

    fn bundle_from(frame: NumericFrame) -> CleanedBundle {
        let feature_names = frame.column_names();
        let channels = frame.width();
        CleanedBundle {
            known_meta: KnownMeta::etth1(),
            source_tag: "etth1__drop_and_impute".to_string(),
            frame,
            feature_names,
            channels,
        }
    }

    fn synthetic_bundle_from(frame: NumericFrame) -> SyntheticBundle {
        let feature_names = frame.column_names();
        let channels = frame.width();
        SyntheticBundle {
            config: TrajectorySpec::lorenz(),
            frame,
            feature_names,
            channels,
        }
    }

    #[test]
    fn profile_uniform_all_ones() {
        let p = NoiseProfile::uniform(10, 3);
        assert!(p.sigma_map.iter().all(|&v| (v - 1.0).abs() < 1e-15));
    }

    #[test]
    fn profile_per_channel_repeats() {
        let p = NoiseProfile::per_channel(&[1.0, 0.5, 2.0], 4);
        for t in 0..4 {
            assert!((p.read_weight(t, 0) - 1.0).abs() < 1e-15);
            assert!((p.read_weight(t, 1) - 0.5).abs() < 1e-15);
            assert!((p.read_weight(t, 2) - 2.0).abs() < 1e-15);
        }
    }

    #[test]
    fn profile_per_phase_three_levels() {
        let p = NoiseProfile::per_phase(100, 2, 0.7, 0.1, 0.0, 0.5, 1.0);
        assert!((p.read_weight(0, 0) - 0.0).abs() < 1e-15);
        assert!((p.read_weight(70, 0) - 0.5).abs() < 1e-15);
        assert!((p.read_weight(80, 0) - 1.0).abs() < 1e-15);
    }

    #[test]
    #[should_panic(expected = "sigma_map length")]
    fn profile_from_map_rejects_wrong_size() {
        let _ = NoiseProfile::from_map(vec![1.0; 10], 3, 4);
    }

    #[test]
    fn inject_noise_skip_passthrough() {
        let input = make_tiny_frame();
        let cfg = NoiseConfig {
            skip_noise: true,
            ..NoiseConfig::new(0.1, 42)
        };

        let out = inject_noise(bundle_from(input.clone()), &cfg).unwrap();
        assert_eq!(out.frame.shape(), input.shape());
        let in_vals: Vec<f64> = input
            .column("HUFL")
            .unwrap()
            .numeric()
            .unwrap()
            .into_iter()
            .filter_map(|v| v)
            .collect();
        let out_vals: Vec<f64> = out
            .frame
            .column("HUFL")
            .unwrap()
            .numeric()
            .unwrap()
            .into_iter()
            .filter_map(|v| v)
            .collect();
        assert_eq!(in_vals, out_vals);
    }

    #[test]
    fn inject_noise_accepts_synthetic_bundle() {
        let input = make_tiny_frame();
        let cfg = NoiseConfig {
            skip_noise: true,
            ..NoiseConfig::new(0.1, 42)
        };

        let out = inject_noise(synthetic_bundle_from(input.clone()), &cfg).unwrap();
        assert_eq!(out.frame.shape(), input.shape());
        match out.source_descriptor {
            SourceDescriptor::Synthetic(spec) => assert_eq!(spec.system, ODESystem::Lorenz),
            SourceDescriptor::Known { .. } => panic!("expected synthetic source descriptor"),
        }
    }

    #[test]
    fn noisy_bundle_tag_is_minimal_when_sigma_zero_and_specific_when_noisy() {
        let clean_like = inject_noise(bundle_from(make_tiny_frame()), &NoiseConfig::new(0.0, 0)).unwrap();
        assert_eq!(clean_like.tag(), "etth1__drop_and_impute");

        let noisy = inject_noise(bundle_from(make_tiny_frame()), &NoiseConfig::new(0.1, 42)).unwrap();
        assert_eq!(noisy.tag(), "etth1__drop_and_impute__noise_0.1__uniform__seed_42");
    }

    #[test]
    fn noise_config_can_carry_spawned_draw_provenance() {
        let draw_ref = DrawRef::new(1000, 1);
        let cfg = NoiseConfig::from_noise_draw(0.1, draw_ref);
        let out = inject_noise(bundle_from(make_tiny_frame()), &cfg).unwrap();
        assert_eq!(out.noise_seed, draw_ref.draw_seed);
        assert_eq!(out.noise_draw_ref, Some(draw_ref));
    }

    #[test]
    fn inject_noise_sigma_one_changes_many_values() {
        let input = make_tiny_frame();
        let original: Vec<f64> = input
            .column("HUFL")
            .unwrap()
            .numeric()
            .unwrap()
            .into_iter()
            .filter_map(|v| v)
            .collect();

        let out = inject_noise(bundle_from(input), &NoiseConfig::new(1.0, 42)).unwrap();
        let noisy: Vec<f64> = out
            .frame
            .column("HUFL")
            .unwrap()
            .numeric()
            .unwrap()
            .into_iter()
            .filter_map(|v| v)
            .collect();

        let diff_count = original
            .iter()
            .zip(noisy.iter())
            .filter(|(a, b)| (*a - *b).abs() > 1e-10)
            .count();
        assert!(diff_count > 50);
    }

    #[test]
    fn inject_noise_same_seed_is_reproducible() {
        let cfg = NoiseConfig::new(0.5, 12345);
        let a = inject_noise(bundle_from(make_tiny_frame()), &cfg).unwrap();
        let b = inject_noise(bundle_from(make_tiny_frame()), &cfg).unwrap();

        let va: Vec<f64> = a
            .frame
            .column("OT")
            .unwrap()
            .numeric()
            .unwrap()
            .into_iter()
            .filter_map(|v| v)
            .collect();
        let vb: Vec<f64> = b
            .frame
            .column("OT")
            .unwrap()
            .numeric()
            .unwrap()
            .into_iter()
            .filter_map(|v| v)
            .collect();

        for (x, y) in va.iter().zip(vb.iter()) {
            assert!((x - y).abs() < 1e-15);
        }
    }

    #[test]
    fn inject_noise_sigma_zero_is_noop() {
        let input = make_tiny_frame();
        let original: Vec<f64> = input
            .column("OT")
            .unwrap()
            .numeric()
            .unwrap()
            .into_iter()
            .filter_map(|v| v)
            .collect();

        let out = inject_noise(bundle_from(input), &NoiseConfig::new(0.0, 0)).unwrap();
        let after: Vec<f64> = out
            .frame
            .column("OT")
            .unwrap()
            .numeric()
            .unwrap()
            .into_iter()
            .filter_map(|v| v)
            .collect();

        assert_eq!(original, after);
    }

    // ── Gap tests (mirroring Python test_contracts.py) ────────────────

    #[test]
    fn inject_noise_different_seed_produces_different_values() {
        let a = inject_noise(bundle_from(make_tiny_frame()), &NoiseConfig::new(0.5, 100)).unwrap();
        let b = inject_noise(bundle_from(make_tiny_frame()), &NoiseConfig::new(0.5, 200)).unwrap();

        let va: Vec<f64> = a
            .frame
            .column("HUFL")
            .unwrap()
            .numeric()
            .unwrap()
            .into_iter()
            .filter_map(|v| v)
            .collect();
        let vb: Vec<f64> = b
            .frame
            .column("HUFL")
            .unwrap()
            .numeric()
            .unwrap()
            .into_iter()
            .filter_map(|v| v)
            .collect();

        let diff_count = va
            .iter()
            .zip(vb.iter())
            .filter(|(a, b)| (*a - *b).abs() > 1e-10)
            .count();
        assert!(
            diff_count > 50,
            "different seeds should produce different noise, got {diff_count} diffs"
        );
    }

    #[test]
    fn common_random_numbers_two_sigmas_differ_only_in_amplitude() {
        // This is the core CRN contract: same seed, different sigma →
        // noise draws are identical, only amplitude differs.
        let seed = 42u64;
        let input = make_tiny_frame();

        let clean_vals: Vec<f64> = input
            .column("HUFL")
            .unwrap()
            .numeric()
            .unwrap()
            .into_iter()
            .filter_map(|v| v)
            .collect();

        let lo = inject_noise(bundle_from(input.clone()), &NoiseConfig::new(0.25, seed)).unwrap();
        let hi = inject_noise(bundle_from(input), &NoiseConfig::new(1.0, seed)).unwrap();

        let lo_vals: Vec<f64> = lo
            .frame
            .column("HUFL")
            .unwrap()
            .numeric()
            .unwrap()
            .into_iter()
            .filter_map(|v| v)
            .collect();
        let hi_vals: Vec<f64> = hi
            .frame
            .column("HUFL")
            .unwrap()
            .numeric()
            .unwrap()
            .into_iter()
            .filter_map(|v| v)
            .collect();

        // Recover the noise draw: (noisy - clean) / sigma
        // If CRN holds, recovered draws should be identical for both sigmas
        let lo_draw: Vec<f64> = lo_vals
            .iter()
            .zip(clean_vals.iter())
            .map(|(n, c)| (n - c) / 0.25)
            .collect();
        let hi_draw: Vec<f64> = hi_vals
            .iter()
            .zip(clean_vals.iter())
            .map(|(n, c)| (n - c) / 1.0)
            .collect();

        for (i, (a, b)) in lo_draw.iter().zip(hi_draw.iter()).enumerate() {
            assert!(
                (a - b).abs() < 1e-10,
                "CRN violation at index {i}: lo_draw={a}, hi_draw={b}"
            );
        }
    }

    #[test]
    fn noise_draw_ref_provenance_survives_injection() {
        let draw = DrawRef::new(5000, 3);
        let cfg = NoiseConfig::from_noise_draw(0.1, draw);
        let out = inject_noise(bundle_from(make_tiny_frame()), &cfg).unwrap();
        assert_eq!(out.noise_draw_ref, Some(draw));
        assert_eq!(out.noise_seed, draw.draw_seed);
    }

    #[test]
    fn synthetic_source_tag_survives_noise_injection() {
        let input = make_tiny_frame();
        let cfg = NoiseConfig::new(0.5, 42);
        let out = inject_noise(synthetic_bundle_from(input), &cfg).unwrap();
        match &out.source_descriptor {
            SourceDescriptor::Synthetic(spec) => {
                assert_eq!(spec.system, ODESystem::Lorenz);
            }
            _ => panic!("synthetic source descriptor should survive noise injection"),
        }
    }
}
