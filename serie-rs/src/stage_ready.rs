use crate::draw_protocol::DrawRef;
#[cfg(test)]
use crate::numeric_frame::NumericColumn;
use crate::numeric_frame::{NumericFrame, NumericFrameError};
use crate::stage_noise::{NoiseRecipe, SourceDescriptor};
use crate::stage_partition::{NormMethod, PartitionedBundle};

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// bundle transitions (stage fn change)
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// ready(bundle, config): PartitionedBundle -> ReadyBundle
// helpers (row-major, normalize, window, bootstrap, reshape): internal details

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// ready
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Substage order inside ready():
// s06_01 frame_to_row_major_f32
// s06_02 normalize_in_place
// s06_03 split row ranges (train/val/test)
// s06_04 sliding_window per split
// s06_05 generate_bootstrap_trials (optional, train only)
// s06_06 reshape_channel_independent (optional)

pub fn sliding_window(
    data: &[f32],    // [N * D] row-major
    n_rows: usize,   // N
    dim: usize,      // D
    seq_len: usize,  // S
    pred_len: usize, // H
) -> WindowedSplit {
    assert!(dim > 0, "dim must be > 0; got {}", dim);
    assert!(seq_len > 0, "seq_len must be > 0; got {}", seq_len);
    assert!(pred_len > 0, "pred_len must be > 0; got {}", pred_len);
    assert_eq!(
        data.len(),
        n_rows * dim,
        "data.len() mismatch: got {}, expected {}",
        data.len(),
        n_rows * dim
    );
    assert!(
        n_rows >= seq_len + pred_len,
        "Not enough rows: N={}, need seq+pred={}",
        n_rows,
        seq_len + pred_len
    );

    let total_len = seq_len + pred_len;
    let n_windows = n_rows - total_len + 1;

    // Pre-allocate exact capacity — no reallocs during the loop
    let mut x = Vec::with_capacity(n_windows * seq_len * dim);
    let mut y = Vec::with_capacity(n_windows * pred_len * dim);

    for w in 0..n_windows {
        let x_start = w * dim;
        let x_end = (w + seq_len) * dim;
        x.extend_from_slice(&data[x_start..x_end]);
        // &data[x_start..x_end] — borrow a slice of data from index x_start to x_end (exclusive)
        //                          no copy, just a pointer + length, like numpy data[start:end]

        // x.extend_from_slice(&slice) — append those bytes onto the end of x
        //                                it's a memcpy into x's pre-allocated space
        //                                x grows by (x_end - x_start) elements

        let y_start = (w + seq_len) * dim;
        let y_end = (w + total_len) * dim;
        y.extend_from_slice(&data[y_start..y_end]);
    }
    WindowedSplit { x, y, n_windows }
}

use rayon::prelude::*;

pub fn sliding_window_parallel(
    data: &[f32],
    n_rows: usize,
    dim: usize,
    seq_len: usize,
    pred_len: usize,
) -> WindowedSplit {
    assert!(dim > 0, "dim must be > 0; got {}", dim);
    assert!(seq_len > 0, "seq_len must be > 0; got {}", seq_len);
    assert!(pred_len > 0, "pred_len must be > 0; got {}", pred_len);
    assert_eq!(
        data.len(),
        n_rows * dim,
        "data.len() mismatch: got {}, expected {}",
        data.len(),
        n_rows * dim
    );
    assert!(
        n_rows >= seq_len + pred_len,
        "Not enough rows: N={}, need seq+pred={}",
        n_rows,
        seq_len + pred_len
    );
    let total_len = seq_len + pred_len;
    let n_windows = n_rows - total_len + 1;

    // Pre-allocate, write in parallel — each window writes to its own slice
    let mut x = vec![0.0f32; n_windows * seq_len * dim];
    let mut y = vec![0.0f32; n_windows * pred_len * dim];

    // x and y are pre-allocated: vec![0.0f32; n_windows * seq_len * dim] etc.
    // We now fill them in parallel — each window is independent, no overlap.

    // Split x into n_windows slices, each seq_len * dim long.
    // Each slice is one window's worth of x data.
    // par_ means rayon hands each slice to a different CPU core.
    x.par_chunks_mut(seq_len * dim)
        // Same for y — split into pred_len * dim chunks.
        // zip pairs them: (x_chunk_0, y_chunk_0), (x_chunk_1, y_chunk_1), ...
        .zip(y.par_chunks_mut(pred_len * dim))
        // Add window index: (0, (x_chunk, y_chunk)), (1, ...), ...
        .enumerate()
        // w = window index
        // x_chunk = &mut [f32] of length seq_len * dim — this window's x destination
        // y_chunk = &mut [f32] of length pred_len * dim — this window's y destination
        .for_each(|(w, (x_chunk, y_chunk))| {
            // Where to read from in data:
            // x reads from row w
            let x_start = w * dim;
            // y reads from row w + seq_len
            let y_start = (w + seq_len) * dim;

            // copy_from_slice: same as extend_from_slice but into a pre-existing slice.
            // Destination already allocated (vec![0.0; ...]), we just overwrite the zeros.
            // Each core writes to its own chunk — no locks, no overlap, no coordination.
            x_chunk.copy_from_slice(&data[x_start..x_start + seq_len * dim]);
            y_chunk.copy_from_slice(&data[y_start..y_start + pred_len * dim]);
        });

    WindowedSplit { x, y, n_windows }
}

// ready structs
/// Config for the ready stage: normalize, split, window.
#[derive(Debug, Clone, PartialEq)]
pub struct ReadyConfig {
    pub seq_len: usize,
    pub pred_len: usize,
    pub norm_method: NormMethod,
    pub eps: f64,
    pub batch_size: usize,
    pub num_workers: usize,
    pub channel_independent: bool,
    pub bootstrap_block_size: usize,
    pub bootstrap_n_trials: usize,
    pub bootstrap_seed: u64,
}

impl ReadyConfig {
    pub fn new(seq_len: usize, pred_len: usize) -> Self {
        Self {
            seq_len,
            pred_len,
            norm_method: NormMethod::None,
            eps: 1e-8,
            batch_size: 95,
            num_workers: 0,
            channel_independent: false,
            bootstrap_block_size: 2 * (seq_len + pred_len), // 1344 for seq=pred=336
            bootstrap_n_trials: 0,                          // 0 = disabled
            bootstrap_seed: 42,
        }
    }
}

/// Everything needed to create DataLoaders on the Python side.
#[derive(Debug, Clone)]
pub struct ReadyBundle {
    pub train: WindowedSplit,
    pub val: WindowedSplit,
    pub test: WindowedSplit,
    pub source_descriptor: SourceDescriptor,
    pub feature_names: Vec<String>,
    pub channels: usize,
    pub seq_len: usize,
    pub pred_len: usize,
    pub center: Vec<f64>, // [D] for denormalization later
    pub scale: Vec<f64>,  // [D] for denormalization later
    pub norm_method: NormMethod,
    // carried forward
    pub sigma: f64,
    pub noise_seed: u64,
    pub noise_draw_ref: Option<DrawRef>,
    pub noise_recipe: NoiseRecipe,
    pub channel_independent: bool,
    // ENBPI/bootstrap indices on TRAIN windows before CI reshape
    pub bootstrap_trials: Vec<BootstrapTrial>,
}

impl ReadyBundle {
    pub fn tag(&self) -> String {
        let mut tag = self.source_descriptor.tag();
        if self.sigma > 0.0 {
            tag.push_str(&format!(
                "__noise_{}__{}__seed_{}",
                format_tag_float(self.sigma),
                self.noise_recipe.tag(),
                self.noise_seed
            ));
        }
        if self.channel_independent {
            tag.push_str("__ci");
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

/// Normalize [N, D] in place: x[t,d] = (x[t,d] - center[d]) / max(scale[d], eps)
fn normalize_in_place(data: &mut [f32], n_rows: usize, dim: usize, center: &[f64], scale: &[f64], eps: f64) {
    for t in 0..n_rows {
        for d in 0..dim {
            let idx = t * dim + d;
            let c = center[d] as f32;
            let s = scale[d].max(eps) as f32;
            data[idx] = (data[idx] - c) / s;
        }
    }
}

/// NumericFrame → contiguous row-major [N, D] f32 buffer.
/// The one unavoidable copy: column-major values → row-major flat buffer.
fn frame_to_row_major_f32(frame: &NumericFrame) -> Result<(Vec<f32>, usize, usize), NumericFrameError> {
    let n_rows = frame.height();
    let dim = frame.width();
    let mut data = vec![0.0f32; n_rows * dim];

    for (d, name) in frame.column_names().iter().enumerate() {
        let series = frame.column(name.as_str())?;
        let f64_ca = series.numeric()?;
        for t in 0..n_rows {
            data[t * dim + d] = f64_ca.get(t).unwrap_or(0.0) as f32;
        }
    }

    Ok((data, n_rows, dim))
}

/// Reshape [B, S, D] → [B*D, S] for channel-independent mode.
/// Each window's [S, D] matrix gets transposed to [D, S],
/// then all D slices are stacked along axis 0.
///
/// Before: window w, timestep s, feature d → data[(w * S + s) * D + d]
/// After:  window w * D + d, timestep s    → out[((w * D + d) * S + s)]
///
/// Python equivalent:
///   einops.rearrange(x, "b s d -> (b d) s 1")
pub fn reshape_channel_independent(
    data: &[f32],     // [B * S * D] row-major
    n_windows: usize, // B
    win_len: usize,   // S (seq_len or pred_len)
    dim: usize,       // D
) -> Vec<f32> {
    // Output: [B*D, S] = [B*D * S] elements
    let mut out = vec![0.0f32; n_windows * dim * win_len];

    for w in 0..n_windows {
        for d in 0..dim {
            for s in 0..win_len {
                // Read from [w, s, d]
                let src = (w * win_len + s) * dim + d;
                // Write to [w*D + d, s]
                let dst = (w * dim + d) * win_len + s;
                out[dst] = data[src];
            }
        }
    }

    out
}

fn reshape_windowed_split_channel_independent(
    split: WindowedSplit,
    seq_len: usize,
    pred_len: usize,
    dim: usize,
) -> WindowedSplit {
    WindowedSplit {
        x: reshape_channel_independent(&split.x, split.n_windows, seq_len, dim),
        y: reshape_channel_independent(&split.y, split.n_windows, pred_len, dim),
        n_windows: split.n_windows * dim,
    }
}

/// One split's windowed data.
#[derive(Debug, Clone)]
pub struct WindowedSplit {
    pub x: Vec<f32>, // [n_windows * seq_len * dim] row-major
    pub y: Vec<f32>, // [n_windows * pred_len * dim] row-major
    pub n_windows: usize,
}

// ready stage fn
/// The ready stage: normalize, split, window.
/// Consumes PartitionedBundle, produces ReadyBundle.
pub fn ready(bundle: PartitionedBundle, config: ReadyConfig) -> Result<ReadyBundle, NumericFrameError> {
    let PartitionedBundle {
        frame,
        source_descriptor,
        feature_names,
        indices,
        train_stats,
        sigma,
        noise_seed,
        noise_draw_ref,
        noise_recipe,
        ..
    } = bundle;

    // ── 1. NumericFrame → contiguous f32 [N, D] ──
    let (mut data, n_rows, dim) = frame_to_row_major_f32(&frame)?;

    // ── 2. Normalize entire array using train stats ──
    let (center, scale) = train_stats.center_and_spread(config.norm_method);
    normalize_in_place(&mut data, n_rows, dim, &center, &scale, config.eps);
    println!("⚖️ Applied '{:?}' normalization.", config.norm_method);

    // ── 3. Split into train/val/test — just borrow slices, no copy ──
    let train_slice = &data[indices.t0 * dim..indices.t1 * dim];
    let val_slice = &data[indices.v0 * dim..indices.v1 * dim];
    let test_slice = &data[indices.s0 * dim..indices.s1 * dim];

    // ── 4. Sliding window each split ──
    let train = sliding_window(train_slice, indices.train_count(), dim, config.seq_len, config.pred_len);
    let val = sliding_window(val_slice, indices.val_count(), dim, config.seq_len, config.pred_len);
    let test = sliding_window(test_slice, indices.test_count(), dim, config.seq_len, config.pred_len);

    println!(
        "🪟 Windowed: train={} val={} test={} windows (seq={}, pred={})",
        train.n_windows, val.n_windows, test.n_windows, config.seq_len, config.pred_len,
    );

    // ── 5. Optional bootstrap trials (ENBPI-style) on TRAIN windows ──
    // Bootstrap is defined over temporal window index, so we generate it
    // before any CI reshape changes sample indexing semantics.
    if config.channel_independent && config.bootstrap_n_trials > 0 {
        return Err(NumericFrameError::new(
            "bootstrap in ready is indexed on temporal windows; disable channel_independent here and apply CI after bootstrap",
        ));
    }
    let bootstrap_trials = if config.bootstrap_n_trials > 0 {
        generate_bootstrap_trials(
            train.n_windows,
            config.bootstrap_block_size,
            config.bootstrap_n_trials,
            config.bootstrap_seed,
            config.seq_len,
            config.pred_len,
        )
    } else {
        Vec::new()
    };

    // ── 6. Optional channel-independent reshape [B,S,D] → [B*D,S] ──
    // Shadow with new immutable bindings
    let (train, val, test) = if config.channel_independent {
        (
            reshape_windowed_split_channel_independent(train, config.seq_len, config.pred_len, dim),
            reshape_windowed_split_channel_independent(val, config.seq_len, config.pred_len, dim),
            reshape_windowed_split_channel_independent(test, config.seq_len, config.pred_len, dim),
        )
    } else {
        (train, val, test)
    };

    Ok(ReadyBundle {
        train,
        val,
        test,
        source_descriptor,
        feature_names,
        channels: dim,
        seq_len: config.seq_len,
        pred_len: config.pred_len,
        center,
        scale,
        norm_method: config.norm_method,
        sigma,
        noise_seed,
        noise_draw_ref,
        noise_recipe,
        channel_independent: config.channel_independent,
        bootstrap_trials,
    })
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// ready details
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// bootstrap helper

#[derive(Debug, Clone)]
pub struct BootstrapTrial {
    pub in_bag_indices: Vec<usize>,
    pub oob_indices: Vec<usize>,
    pub oob_block_mask: Vec<bool>,
}

// bootstrap helper fn (used by ready)
pub fn generate_bootstrap_trials(
    n_windows: usize,
    block_size: usize,
    n_trials: usize,
    seed: u64,
    seq_len: usize,
    pred_len: usize,
) -> Vec<BootstrapTrial> {
    let k = n_windows / block_size;
    let remainder = n_windows % block_size;

    // Guard band: a window's raw footprint is seq_len + pred_len rows.
    // Windows within this distance of a block edge share raw rows
    // with the neighboring block.
    let guard = seq_len + pred_len - 1;
    assert!(
        block_size > 2 * guard,
        "block_size={} too small: need > 2 * guard={} to have any clean interior OOB windows",
        block_size,
        2 * guard,
    );
    assert!(
        k >= 4,
        "Only {} blocks (N={}, block_size={}). Need ≥4.",
        k,
        n_windows,
        block_size
    );

    use rand::RngExt;
    use rand::SeedableRng; // brings random_range() into scope

    use rand_pcg::Pcg64;

    let mut rng = Pcg64::seed_from_u64(seed);

    (0..n_trials)
        .map(|_| {
            let chosen: Vec<usize> = (0..k).map(|_| rng.random_range(0..k)).collect();

            let mut in_bag_blocks = vec![false; k];
            for &b in &chosen {
                in_bag_blocks[b] = true;
            }

            let mut in_bag_indices = Vec::with_capacity(k * block_size + remainder);
            for &b in &chosen {
                let start = b * block_size;
                for w in start..start + block_size {
                    in_bag_indices.push(w);
                }
            }
            for w in k * block_size..n_windows {
                in_bag_indices.push(w);
            }

            // OOB: only INTERIOR windows of OOB blocks.
            // Skip `guard` windows at each edge to avoid raw-data overlap
            // with neighboring in-bag blocks.
            //
            //   block:  [----guard----|----clean interior----|----guard----]
            //            contaminated   genuinely unseen      contaminated
            let mut oob_indices = Vec::new();
            let oob_block_mask: Vec<bool> = in_bag_blocks.iter().map(|&b| !b).collect();
            for (block_idx, &is_oob) in oob_block_mask.iter().enumerate() {
                if is_oob {
                    let block_start = block_idx * block_size;
                    let block_end = block_start + block_size;

                    // Trim guard from both edges
                    let safe_start = block_start + guard;
                    let safe_end = block_end.saturating_sub(guard);

                    if safe_start < safe_end {
                        for w in safe_start..safe_end {
                            oob_indices.push(w);
                        }
                    }
                }
            }

            BootstrapTrial {
                in_bag_indices,
                oob_indices,
                oob_block_mask,
            }
        })
        .collect()
}

/// Tests for the ready stage: normalization, windowing, bootstrap, and CI reshape.
///
/// Why these tests exist:
/// - Sliding window must match naive indexing: window[i] = series[i..i+len].
///   An off-by-one here means every training sample is shifted by one timestep,
///   which silently trains the model to predict the wrong horizon.
/// - Parallel windowing must match sequential: rayon parallelism must not change
///   the output. A race condition here produces non-deterministic training data.
/// - Bootstrap guard bands prevent OOB (out-of-bag) samples from overlapping
///   with the window interior — without guard bands, the OOB residuals used
///   for conformal prediction are not truly out-of-sample.
/// - Channel-independent reshape (B,S,D → B*D,S,1) must transpose correctly
///   or each channel sees another channel's data during training.
/// - CI mode and bootstrap trials are mutually exclusive — allowing both would
///   produce undefined behavior in the training loop.
#[cfg(test)]
mod tests {
    use super::*;
    use crate::stage_clean::KnownMeta;
    use crate::stage_noise::{NoiseRecipe, NoisyBundle, SourceDescriptor};
    use crate::stage_partition::{PartitionConfig, PartitionStats, partition};

    fn make_traceable_f32(n_rows: usize, dim: usize) -> Vec<f32> {
        (0..n_rows)
            .flat_map(|t| (0..dim).map(move |d| (t * 10 + d) as f32))
            .collect()
    }

    fn make_traceable_frame(n_rows: usize, dim: usize) -> NumericFrame {
        let cols: Vec<NumericColumn> = (0..dim)
            .map(|d| {
                let vals: Vec<f64> = (0..n_rows).map(|t| (t * 10 + d) as f64).collect();
                NumericColumn::new(format!("ch_{}", d).into(), &vals)
            })
            .collect();
        NumericFrame::new(n_rows, cols).unwrap()
    }

    fn make_tiny_frame() -> NumericFrame {
        let n = 120;
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

    fn partitioned_bundle_from(frame: NumericFrame) -> crate::stage_partition::PartitionedBundle {
        let feature_names = frame.column_names();
        let channels = frame.width();
        let noisy = NoisyBundle {
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
        };

        partition(noisy, PartitionConfig::p70_10_20()).unwrap()
    }

    #[test]
    fn frame_to_row_major_layout_is_traceable() {
        let frame = NumericFrame::new(
            3,
            vec![
                NumericColumn::new("a".into(), &[1.0f64, 2.0, 3.0]),
                NumericColumn::new("b".into(), &[10.0f64, 20.0, 30.0]),
            ],
        )
        .unwrap();
        let (data, n, d) = frame_to_row_major_f32(&frame).unwrap();
        assert_eq!((n, d), (3, 2));
        assert!((data[0] - 1.0).abs() < 1e-6);
        assert!((data[1] - 10.0).abs() < 1e-6);
        assert!((data[2] - 2.0).abs() < 1e-6);
        assert!((data[3] - 20.0).abs() < 1e-6);
    }

    #[test]
    fn frame_to_row_major_traceable_values() {
        let (data, n, d) = frame_to_row_major_f32(&make_traceable_frame(5, 3)).unwrap();
        assert_eq!((n, d), (5, 3));
        for t in 0..5 {
            for d_i in 0..3 {
                let expected = (t * 10 + d_i) as f32;
                assert!((data[t * 3 + d_i] - expected).abs() < 0.5);
            }
        }
    }

    #[test]
    fn normalize_in_place_basic() {
        let mut data = vec![10.0f32, 20.0, 12.0, 22.0, 14.0, 24.0, 16.0, 26.0];
        normalize_in_place(&mut data, 4, 2, &[13.0, 23.0], &[2.0, 2.0], 1e-8);
        assert!((data[0] - -1.5).abs() < 1e-5);
        assert!((data[7] - 1.5).abs() < 1e-5);
    }

    #[test]
    fn normalize_in_place_eps_guard() {
        let mut data = vec![5.0f32, 10.0f32];
        normalize_in_place(&mut data, 2, 1, &[5.0], &[0.0], 1e-8);
        assert!(data[0].abs() < 1e-5);
        assert!(data[1].is_finite());
    }

    #[test]
    fn sliding_window_matches_naive_indexing() {
        let data = make_traceable_f32(6, 2);
        let split = sliding_window(&data, 6, 2, 3, 2);
        assert_eq!(split.n_windows, 2);
        for w in 0..split.n_windows {
            for s in 0..3 {
                for d in 0..2 {
                    assert_eq!(split.x[(w * 3 + s) * 2 + d], data[(w + s) * 2 + d]);
                }
            }
            for h in 0..2 {
                for d in 0..2 {
                    assert_eq!(split.y[(w * 2 + h) * 2 + d], data[(w + 3 + h) * 2 + d]);
                }
            }
        }
    }

    #[test]
    fn sliding_window_parallel_matches_sequential() {
        let data = make_traceable_f32(50, 3);
        let seq = sliding_window(&data, 50, 3, 5, 3);
        let par = sliding_window_parallel(&data, 50, 3, 5, 3);
        assert_eq!(seq.n_windows, par.n_windows);
        assert_eq!(seq.x, par.x);
        assert_eq!(seq.y, par.y);
    }

    #[test]
    fn reshape_ci_transposes_bsd_to_bds() {
        let data: Vec<f32> = vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, // w0
            7.0, 8.0, 9.0, 10.0, 11.0, 12.0, // w1
        ];
        let out = reshape_channel_independent(&data, 2, 3, 2);
        assert_eq!(&out[0..3], &[1.0, 3.0, 5.0]);
        assert_eq!(&out[3..6], &[2.0, 4.0, 6.0]);
        assert_eq!(&out[6..9], &[7.0, 9.0, 11.0]);
        assert_eq!(&out[9..12], &[8.0, 10.0, 12.0]);
    }

    #[test]
    fn reshape_ci_dim1_is_noop() {
        let data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        assert_eq!(reshape_channel_independent(&data, 2, 3, 1), data);
    }

    #[test]
    fn bootstrap_oob_is_interior_only() {
        let seq_len = 3;
        let pred_len = 2;
        let guard = seq_len + pred_len - 1;
        let trials = generate_bootstrap_trials(100, 20, 5, 77, seq_len, pred_len);
        for t in trials {
            for w in t.oob_indices {
                let pos = w % 20;
                assert!(pos >= guard);
                assert!(pos < 20 - guard);
            }
        }
    }

    #[test]
    fn bootstrap_seed_is_deterministic() {
        let a = generate_bootstrap_trials(200, 20, 3, 42, 3, 2);
        let b = generate_bootstrap_trials(200, 20, 3, 42, 3, 2);
        assert_eq!(a[0].in_bag_indices, b[0].in_bag_indices);
    }

    #[test]
    fn ready_stage_full_pipeline_shapes() {
        let partitioned = partitioned_bundle_from(make_tiny_frame());
        let dim = partitioned.channels;
        let out = ready(partitioned, ReadyConfig::new(5, 3)).unwrap();

        assert!(out.train.n_windows > 0);
        assert!(out.val.n_windows > 0);
        assert!(out.test.n_windows > 0);
        assert_eq!(out.channels, dim);
        assert_eq!(out.train.x.len(), out.train.n_windows * 5 * dim);
    }

    #[test]
    fn ready_channel_independent_reshapes_all_splits() {
        let partitioned = partitioned_bundle_from(make_tiny_frame());
        let dim = partitioned.channels;

        let cfg = ReadyConfig {
            channel_independent: true,
            ..ReadyConfig::new(5, 3)
        };
        let out = ready(partitioned, cfg).unwrap();

        assert_eq!(out.train.x.len(), out.train.n_windows * 5);
        assert_eq!(out.train.y.len(), out.train.n_windows * 3);
        assert_eq!(out.val.x.len(), out.val.n_windows * 5);
        assert_eq!(out.test.x.len(), out.test.n_windows * 5);
        assert!(out.train.n_windows % dim == 0);
        assert!(out.val.n_windows % dim == 0);
        assert!(out.test.n_windows % dim == 0);
    }

    #[test]
    fn ready_rejects_ci_with_bootstrap_trials() {
        let partitioned = partitioned_bundle_from(make_tiny_frame());
        let cfg = ReadyConfig {
            channel_independent: true,
            bootstrap_n_trials: 3,
            ..ReadyConfig::new(5, 3)
        };

        let err = ready(partitioned, cfg).unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("disable channel_independent"));
    }

    #[test]
    fn ready_bundle_tag_includes_noise_and_ci_only_when_needed() {
        let base = ready(partitioned_bundle_from(make_tiny_frame()), ReadyConfig::new(5, 3)).unwrap();
        assert_eq!(base.tag(), "etth1__drop_and_impute__noise_0.1__uniform__seed_42");

        let ci = ready(
            partitioned_bundle_from(make_tiny_frame()),
            ReadyConfig {
                channel_independent: true,
                ..ReadyConfig::new(5, 3)
            },
        )
        .unwrap();
        assert_eq!(ci.tag(), "etth1__drop_and_impute__noise_0.1__uniform__seed_42__ci");
    }

    #[test]
    fn partition_stats_round_trip_for_norm_center_scale() {
        let frame = make_tiny_frame();
        let stats = PartitionStats::from_frame_slice(&frame, 0, 70).unwrap();
        let (center, scale) = stats.center_and_spread(NormMethod::Standard);
        assert_eq!(center.len(), frame.width());
        assert_eq!(scale.len(), frame.width());
    }
}
