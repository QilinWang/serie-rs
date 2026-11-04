//! Read the grouped `dgp_bank.safetensors` artifact that Python `serie` writes
//! (`serie.api.export_dgp_bank_safetensors`, schema `serie.dgp_bank_safetensors.v1`).
//!
//! This is the Rust *consumer* side of that file — the serie-rs backlog item
//! "consume serie-style draw banks" (see `agentic-design.md` → Serie Parity
//! Backlog). One safetensors file holds:
//!   - `clean`  `[N_trajectory, T, D]` — clean trajectory draws,
//!   - `noise`  `[N_noise, T, D]`      — reusable unit-noise draws,
//!   - optional `noisy_sigma_*`        — materialized noisy cells `[T, D]`,
//! plus the run metadata (seeds, sigmas, feature names, …) in the header.
//!
//! serie-rs both READS this file ([`load_dgp_bank_safetensors`]) and WRITES it
//! ([`write_dgp_bank_safetensors`], used by `serie-rs generate`). The contract
//! mirrors `serie.load_dgp_bank_safetensors`: require `clean` + `noise`, parse
//! `feature_names` from a JSON string list, default to `f0..f{D-1}` if absent.

use std::collections::{BTreeMap, HashMap};
use std::error::Error;
use std::path::{Path, PathBuf};

use ndarray::{ArrayD, IxDyn};
use safetensors::tensor::TensorView;
use safetensors::{Dtype, SafeTensors};

use crate::controlled_linear::{
    ActionIndependentGammaAttenuationCanary, ActionIndependentGaussianAttenuationCanary, Level0ACanary,
};
use crate::controlled_pendulum::{
    ControlledPendulum, PendulumState, SinusoidalTorqueDriver, sinusoidally_forced_trajectory,
};
use crate::draw_protocol::{DgpDrawBank, spawn_draw_seeds};
#[cfg(feature = "known-bank")]
use crate::stage_clean::{
    DEFAULT_DROP_COL_HOUR_THRESHOLD, DEFAULT_DROP_COL_THRESHOLD, DEFAULT_SENTINEL_VALUE_TOL, ImputeMethod, KnownMeta,
};
use crate::stage_synthetic::{TrajectorySpec, generate_timestep_major};

type Fallible<T> = Result<T, Box<dyn Error + Send + Sync>>;

/// One loaded `dgp_bank.safetensors`. Mirrors Python `serie.DGPBankTensors`.
#[derive(Debug, Clone)]
pub struct DgpBankTensors {
    /// The file this was read from.
    pub path: PathBuf,
    /// Every named tensor, any rank: `clean`/`noise` are 3-D `[draw, T, D]`,
    /// `noisy_sigma_*` are 2-D `[T, D]`. Kept as dynamic-dim f32 arrays.
    pub tensors: BTreeMap<String, ArrayD<f32>>,
    /// String header metadata exactly as serie wrote it (schema, seeds, sigmas, …).
    /// Values serie stored as JSON are kept verbatim — parse them as needed.
    pub metadata: HashMap<String, String>,
    /// Feature (channel) names for the D axis.
    pub feature_names: Vec<String>,
}

impl DgpBankTensors {
    /// Clean trajectory draws, `[N_trajectory, T, D]`.
    pub fn clean(&self) -> &ArrayD<f32> {
        &self.tensors["clean"]
    }
    /// Reusable unit-noise draws, `[N_noise, T, D]`.
    pub fn noise(&self) -> &ArrayD<f32> {
        &self.tensors["noise"]
    }
    /// Names of the materialized noisy cells (`noisy_sigma_*`), sorted.
    pub fn noisy_tensor_names(&self) -> Vec<&str> {
        self.tensors
            .keys()
            .map(String::as_str)
            .filter(|name| name.starts_with("noisy_sigma_"))
            .collect()
    }
}

/// Load a grouped `dgp_bank.safetensors` artifact (the file
/// `serie.export_dgp_bank_safetensors` writes). Read-only.
pub fn load_dgp_bank_safetensors(path: impl AsRef<Path>) -> Fallible<DgpBankTensors> {
    let path = path.as_ref().to_path_buf();
    let bytes = std::fs::read(&path)?;
    from_bytes(&bytes, path)
}

/// Read ONLY the header metadata of a `dgp_bank.safetensors` file: the 8-byte
/// little-endian header length, then the header JSON's `__metadata__` string
/// map (empty map when absent). No tensor data is read — provenance readers
/// (the fern-run parent copies the bank's `source_asset_sha` pin into a
/// `data_assets` run_event) should not pay for a full bank load.
pub fn load_dgp_bank_metadata(path: impl AsRef<Path>) -> Fallible<HashMap<String, String>> {
    use std::io::Read;

    let mut file = std::fs::File::open(path.as_ref())?;
    let mut header_len_bytes = [0u8; 8];
    file.read_exact(&mut header_len_bytes)?;
    let header_len = usize::try_from(u64::from_le_bytes(header_len_bytes))?;
    let mut header_bytes = vec![0u8; header_len];
    file.read_exact(&mut header_bytes)?;
    let header: serde_json::Value = serde_json::from_slice(&header_bytes)?;
    Ok(match header.get("__metadata__") {
        None => HashMap::new(),
        Some(metadata) => serde_json::from_value(metadata.clone())?,
    })
}

/// Parse an in-memory `dgp_bank.safetensors` buffer (the testable core).
fn from_bytes(bytes: &[u8], path: PathBuf) -> Fallible<DgpBankTensors> {
    // The `__metadata__` header string map.
    let (_, header) = SafeTensors::read_metadata(bytes)?;
    let metadata = header.metadata().clone().unwrap_or_default();

    // Every named tensor, read as dynamic-dim f32 (clean/noise are 3-D, noisy 2-D).
    let safetensors = SafeTensors::deserialize(bytes)?;
    let mut tensors: BTreeMap<String, ArrayD<f32>> = BTreeMap::new();
    for name in safetensors.names() {
        let view = safetensors.tensor(name)?;
        if view.dtype() != Dtype::F32 {
            return Err(format!("dgp_bank tensor {name:?} has dtype {:?}, expected F32", view.dtype()).into());
        }
        let shape: Vec<usize> = view.shape().to_vec();
        let floats: Vec<f32> = view
            .data()
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect();
        let array =
            ArrayD::from_shape_vec(IxDyn(&shape), floats).map_err(|e| format!("dgp_bank tensor {name:?}: {e}"))?;
        tensors.insert(name.to_string(), array);
    }

    // Required tensors (same check serie's loader makes).
    for required in ["clean", "noise"] {
        if !tensors.contains_key(required) {
            return Err(format!("{} missing required tensor {required:?}", path.display()).into());
        }
    }

    // D = last axis of `clean`.
    let d = *tensors["clean"].shape().last().ok_or("clean tensor has no axes")?;

    // feature_names: serie stores it as a JSON string list; default to f0..f{D-1}.
    let feature_names: Vec<String> = match metadata.get("feature_names") {
        None => (0..d).map(|i| format!("f{i}")).collect(),
        Some(raw) => {
            serde_json::from_str(raw).map_err(|e| format!("feature_names metadata is not a JSON string list: {e}"))?
        }
    };
    if feature_names.len() != d {
        return Err(format!("feature_names length {} does not match D={d}", feature_names.len()).into());
    }

    Ok(DgpBankTensors {
        path,
        tensors,
        metadata,
        feature_names,
    })
}

// ── writer (serie-rs as the PRODUCER side) ──────────────────────────────────

/// Serialize a grouped `dgp_bank.safetensors` buffer (the testable core of
/// [`write_dgp_bank_safetensors`]).
fn dgp_bank_to_bytes(
    clean: &ArrayD<f32>,
    noise: &ArrayD<f32>,
    noisy: &[(String, ArrayD<f32>)],
    extra_tensors: &[(String, ArrayD<f32>)],
    feature_names: &[String],
    extra_metadata: &[(String, String)],
) -> Fallible<Vec<u8>> {
    // Each TensorView borrows its byte buffer, so materialize every buffer first,
    // then build the views referencing them (no realloc can invalidate a borrow).
    let mut buffers: Vec<(String, Vec<usize>, Vec<u8>)> = Vec::new();
    for (name, array) in std::iter::once(("clean", clean))
        .chain(std::iter::once(("noise", noise)))
        .chain(noisy.iter().map(|(name, array)| (name.as_str(), array)))
        .chain(extra_tensors.iter().map(|(name, array)| (name.as_str(), array)))
    {
        let bytes: Vec<u8> = array.iter().flat_map(|x| x.to_le_bytes()).collect();
        buffers.push((name.to_string(), array.shape().to_vec(), bytes));
    }

    let data: BTreeMap<String, TensorView> = buffers
        .iter()
        .map(|(name, shape, bytes)| Ok((name.clone(), TensorView::new(Dtype::F32, shape.clone(), bytes)?)))
        .collect::<Fallible<_>>()?;

    let mut metadata: HashMap<String, String> = HashMap::new();
    metadata.insert("schema".to_string(), "serie.dgp_bank_safetensors.v1".to_string());
    metadata.insert("feature_names".to_string(), serde_json::to_string(feature_names)?);
    for (key, value) in extra_metadata {
        metadata.insert(key.clone(), value.clone());
    }

    Ok(safetensors::serialize(&data, Some(metadata))?)
}

/// Write a grouped `dgp_bank.safetensors` artifact in serie's schema
/// (`serie.dgp_bank_safetensors.v1`): `clean` `[N_trajectory, T, D]`, `noise`
/// `[N_noise, T, D]`, optional `noisy_sigma_*` `[T, D]`, plus `feature_names` and
/// any extra header metadata. The inverse of [`load_dgp_bank_safetensors`].
pub fn write_dgp_bank_safetensors(
    path: impl AsRef<Path>,
    clean: &ArrayD<f32>,
    noise: &ArrayD<f32>,
    noisy: &[(String, ArrayD<f32>)],
    extra_tensors: &[(String, ArrayD<f32>)],
    feature_names: &[String],
    extra_metadata: &[(String, String)],
) -> Fallible<()> {
    let bytes = dgp_bank_to_bytes(clean, noise, noisy, extra_tensors, feature_names, extra_metadata)?;
    std::fs::write(path, bytes)?;
    Ok(())
}

// ── generator (serie-rs as the BANK GENERATOR for the Rust run parent) ──────

const ROSSLER_IC_HOLDOUT_PREFIX: &str = "rossler_ic_holdout_";
const ROSSLER_IC_HOLDOUT_DELTA: f64 = 0.01;

/// Return one corner of the complete ±0.01 cube around x0=(1,1,1).
///
/// The eight corners are the confirmation units. They are fixed before any
/// result is inspected, so adding trajectories cannot become a search for
/// initial conditions that preserve the discovery-grid reversal.
fn rossler_initial_condition_holdout(source_name: &str) -> Option<[f64; 3]> {
    let ordinal = source_name
        .strip_prefix(ROSSLER_IC_HOLDOUT_PREFIX)?
        .parse::<usize>()
        .ok()?;
    if ordinal >= 8 {
        return None;
    }
    let signed_delta = |bit: usize| {
        if ordinal & (1 << bit) == 0 {
            -ROSSLER_IC_HOLDOUT_DELTA
        } else {
            ROSSLER_IC_HOLDOUT_DELTA
        }
    };
    Some([1.0 + signed_delta(2), 1.0 + signed_delta(1), 1.0 + signed_delta(0)])
}

const ROSSLER_IC_RANDOM_PREFIX: &str = "rossler_ic_random_";
const ROSSLER_IC_RANDOM_COUNT: usize = 16;
/// Fixed before any of these trajectories was generated, so the drawn set is a
/// predeclaration and not a search. The value is the date it was registered.
const ROSSLER_IC_RANDOM_SEED: u64 = 2026_07_30;

/// Return one initial condition drawn uniformly from the ±0.01 cube around
/// x0=(1,1,1).
///
/// WHY THIS EXISTS BESIDE THE CORNER HOLDOUT. The eight corners are the extreme
/// points of that cube and all sit at the same distance from the historical x0,
/// so they are a designed set rather than a sample. They are also only eight
/// units, and an exact two-sided sign test on eight paired units demands
/// UNANIMITY to reject at the five per cent level: seven of eight already
/// returns 0.0703. These sixteen draws sample the interior of the same
/// neighbourhood and make the reversal rate an estimable quantity.
///
/// The stream is SplitMix64 from a fixed seed, advanced three values per unit,
/// so the whole set is reproducible from this function alone and every drawn
/// coordinate is pinned by `rossler_random_initial_conditions_are_predeclared`.
fn rossler_initial_condition_random(source_name: &str) -> Option<[f64; 3]> {
    let ordinal = source_name
        .strip_prefix(ROSSLER_IC_RANDOM_PREFIX)?
        .parse::<usize>()
        .ok()?;
    if ordinal >= ROSSLER_IC_RANDOM_COUNT {
        return None;
    }
    let mut state = ROSSLER_IC_RANDOM_SEED;
    let mut next = || {
        state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    };
    let mut initial_condition = [0.0_f64; 3];
    for unit in 0..=ordinal {
        for axis in 0..3 {
            // The top 53 bits give a uniform on [0,1) exactly representable in
            // f64; mapping to [-1,1) then scaling keeps the cube symmetric.
            let unit_interval = (next() >> 11) as f64 / (1_u64 << 53) as f64;
            if unit == ordinal {
                initial_condition[axis] = 1.0 + ROSSLER_IC_HOLDOUT_DELTA * (2.0 * unit_interval - 1.0);
            }
        }
    }
    Some(initial_condition)
}

const LORENZ_IC_PREFIX: &str = "lorenz_ic_";
const LORENZ_IC_COUNT: usize = 16;
/// Fixed before any of these trajectories existed, so the drawn set is a
/// predeclaration and not a search. The value is the date it was registered.
const LORENZ_IC_SEED: u64 = 2026_08_11;
/// Half-width of the cube the initial conditions are drawn from, in state
/// units. Rossler's holdout uses 0.01 around x0=(1,1,1); Lorenz-63 lives on an
/// attractor roughly twenty units across, so the same ABSOLUTE offset would be
/// a thousand times smaller RELATIVE to the state and every draw would follow
/// the same orbit for the whole record. 1.0 puts the sixteen starts genuinely
/// apart on the attractor while staying far inside its basin.
const LORENZ_IC_DELTA: f64 = 1.0;

/// One Lorenz-63 initial condition drawn uniformly from the cube of half-width
/// `LORENZ_IC_DELTA` around the registered x0.
///
/// WHY THIS EXISTS. Every chaotic result in the campaign so far rests on ONE
/// trajectory per world, with four frozen measurement draws on top of it. That
/// design asks what the instrument does and holds the realisation fixed, so a
/// finding cannot be separated from the particular orbit it was measured on --
/// and on Rossler the reversal did move when the trajectory changed. Sixteen
/// independent starts make "does this transport across realisations" an
/// estimable question rather than an assumption.
///
/// SplitMix64 from a fixed seed, advanced three values per unit, so the whole
/// set is reproducible from this function alone.
fn lorenz_initial_condition_random(source_name: &str) -> Option<[f64; 3]> {
    let ordinal = source_name.strip_prefix(LORENZ_IC_PREFIX)?.parse::<usize>().ok()?;
    if ordinal >= LORENZ_IC_COUNT {
        return None;
    }
    let base = TrajectorySpec::lorenz().x0;
    let mut state = LORENZ_IC_SEED;
    let mut next = || {
        state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    };
    let mut initial_condition = [0.0_f64; 3];
    for unit in 0..=ordinal {
        for axis in 0..3 {
            let unit_interval = (next() >> 11) as f64 / (1_u64 << 53) as f64;
            if unit == ordinal {
                initial_condition[axis] = base[axis] + LORENZ_IC_DELTA * (2.0 * unit_interval - 1.0);
            }
        }
    }
    Some(initial_condition)
}

/// Map a full catalog source name (e.g. "ou_param_shock") to its serie-rs
/// `TrajectorySpec` preset. The name vocabulary is the fern-run recipe/cell
/// `source_name`; `TrajectorySpec::tag()` is NOT used as the lookup key
/// because its `{:?}`-derived base ("seasonalar") can differ from the catalog
/// name ("seasonal_ar").
fn trajectory_spec_for_source(source_name: &str) -> Fallible<TrajectorySpec> {
    if let Some(initial_condition) =
        rossler_initial_condition_holdout(source_name).or_else(|| rossler_initial_condition_random(source_name))
    {
        let mut spec = TrajectorySpec::rossler();
        spec.x0 = initial_condition.to_vec();
        return Ok(spec);
    }
    if let Some(initial_condition) = lorenz_initial_condition_random(source_name) {
        let mut spec = TrajectorySpec::lorenz();
        spec.x0 = initial_condition.to_vec();
        return Ok(spec);
    }
    let spec = match source_name {
        "lorenz" => TrajectorySpec::lorenz(),
        "lorenz_param_shock" => TrajectorySpec::lorenz_param_shock(),
        "lorenz_state_shock" => TrajectorySpec::lorenz_state_shock(),
        "lorenz_switch" => TrajectorySpec::lorenz_switch(),
        "lorenz96" => TrajectorySpec::lorenz96(),
        "lorenz96_switch" => TrajectorySpec::lorenz96_switch(),
        "rossler" => TrajectorySpec::rossler(),
        "rossler_param_shock" => TrajectorySpec::rossler_param_shock(),
        "chua" => TrajectorySpec::chua(),
        "chua_param_shock" => TrajectorySpec::chua_param_shock(),
        "chua_switch" => TrajectorySpec::chua_switch(),
        "ou" => TrajectorySpec::ou(),
        "ou_param_shock" => TrajectorySpec::ou_param_shock(),
        "doublewell" => TrajectorySpec::doublewell(),
        "doublewell_barrier_collapse" => TrajectorySpec::doublewell_barrier_collapse(),
        "doublewell_param_shock" => TrajectorySpec::doublewell_param_shock(),
        "doublewell_switch" => TrajectorySpec::doublewell_switch(),
        "slds" => TrajectorySpec::slds(),
        "slds_param_shock" => TrajectorySpec::slds_param_shock(),
        "slds_switch" => TrajectorySpec::slds_switch(),
        "seasonal_ar" => TrajectorySpec::seasonal_ar(),
        "seasonal_ar_param_shock" => TrajectorySpec::seasonal_ar_param_shock(),
        "garch" => TrajectorySpec::garch(),
        "garch_param_shock" => TrajectorySpec::garch_param_shock(),
        "lg_ssm" => TrajectorySpec::lg_ssm(),
        "lg_ssm_switch" => TrajectorySpec::lg_ssm_switch(),
        "lg_ssm_regime_switch" => TrajectorySpec::lg_ssm_regime_switch(),
        other => {
            return Err(format!(
                "no serie-rs generator preset for source {other:?}; generate_dgp_bank \
                 covers the wired synthetic sources and the prepared known-data assets \
                 (only ks/ks_param_shock stay Python-generated)"
            )
            .into());
        }
    };
    Ok(spec)
}

/// Whether this build owns a Rust materializer for `source_name`.
///
/// This queries the same dispatch tables used by [`generate_dgp_bank`]; it is
/// not a second catalog. Fern uses it to prove that every source advertised as
/// `serie_rs` capability reaches a real synthetic generator or prepared-data
/// reader before a Recipe is registered.
pub fn source_handler_is_registered(source_name: &str) -> bool {
    source_name == "controlled_pendulum"
        || source_name == "level0a_controlled_lgssm"
        || source_name == ActionIndependentGammaAttenuationCanary::SOURCE_NAME
        || source_name == ActionIndependentGaussianAttenuationCanary::SOURCE_NAME
        || trajectory_spec_for_source(source_name).is_ok()
        || is_known_source_name(source_name)
}

/// Observation-noise law for the frozen unit-variance bank draws. Mirrors the
/// serie Python `noise_kind` vocabulary; every law is variance-matched to one
/// so relative doses stay train-std-scale interventions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NoiseKind {
    Gaussian,
    /// Laplace with scale b = 1/sqrt(2), so variance 2b^2 = 1 like the
    /// Gaussian path (same convention as serie Python `rng.laplace`).
    /// Excess kurtosis 3 — the middle rung of the tail-weight ladder
    /// gaussian (0) -> laplace (3) -> student_t3 (infinite).
    Laplace,
    /// Student-t with nu=3, variance-matched: each draw is multiplied by
    /// sqrt((nu-2)/nu) = sqrt(1/3) so it has variance one like the Gaussian
    /// path; relative doses stay train-std-scale interventions.
    StudentT3,
    /// Huber's gross-error model, variance-matched. With probability
    /// `CONTAMINATION_RATE` a channel--time reading's standard normal draw is multiplied
    /// by `CONTAMINATION_SCALE`; the whole draw is then divided by
    /// sqrt((1-eps) + eps*k^2) so the variance is one like every other law.
    /// This varies how SPARSE the tail mass is, not how heavy a symmetric
    /// tail is, so it is orthogonal to the Laplace/Student-t3 ladder rather
    /// than an extension of it.
    Contaminated,
}

/// Fraction of channel--time readings that receive the enlarged value. Pinned before any
/// result was seen; it is not a tuning knob.
pub const CONTAMINATION_RATE: f64 = 0.02;
/// Multiplier applied to a contaminated timestamp's standard normal draw.
pub const CONTAMINATION_SCALE: f64 = 10.0;

impl NoiseKind {
    /// The bank-metadata / serie Python `noise_kind` string.
    pub fn value(self) -> &'static str {
        match self {
            Self::Gaussian => "gaussian",
            Self::Laplace => "laplace",
            Self::StudentT3 => "student_t3",
            Self::Contaminated => "contaminated",
        }
    }
}

/// A nonzero stationary AR(1) coefficient.
///
/// The private field is the type boundary: after a raw config float has been
/// converted once, downstream Rust code cannot construct phi=0, NaN, infinity,
/// or a non-stationary `|phi| >= 1` by accident.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StationaryAr1Phi(f64);

impl StationaryAr1Phi {
    pub fn value(self) -> f64 {
        self.0
    }
}

impl TryFrom<f64> for StationaryAr1Phi {
    type Error = String;

    fn try_from(phi: f64) -> Result<Self, Self::Error> {
        if !phi.is_finite() || phi == 0.0 || !(-1.0..1.0).contains(&phi) {
            return Err(format!(
                "AR(1) phi must be finite, nonzero, and lie in (-1, 1), got {phi}"
            ));
        }
        Ok(Self(phi))
    }
}

/// Complete observation-noise process used to build one frozen bank.
///
/// AR(1) is the TEMPORAL structure, not another spelling of “Gaussian”.
/// This campaign uses Gaussian innovations specifically so
///
///   eps_t = phi eps_{t-1} + sqrt(1-phi^2) eta_t, eta_t ~ N(0,1)
///
/// keeps the stationary one-time marginal exactly N(0,1) while changing only
/// temporal dependence. An AR(1) process can have other innovation laws, but
/// that is a different factorial experiment because its stationary marginal
/// is generally not the named innovation law.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ObservationNoiseProcess {
    Iid(NoiseKind),
    Ar1WithGaussianInnovations(StationaryAr1Phi),
}

impl ObservationNoiseProcess {
    /// Validate the legacy wire pair once, at the config-to-domain boundary.
    ///
    /// `phi=0` preserves every existing iid bank and hash. A nonzero phi is
    /// accepted only with the Gaussian innovation law used by this isolated
    /// correlation experiment; invalid combinations cannot exist after this
    /// constructor returns.
    pub fn from_wire(noise_kind: NoiseKind, noise_ar1_phi: f64) -> Fallible<Self> {
        if noise_ar1_phi == 0.0 {
            return Ok(Self::Iid(noise_kind));
        }
        if noise_kind != NoiseKind::Gaussian {
            return Err(format!(
                "AR(1) temporal dependence with phi={noise_ar1_phi} is paired with Gaussian \
                 innovations in this protocol; got innovation law {}",
                noise_kind.value()
            )
            .into());
        }
        Ok(Self::Ar1WithGaussianInnovations(
            StationaryAr1Phi::try_from(noise_ar1_phi).map_err(|error| error.to_string())?,
        ))
    }

    pub fn innovation_kind(self) -> NoiseKind {
        match self {
            Self::Iid(noise_kind) => noise_kind,
            Self::Ar1WithGaussianInnovations(_) => NoiseKind::Gaussian,
        }
    }

    pub fn ar1_phi(self) -> Option<StationaryAr1Phi> {
        match self {
            Self::Iid(_) => None,
            Self::Ar1WithGaussianInnovations(phi) => Some(phi),
        }
    }
}

/// Materialize `n_draws` frozen unit-noise draws as a row-major `[N*T*D]` f32
/// buffer, one PCG stream per spawned draw seed (serie-rs's own noise RNG).
pub fn unit_noise_draws(
    noise_kind: NoiseKind,
    seed_base: u64,
    n_draws: usize,
    timesteps: usize,
    channels: usize,
) -> Fallible<Vec<f32>> {
    use rand::SeedableRng;
    use rand_distr::{Distribution, StandardNormal, StudentT};
    use rand_pcg::Pcg64;

    let seeds = spawn_draw_seeds(seed_base, n_draws)?;
    let mut flat = Vec::with_capacity(n_draws * timesteps * channels);
    for seed in seeds {
        let mut rng = Pcg64::seed_from_u64(seed);
        match noise_kind {
            NoiseKind::Gaussian => {
                for _ in 0..timesteps * channels {
                    let value: f64 = StandardNormal.sample(&mut rng);
                    flat.push(value as f32);
                }
            }
            NoiseKind::Laplace => {
                // Variance-matched Laplace via inverse CDF of one uniform:
                // x = -b * sign(u - 1/2) * ln(1 - 2|u - 1/2|), b = 1/sqrt(2)
                // gives variance 2b^2 = 1. The clamp guards the measure-zero
                // u = 0 draw (ln 0) without touching any other value.
                use rand::RngExt;
                let scale = 1.0f64 / 2.0f64.sqrt();
                for _ in 0..timesteps * channels {
                    let uniform: f64 = rng.random();
                    let centered: f64 = uniform - 0.5;
                    let tail_mass: f64 = (1.0 - 2.0 * centered.abs()).max(f64::MIN_POSITIVE);
                    flat.push((-scale * centered.signum() * tail_mass.ln()) as f32);
                }
            }
            NoiseKind::StudentT3 => {
                // Variance-matched Student-t3: t3 has variance nu/(nu-2) = 3,
                // so each draw is scaled by sqrt(1/3) to unit variance.
                let student_t3 = StudentT::new(3.0).map_err(|e| format!("StudentT::new(3.0): {e}"))?;
                let scale = (1.0f64 / 3.0).sqrt();
                for _ in 0..timesteps * channels {
                    let value: f64 = student_t3.sample(&mut rng);
                    flat.push((value * scale) as f32);
                }
            }
            NoiseKind::Contaminated => {
                // eps_t = ((1 - Z_t) + k Z_t) eta_t / sqrt((1-eps) + eps k^2),
                // with Z_t ~ Bernoulli(eps) and eta_t ~ N(0,1). The divisor is
                // exactly E[multiplier^2]^(1/2), so the draw has unit variance
                // like every other law and a relative dose keeps its meaning.
                // Two uniforms per reading would be wasteful; one Bernoulli
                // and one normal is what the model needs.
                use rand::RngExt;
                let rate = CONTAMINATION_RATE;
                let spike = CONTAMINATION_SCALE;
                let normalizer = ((1.0 - rate) + rate * spike * spike).sqrt();
                for _ in 0..timesteps * channels {
                    let contaminated: bool = rng.random::<f64>() < rate;
                    let multiplier = if contaminated { spike } else { 1.0 };
                    let value: f64 = StandardNormal.sample(&mut rng);
                    flat.push((multiplier * value / normalizer) as f32);
                }
            }
        }
    }
    Ok(flat)
}

/// Impose AR(1) temporal correlation on unit-variance noise, in place.
///
/// Every noise law in this crate draws each timestep INDEPENDENTLY. That is an
/// assumption about the measurement process, not a fact about it: sensor
/// drift, thermal effects, and calibration offsets are correlated in time.
/// This is a separate axis from the marginal law — `noise_kind` says what a
/// single reading's error looks like, `phi` says how much this reading's error
/// remembers the last one.
///
///   eps_t = phi * eps_{t-1} + sqrt(1 - phi^2) * eta_t,   eta_t ~ the iid draw
///
/// The `sqrt(1 - phi^2)` factor is what makes the axis comparable: it holds
/// `Var(eps_t) = 1` for every phi, so a relative dose keeps meaning the same
/// multiple of the clean training standard deviation. Without it, raising phi
/// would raise the dose too, and the two effects could not be separated.
///
/// `eps_0` is the untouched first iid draw, which already has variance one —
/// that IS the stationary distribution of this recursion, so the series is
/// stationary from the first step and needs no burn-in. Starting from zero
/// instead would leave the opening timesteps under-dosed.
///
/// The innovation law does not appear in this function's signature. The
/// caller must first construct [`ObservationNoiseProcess`], whose AR(1)
/// variant can contain only Gaussian innovations. That makes the experimental
/// restriction a type invariant instead of a repeated runtime `if`.
fn impose_ar1_correlation(
    flat: &mut [f32],
    phi: StationaryAr1Phi,
    n_draws: usize,
    timesteps: usize,
    channels: usize,
) -> Fallible<()> {
    let expected = n_draws * timesteps * channels;
    if flat.len() != expected {
        return Err(format!("noise buffer is {} values, expected {expected}", flat.len()).into());
    }
    let phi = phi.value();
    let innovation_scale = (1.0 - phi * phi).sqrt() as f32;
    let phi = phi as f32;
    for draw in 0..n_draws {
        let base = draw * timesteps * channels;
        for channel in 0..channels {
            // Row-major [draw][timestep][channel], so one channel's series is
            // strided by `channels`; eps_0 is left as drawn.
            for step in 1..timesteps {
                let previous = flat[base + (step - 1) * channels + channel];
                let here = &mut flat[base + step * channels + channel];
                *here = phi * previous + innovation_scale * *here;
            }
        }
    }
    Ok(())
}

fn impose_temporal_dependence(
    flat: &mut [f32],
    noise_process: ObservationNoiseProcess,
    n_draws: usize,
    timesteps: usize,
    channels: usize,
) -> Fallible<()> {
    match noise_process {
        ObservationNoiseProcess::Iid(_) => Ok(()),
        ObservationNoiseProcess::Ar1WithGaussianInnovations(phi) => {
            impose_ar1_correlation(flat, phi, n_draws, timesteps, channels)
        }
    }
}

/// Generate one grouped `dgp_bank.safetensors` for a wired synthetic source:
/// spawn trajectory child seeds, integrate one clean trajectory per seed into
/// `clean [N_trajectory, T, D]`, draw frozen unit-variance noise under
/// `noise_kind` into `noise [N_noise, T, D]`, and write the file with the full
/// serie metadata header (seed bases, spawned draw-seed lists, noise law,
/// shapes) so Python's `serie.api.dgp_bank_from_tensors` can rebuild a
/// validated `DGPBank`.
///
/// Deterministic sources ignore the trajectory seed, so every clean draw is
/// identical — the same semantics as Python `serie.build_dgp_bank`; plan-level
/// draw designs keep `n_trajectory_draws = 1` for them.
///
/// Cross-engine note: the same child seeds feed a different RNG stack than
/// numpy, so this bank is NOT bitwise-identical to a Python-generated one.
/// The hashed run context carries the difference as `data_engine = "serie_rs"`.
///
/// Known-data sources (etth1/etth2/ettm1/ettm2, solar_energy(_subset),
/// hugadb_accel_emg, hugadb_walk_bicycling_switch) do not integrate anything:
/// with the `known-bank` feature enabled, the clean draw is the prepared IPC
/// asset under `known_data_root`, loaded the way Python `load_data(KnownMeta)`
/// loaded it. Their clean channels involve NO RNG, so unlike synthetic sources
/// they ARE bitwise-identical across engines; only the frozen noise stream
/// differs.
pub fn generate_dgp_bank(
    source_name: &str,
    trajectory_seed_base: u64,
    noise_seed_base: u64,
    n_trajectory_draws: usize,
    n_noise_draws: usize,
    noise_process: ObservationNoiseProcess,
    out_path: impl AsRef<Path>,
) -> Fallible<()> {
    if source_name == ActionIndependentGammaAttenuationCanary::SOURCE_NAME
        || source_name == ActionIndependentGaussianAttenuationCanary::SOURCE_NAME
    {
        return generate_radial_attenuation_dgp_bank(
            source_name,
            trajectory_seed_base,
            noise_seed_base,
            n_trajectory_draws,
            n_noise_draws,
            noise_process,
            out_path,
        );
    }
    if source_name == "controlled_pendulum" {
        return generate_controlled_pendulum_dgp_bank(
            trajectory_seed_base,
            noise_seed_base,
            n_trajectory_draws,
            n_noise_draws,
            noise_process,
            out_path,
        );
    }
    if source_name == "level0a_controlled_lgssm" {
        return generate_level0a_controlled_lgssm_dgp_bank(
            trajectory_seed_base,
            noise_seed_base,
            n_trajectory_draws,
            n_noise_draws,
            noise_process,
            out_path,
        );
    }
    if is_known_source_name(source_name) {
        #[cfg(feature = "known-bank")]
        {
            let asset = known_source_asset(source_name).expect("known source name has an asset mapping");
            return generate_known_dgp_bank(
                source_name,
                &asset,
                trajectory_seed_base,
                noise_seed_base,
                n_trajectory_draws,
                n_noise_draws,
                noise_process,
                out_path,
            );
        }
        #[cfg(not(feature = "known-bank"))]
        {
            return Err(format!(
                "known-data source {source_name:?} requires the serie-rs `known-bank` feature \
                 to read prepared IPC assets"
            )
            .into());
        }
    }
    let preset = trajectory_spec_for_source(source_name)?;
    let is_stochastic = preset.system.is_stochastic();
    let trajectory_draw_seeds = spawn_draw_seeds(trajectory_seed_base, n_trajectory_draws)?;

    let mut clean_flat: Vec<f32> = Vec::new();
    let mut timesteps = 0usize;
    let mut channels = 0usize;
    let mut feature_names: Vec<String> = Vec::new();
    for (draw_ordinal, draw_seed) in trajectory_draw_seeds.iter().enumerate() {
        let mut spec = preset.clone();
        spec.seed = *draw_seed;
        let trajectory = generate_timestep_major(spec)?;
        let names = trajectory.feature_names;
        let t = trajectory.timesteps;
        let d = trajectory.channels;
        let flat: Vec<f32> = trajectory
            .timestep_major_values
            .into_iter()
            .map(|value| value as f32)
            .collect();
        if draw_ordinal == 0 {
            timesteps = t;
            channels = d;
            feature_names = names;
        } else if (t, d) != (timesteps, channels) {
            return Err(format!(
                "trajectory draw {draw_ordinal} produced [T={t}, D={d}], expected [T={timesteps}, D={channels}]"
            )
            .into());
        }
        clean_flat.extend_from_slice(&flat);
    }
    let clean = ArrayD::from_shape_vec(IxDyn(&[n_trajectory_draws, timesteps, channels]), clean_flat)?;

    let noise_kind = noise_process.innovation_kind();
    let mut noise_flat = unit_noise_draws(noise_kind, noise_seed_base, n_noise_draws, timesteps, channels)?;
    impose_temporal_dependence(&mut noise_flat, noise_process, n_noise_draws, timesteps, channels)?;
    let noise = ArrayD::from_shape_vec(IxDyn(&[n_noise_draws, timesteps, channels]), noise_flat)?;

    let bank = DgpDrawBank::new(
        source_name,
        is_stochastic,
        trajectory_seed_base,
        noise_seed_base,
        n_trajectory_draws,
        n_noise_draws,
        timesteps,
        channels,
    )?;
    let mut extra_metadata = serie_bank_metadata(&bank, noise_process)?;
    if let Some(initial_condition) = rossler_initial_condition_holdout(source_name) {
        extra_metadata.push((
            "initial_condition".to_string(),
            serde_json::to_string(&initial_condition)?,
        ));
        extra_metadata.push((
            "initial_condition_design".to_string(),
            "complete_pm_0p01_cube_around_1_1_1".to_string(),
        ));
    }
    write_dgp_bank_safetensors(out_path, &clean, &noise, &[], &[], &feature_names, &extra_metadata)
}

/// Materialize the action-free Gamma(4,4) attenuation world.
///
/// One scale draw is shared by the two innovation coordinates at a timestep,
/// but scales are independent across timesteps. This gives a non-Gaussian
/// radial law without changing the process covariance or introducing action.
fn generate_radial_attenuation_dgp_bank(
    source_name: &str,
    trajectory_seed_base: u64,
    noise_seed_base: u64,
    n_trajectory_draws: usize,
    n_noise_draws: usize,
    noise_process: ObservationNoiseProcess,
    out_path: impl AsRef<Path>,
) -> Fallible<()> {
    use rand::SeedableRng;
    use rand_distr::{Distribution, Gamma, StandardNormal};
    use rand_pcg::Pcg64;

    const BURN_IN: usize = 512;
    const TIMESTEPS: usize = 25_000;
    const CHANNELS: usize = 2;

    let gamma = Gamma::new(
        ActionIndependentGammaAttenuationCanary::GAMMA_SHAPE,
        1.0 / ActionIndependentGammaAttenuationCanary::GAMMA_RATE,
    )?;
    let trajectory_draw_seeds = spawn_draw_seeds(trajectory_seed_base, n_trajectory_draws)?;
    let mut clean_values = Vec::with_capacity(n_trajectory_draws * TIMESTEPS * CHANNELS);
    for draw_seed in &trajectory_draw_seeds {
        let mut rng = Pcg64::seed_from_u64(*draw_seed);
        let mut state = [0.0_f64; CHANNELS];
        for timestep in 0..(BURN_IN + TIMESTEPS) {
            let radial_scale = if source_name == ActionIndependentGammaAttenuationCanary::SOURCE_NAME {
                gamma.sample(&mut rng)
            } else if source_name == ActionIndependentGaussianAttenuationCanary::SOURCE_NAME {
                1.0
            } else {
                return Err(format!("unknown radial attenuation source {source_name:?}").into());
            };
            for coordinate in &mut state {
                let gaussian: f64 = StandardNormal.sample(&mut rng);
                *coordinate = ActionIndependentGammaAttenuationCanary::DRIFT * *coordinate
                    + ActionIndependentGammaAttenuationCanary::SCALE * radial_scale.sqrt() * gaussian;
            }
            if timestep >= BURN_IN {
                clean_values.extend(state.iter().map(|value| *value as f32));
            }
        }
    }
    let clean = ArrayD::from_shape_vec(IxDyn(&[n_trajectory_draws, TIMESTEPS, CHANNELS]), clean_values)?;

    let noise_kind = noise_process.innovation_kind();
    let mut noise_values = unit_noise_draws(noise_kind, noise_seed_base, n_noise_draws, TIMESTEPS, CHANNELS)?;
    impose_temporal_dependence(&mut noise_values, noise_process, n_noise_draws, TIMESTEPS, CHANNELS)?;
    let noise = ArrayD::from_shape_vec(IxDyn(&[n_noise_draws, TIMESTEPS, CHANNELS]), noise_values)?;

    let bank = DgpDrawBank::new(
        source_name,
        true,
        trajectory_seed_base,
        noise_seed_base,
        n_trajectory_draws,
        n_noise_draws,
        TIMESTEPS,
        CHANNELS,
    )?;
    let mut metadata = serie_bank_metadata(&bank, noise_process)?;
    let (radial_scale_law, radial_gamma_shape, radial_gamma_rate) =
        if source_name == ActionIndependentGammaAttenuationCanary::SOURCE_NAME {
            ("gamma", "4", "4")
        } else {
            ("constant_one", "not_applicable", "not_applicable")
        };
    metadata.extend([
        ("state_transition".to_string(), "0.7*I2".to_string()),
        ("process_covariance".to_string(), "0.0225*I2".to_string()),
        ("radial_scale_law".to_string(), radial_scale_law.to_string()),
        ("radial_gamma_shape".to_string(), radial_gamma_shape.to_string()),
        ("radial_gamma_rate".to_string(), radial_gamma_rate.to_string()),
        ("radial_scale_dependence".to_string(), "independent_step".to_string()),
        ("action_conditioning".to_string(), "none".to_string()),
        ("burn_in_timesteps".to_string(), BURN_IN.to_string()),
    ]);
    write_dgp_bank_safetensors(
        out_path,
        &clean,
        &noise,
        &[],
        &[],
        &["x0".to_string(), "x1".to_string()],
        &metadata,
    )
}

/// Materialize one partially observed forced Pendulum together with its
/// separate transition-aligned action axis. `actions[draw,t]` drives
/// `clean[draw,t] -> clean[draw,t+1]`; the Python window stage preserves that
/// alignment when it creates past and planned-future action tensors.
fn generate_controlled_pendulum_dgp_bank(
    trajectory_seed_base: u64,
    noise_seed_base: u64,
    n_trajectory_draws: usize,
    n_noise_draws: usize,
    noise_process: ObservationNoiseProcess,
    out_path: impl AsRef<Path>,
) -> Fallible<()> {
    const STEPS: usize = 4096;
    let plant = ControlledPendulum::try_new(0.05, 0.15, 9.81, 0.8)?;
    let driver = SinusoidalTorqueDriver::try_new(0.6, 0.2)?;
    let trajectory_draw_seeds = spawn_draw_seeds(trajectory_seed_base, n_trajectory_draws)?;
    let mut clean_values = Vec::with_capacity(n_trajectory_draws * (STEPS + 1));
    let mut action_values = Vec::with_capacity(n_trajectory_draws * STEPS);
    for draw_seed in &trajectory_draw_seeds {
        let initial_phase = ((*draw_seed % 1_000_003) as f64 / 1_000_003.0) * std::f64::consts::TAU;
        let trajectory = sinusoidally_forced_trajectory(
            &plant,
            &driver,
            PendulumState {
                angle: 0.1,
                angular_velocity: -0.3,
            },
            initial_phase,
            STEPS,
        )?;
        clean_values.extend(trajectory.observed_angles.into_iter().map(|value| value as f32));
        action_values.extend(trajectory.actions.into_iter().map(|value| value as f32));
    }
    let clean = ArrayD::from_shape_vec(IxDyn(&[n_trajectory_draws, STEPS + 1, 1]), clean_values)?;
    let actions = ArrayD::from_shape_vec(IxDyn(&[n_trajectory_draws, STEPS, 1]), action_values)?;
    let noise_kind = noise_process.innovation_kind();
    let mut noise_values = unit_noise_draws(noise_kind, noise_seed_base, n_noise_draws, STEPS + 1, 1)?;
    impose_temporal_dependence(&mut noise_values, noise_process, n_noise_draws, STEPS + 1, 1)?;
    let noise = ArrayD::from_shape_vec(IxDyn(&[n_noise_draws, STEPS + 1, 1]), noise_values)?;
    let bank = DgpDrawBank::new(
        "controlled_pendulum",
        false,
        trajectory_seed_base,
        noise_seed_base,
        n_trajectory_draws,
        n_noise_draws,
        STEPS + 1,
        1,
    )?;
    let mut metadata = serie_bank_metadata(&bank, noise_process)?;
    metadata.extend([
        ("action_feature_names".to_string(), "[\"torque\"]".to_string()),
        (
            "action_alignment".to_string(),
            "actions[t]_drives_clean[t+1]".to_string(),
        ),
        ("controlled_pendulum_timestep".to_string(), "0.05".to_string()),
        ("controlled_pendulum_damping".to_string(), "0.15".to_string()),
        (
            "controlled_pendulum_gravity_over_length".to_string(),
            "9.81".to_string(),
        ),
        ("controlled_pendulum_torque_gain".to_string(), "0.8".to_string()),
        ("sinusoidal_torque_amplitude".to_string(), "0.6".to_string()),
        ("sinusoidal_torque_phase_increment".to_string(), "0.2".to_string()),
    ]);
    write_dgp_bank_safetensors(
        out_path,
        &clean,
        &noise,
        &[],
        &[("actions".to_string(), actions)],
        &["angle".to_string()],
        &metadata,
    )
}

/// The 33-sequence Level 0A impulse bank as a Serie draw bank.
///
/// Each draw is one frozen action sequence from the pinned `x0`. The stored
/// clean path is the analytic conditional mean, not one process-noise
/// realization: the wiring canary uses the exact expectation. Observation
/// noise is a zero tensor. The draw count is part of the fixture; a different
/// `n_trajectory_draws` is refused rather than silently truncating the bank.
///
/// The seed bases do not change one byte of this bank — every path is pinned —
/// but they are still recorded, because the worker checks that the bank it
/// loaded was built for the cell it is running
/// (`_assert_worker_bank_matches_request` in `fr_tr.py`). Writing zero here
/// while the cell asked for 42 refused every Level 0A launch.
fn generate_level0a_controlled_lgssm_dgp_bank(
    trajectory_seed_base: u64,
    noise_seed_base: u64,
    n_trajectory_draws: usize,
    n_noise_draws: usize,
    noise_process: ObservationNoiseProcess,
    out_path: impl AsRef<Path>,
) -> Fallible<()> {
    let sequences = Level0ACanary::action_bank()?.sequences().to_vec();
    if n_trajectory_draws != sequences.len() {
        return Err(format!(
            "level0a_controlled_lgssm is the pinned {}-sequence bank, not a prefix sampler; n_trajectory_draws must be {}, got {n_trajectory_draws}",
            sequences.len(),
            sequences.len()
        )
        .into());
    }
    let world = Level0ACanary::world()?;
    let initial_state = Level0ACanary::initial_state();
    let horizon = Level0ACanary::HORIZON;
    let timesteps = horizon + 1;
    let channels = 2;
    let mut clean_values = Vec::with_capacity(n_trajectory_draws * timesteps * channels);
    let mut action_values = Vec::with_capacity(n_trajectory_draws * horizon * channels);
    for sequence in &sequences {
        let oracle = world.direct_horizon(&initial_state, sequence)?;
        clean_values.extend(initial_state.iter().map(|value| *value as f32));
        for mean in &oracle.means {
            clean_values.extend(mean.iter().map(|value| *value as f32));
        }
        for timestep in 0..horizon {
            for action_coordinate in 0..2 {
                action_values.push(sequence[(timestep, action_coordinate)] as f32);
            }
        }
    }
    let clean = ArrayD::from_shape_vec(IxDyn(&[n_trajectory_draws, timesteps, channels]), clean_values)?;
    let actions = ArrayD::from_shape_vec(IxDyn(&[n_trajectory_draws, horizon, 2]), action_values)?;
    let noise = ArrayD::from_elem(IxDyn(&[n_noise_draws, timesteps, channels]), 0.0f32);
    let _ = noise_process;
    let bank = DgpDrawBank::new(
        "level0a_controlled_lgssm",
        false,
        trajectory_seed_base,
        noise_seed_base,
        n_trajectory_draws,
        n_noise_draws,
        timesteps,
        channels,
    )?;
    let mut metadata = serie_bank_metadata(&bank, noise_process)?;
    metadata.extend([
        ("action_feature_names".to_string(), "[\"u0\",\"u1\"]".to_string()),
        (
            "action_alignment".to_string(),
            "actions[t]_drives_clean[t+1]".to_string(),
        ),
        (
            "physics_context_id".to_string(),
            Level0ACanary::PHYSICS_CONTEXT_ID.to_string(),
        ),
        ("action_conditioning".to_string(), "future_only".to_string()),
        ("episode_axis".to_string(), "trajectory_draw".to_string()),
        ("clean_path".to_string(), "analytic_conditional_mean".to_string()),
        ("observation_noise".to_string(), "none_zero_tensor".to_string()),
    ]);
    write_dgp_bank_safetensors(
        out_path,
        &clean,
        &noise,
        &[],
        &[("actions".to_string(), actions)],
        &["x0".to_string(), "x1".to_string()],
        &metadata,
    )
}

/// The full serie metadata header (decision 2026-07-01-03): everything
/// Python's DGPBank spawn-consistency validation reads on load, stored the
/// way `serie.api.export_dgp_bank_safetensors` stores it — strings raw,
/// non-strings JSON-encoded. draw_design follows the stochastic rule both
/// engines share: stochastic sources pair trajectory/noise draws ("joint"),
/// deterministic and known-data sources vary noise only ("noise_only").
fn serie_bank_metadata(bank: &DgpDrawBank, noise_process: ObservationNoiseProcess) -> Fallible<Vec<(String, String)>> {
    let trajectory_draw_seeds = spawn_draw_seeds(bank.trajectory_seed_base, bank.n_trajectory_draws)?;
    let noise_draw_seeds = spawn_draw_seeds(bank.noise_seed_base, bank.n_noise_draws)?;
    let mut metadata = vec![
        ("source_name".to_string(), bank.source_name.clone()),
        (
            "noise_kind".to_string(),
            noise_process.innovation_kind().value().to_string(),
        ),
        ("is_stochastic".to_string(), bank.is_stochastic.to_string()),
        (
            "draw_design".to_string(),
            if bank.is_stochastic { "joint" } else { "noise_only" }.to_string(),
        ),
        ("data_engine".to_string(), "serie_rs".to_string()),
        (
            "axis_order".to_string(),
            "clean/noise=[draw,T,D]; noisy=[T,D]".to_string(),
        ),
        (
            "shape".to_string(),
            serde_json::json!({
                "clean": [bank.n_trajectory_draws, bank.rows, bank.dim],
                "noise": [bank.n_noise_draws, bank.rows, bank.dim],
            })
            .to_string(),
        ),
        (
            "trajectory_seed_base".to_string(),
            bank.trajectory_seed_base.to_string(),
        ),
        ("noise_seed_base".to_string(), bank.noise_seed_base.to_string()),
        (
            "trajectory_draw_seeds".to_string(),
            serde_json::to_string(&trajectory_draw_seeds)?,
        ),
        (
            "noise_draw_seeds".to_string(),
            serde_json::to_string(&noise_draw_seeds)?,
        ),
    ];
    if let Some(phi) = noise_process.ar1_phi() {
        metadata.extend([
            ("noise_temporal_process".to_string(), "ar1".to_string()),
            ("noise_ar1_phi".to_string(), phi.value().to_string()),
        ]);
    }
    Ok(metadata)
}

// ── known-data sources (serie-rs as the PREPARED-ASSET BANK GENERATOR) ──────

const KNOWN_DATA_SOURCE_NAMES: &[&str] = &[
    "etth1",
    "etth2",
    "ettm1",
    "ettm2",
    "solar_energy",
    "solar_energy_subset",
    "hugadb_accel_emg",
    "hugadb_walk_bicycling_switch",
    "hugadb_walk_run_natural",
    "hugadb_walk_only_natural",
];

fn is_known_source_name(source_name: &str) -> bool {
    KNOWN_DATA_SOURCE_NAMES.contains(&source_name)
}

/// Root for user-prepared known datasets. Cross-language mirror of Python
/// `serie.pipeline.known_data_root()`: the `SERIE_KNOWN_ROOT` env var wins
/// when set and non-empty (a leading `~` is expanded, like `Path.expanduser`),
/// else `$HOME/.cache/serie/known`. Keep both sides in sync — this generator
/// and any Python reader must resolve the SAME prepared assets.
#[cfg(feature = "known-bank")]
pub fn known_data_root() -> Fallible<PathBuf> {
    let home = || -> Fallible<PathBuf> {
        std::env::var_os("HOME")
            .map(PathBuf::from)
            .ok_or_else(|| "HOME is not set; cannot resolve the known-data root".into())
    };
    match std::env::var("SERIE_KNOWN_ROOT") {
        Ok(root) if !root.is_empty() => {
            if root == "~" {
                home()
            } else if let Some(rest) = root.strip_prefix("~/") {
                Ok(home()?.join(rest))
            } else {
                Ok(PathBuf::from(root))
            }
        }
        _ => Ok(home()?.join(".cache").join("serie").join("known")),
    }
}

/// One prepared known-data source serie-rs can read into a bank: the IPC
/// asset below [`known_data_root`], the datetime column dropped before
/// modeling, and — for sources whose Python load path actually cleans (the
/// ETT family) — the sentinel metadata driving column drops and short-run
/// imputation.
#[cfg(feature = "known-bank")]
struct KnownSourceAsset {
    /// Asset path relative to known_data_root(), e.g. "ett/ETTh1.ipc"
    /// (Python `KnownMeta.path`).
    asset_subpath: &'static str,
    /// Datetime column dropped right after the IPC read (Python
    /// `KnownMeta.date_column`); None when the asset has no such column.
    date_column: Option<&'static str>,
    /// Sentinel cleaning metadata (only `sentinels_for`/`steps_per_hour`/
    /// `impute_method` are used — the struct's own CSV path is ignored).
    /// None = the prepared IPC is final for this source: no sentinels are
    /// declared, so Python's clean stages are identity and the bank is a
    /// plain read + f32 cast (pinned bitwise by the known-data parity test).
    sentinel_meta: Option<KnownMeta>,
}

/// The prepared file a recorded source will be read from, for a caller that
/// wants to check it is there BEFORE starting work.
///
/// `None` means the name is not a recorded source at all (a simulated world
/// needs no prepared asset). `Some` resolves the same path the bank reader
/// will open, so a missing or misplaced asset can be reported by a preflight
/// instead of failing ten seconds into a sweep — which is how an unstaged
/// asset root has repeatedly killed lanes and idled paid GPUs.
#[cfg(feature = "known-bank")]
pub fn prepared_asset_path(source_name: &str) -> Option<Fallible<PathBuf>> {
    let asset = known_source_asset(source_name)?;
    Some(known_data_root().map(|root| root.join(asset.asset_subpath)))
}

/// The fern-run catalog's known-data sources, mirroring the
/// `serie.pipeline.KnownMeta` presets. Deliberately NOT wired: `weather` —
/// its only experiment (WEATHER_AIR) selects columns with `keep_col_idx` and
/// is not in the recipe catalog — and `hugadb_walk_bicycling_switch_ood80`,
/// which is not a catalog source either.
#[cfg(feature = "known-bank")]
fn known_source_asset(source_name: &str) -> Option<KnownSourceAsset> {
    let asset = match source_name {
        "etth1" => KnownSourceAsset {
            asset_subpath: "ett/ETTh1.ipc",
            date_column: Some("date"),
            sentinel_meta: Some(KnownMeta::etth1()),
        },
        "etth2" => KnownSourceAsset {
            asset_subpath: "ett/ETTh2.ipc",
            date_column: Some("date"),
            sentinel_meta: Some(KnownMeta::etth2()),
        },
        "ettm1" => KnownSourceAsset {
            asset_subpath: "ett/ETTm1.ipc",
            date_column: Some("date"),
            sentinel_meta: Some(KnownMeta::ettm1()),
        },
        "ettm2" => KnownSourceAsset {
            asset_subpath: "ett/ETTm2.ipc",
            date_column: Some("date"),
            sentinel_meta: Some(KnownMeta::ettm2()),
        },
        "solar_energy" => KnownSourceAsset {
            asset_subpath: "solar/solar_AL.ipc",
            date_column: None,
            sentinel_meta: None,
        },
        "solar_energy_subset" => KnownSourceAsset {
            asset_subpath: "solar/solar_AL_subset16.ipc",
            date_column: None,
            sentinel_meta: None,
        },
        "hugadb_accel_emg" => KnownSourceAsset {
            asset_subpath: "hugadb/hugadb_accel_emg.ipc",
            date_column: None,
            sentinel_meta: None,
        },
        // One continuous single-subject recording with a performed
        // walk -> run -> walk change at 36.4% of the record: the transitions
        // were done by the subject, not concatenated by us.
        "hugadb_walk_run_natural" => KnownSourceAsset {
            asset_subpath: "hugadb/hugadb_walk_run_natural.ipc",
            date_column: None,
            sentinel_meta: None,
        },
        // The stationary control for the world above: the SAME subject,
        // sensors, and session, walking throughout. Paired with it, the
        // activity change is the only designed difference between the two
        // recorded worlds.
        "hugadb_walk_only_natural" => KnownSourceAsset {
            asset_subpath: "hugadb/hugadb_walk_only_natural.ipc",
            date_column: None,
            sentinel_meta: None,
        },
        "hugadb_walk_bicycling_switch" => KnownSourceAsset {
            asset_subpath: "hugadb/hugadb_walk_bicycling_switch.ipc",
            date_column: None,
            sentinel_meta: None,
        },
        _ => return None,
    };
    Some(asset)
}

#[cfg(feature = "known-bank")]
struct KnownColumn {
    name: String,
    values: Vec<Option<f64>>,
}

#[cfg(feature = "known-bank")]
struct KnownMatrix {
    columns: Vec<KnownColumn>,
    timesteps: usize,
}

/// Read a prepared Arrow IPC file into numeric columns without a table engine.
#[cfg(feature = "known-bank")]
fn read_known_ipc(bytes: &[u8], date_column: Option<&str>) -> Fallible<KnownMatrix> {
    use arrow_array::{Array, Float32Array, Float64Array, Int64Array};
    use arrow_ipc::reader::FileReader;

    fn append_numeric(array: &dyn Array, values: &mut Vec<Option<f64>>) -> Fallible<()> {
        if let Some(array) = array.as_any().downcast_ref::<Float64Array>() {
            values.extend((0..array.len()).map(|row| (!array.is_null(row)).then(|| array.value(row))));
        } else if let Some(array) = array.as_any().downcast_ref::<Float32Array>() {
            values.extend((0..array.len()).map(|row| (!array.is_null(row)).then(|| f64::from(array.value(row)))));
        } else if let Some(array) = array.as_any().downcast_ref::<Int64Array>() {
            values
                .extend((0..array.len()).map(|row| (!array.is_null(row)).then(|| f64::from(array.value(row) as f32))));
        } else {
            return Err(format!(
                "prepared known-data IPC has unsupported numeric type {:?}",
                array.data_type()
            )
            .into());
        }
        Ok(())
    }

    let mut reader = FileReader::try_new(std::io::Cursor::new(bytes), None)?;
    let selected_fields: Vec<(usize, String)> = reader
        .schema()
        .fields()
        .iter()
        .enumerate()
        .filter(|(_, field)| date_column != Some(field.name().as_str()))
        .map(|(index, field)| (index, field.name().clone()))
        .collect();
    if selected_fields.is_empty() {
        return Err("prepared known-data IPC has no feature columns after dropping its date column".into());
    }
    let mut columns: Vec<KnownColumn> = selected_fields
        .iter()
        .map(|(_, name)| KnownColumn {
            name: name.clone(),
            values: Vec::new(),
        })
        .collect();
    let mut timesteps = 0usize;
    for batch in &mut reader {
        let batch = batch?;
        timesteps = timesteps
            .checked_add(batch.num_rows())
            .ok_or("prepared known-data IPC row count overflow")?;
        for (column, (source_index, _)) in columns.iter_mut().zip(&selected_fields) {
            append_numeric(batch.column(*source_index).as_ref(), &mut column.values)?;
        }
    }
    if timesteps == 0 {
        return Err("prepared known-data IPC contains no rows".into());
    }
    if columns.iter().any(|column| column.values.len() != timesteps) {
        return Err("prepared known-data IPC columns have inconsistent row counts".into());
    }
    Ok(KnownMatrix { columns, timesteps })
}

/// Sentinel-clean one known-data matrix the way the Python worker's load did
/// (`serie.pipeline` StageDropBadColumns then StageImputeShortRuns, under the
/// KnownLoadMode.CLEAN defaults):
///
/// 1. Per column, on the RAW frame: sentinel fraction and longest sentinel
///    run. Drop columns with
///    `frac >= drop_col_threshold` or a run of at least
///    `drop_col_hour_threshold` hours. Column ORDER is preserved; only
///    membership changes.
/// 2. On the surviving columns: mask every sentinel run to null, then
///    interpolate (`KnownMeta.impute_method`) and fill the edges
///    forward/backward. Sentinel-less columns pass through unchanged.
///
/// Order matters: Python evaluates the drop statistics BEFORE any masking.
/// The array reader and the full NumericFrame pipeline share the same KnownMeta
/// and named defaults, so there is one source for sentinel policy and knobs.
#[cfg(feature = "known-bank")]
fn clean_known_matrix(matrix: KnownMatrix, meta: &KnownMeta) -> Fallible<KnownMatrix> {
    fn longest_true_run(flags: &[bool]) -> usize {
        let mut longest = 0usize;
        let mut current = 0usize;
        for flag in flags {
            if *flag {
                current += 1;
                longest = longest.max(current);
            } else {
                current = 0;
            }
        }
        longest
    }

    fn impute(values: &mut [Option<f64>], method: ImputeMethod, column_name: &str) -> Fallible<()> {
        let present: Vec<usize> = values
            .iter()
            .enumerate()
            .filter_map(|(index, value)| value.map(|_| index))
            .collect();
        let Some(&first) = present.first() else {
            return Err(format!("known-data column {column_name:?} has no non-null value to impute from").into());
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
                let value = match method {
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
                };
                values[index] = Some(value);
            }
        }
        let last_value = values[last].expect("present index has a value");
        values[(last + 1)..].fill(Some(last_value));
        Ok(())
    }

    let n_rows = matrix.timesteps;
    let threshold_steps = (DEFAULT_DROP_COL_HOUR_THRESHOLD * meta.steps_per_hour()) as usize;
    let mut surviving = Vec::with_capacity(matrix.columns.len());
    for mut column in matrix.columns {
        let sentinels = meta.sentinels_for(&column.name);
        let sentinel_flags: Vec<bool> = column
            .values
            .iter()
            .map(|value| {
                value.is_some_and(|value| {
                    sentinels
                        .iter()
                        .any(|sentinel| (value - sentinel).abs() <= DEFAULT_SENTINEL_VALUE_TOL)
                })
            })
            .collect();
        let sentinel_fraction = sentinel_flags.iter().filter(|flag| **flag).count() as f64 / n_rows as f64;
        if sentinel_fraction >= DEFAULT_DROP_COL_THRESHOLD || longest_true_run(&sentinel_flags) >= threshold_steps {
            continue;
        }
        for (value, is_sentinel) in column.values.iter_mut().zip(sentinel_flags) {
            if is_sentinel {
                *value = None;
            }
        }
        impute(&mut column.values, meta.impute_method(), &column.name)?;
        surviving.push(column);
    }
    if surviving.is_empty() {
        return Err("known-data sentinel cleaning removed every feature column".into());
    }
    Ok(KnownMatrix {
        columns: surviving,
        timesteps: n_rows,
    })
}

/// Mirror of Python `serie.pipeline.StageConvertToNumpy`: cast every column
/// to Float32 FIRST, then read the values row-major `[T*D]` (C order), column
/// order following the prepared IPC schema.
#[cfg(feature = "known-bank")]
fn known_matrix_to_rowmajor_f32(matrix: KnownMatrix) -> Fallible<(Vec<f32>, usize, Vec<String>)> {
    let timesteps = matrix.timesteps;
    let feature_names: Vec<String> = matrix.columns.iter().map(|column| column.name.clone()).collect();
    let channels = feature_names.len();
    let mut flat = Vec::with_capacity(timesteps * channels);
    for ti in 0..timesteps {
        for column in &matrix.columns {
            let value = column.values[ti]
                .ok_or_else(|| format!("known-data column {:?} still has a null after imputation", column.name))?;
            flat.push(value as f32);
        }
    }
    Ok((flat, timesteps, feature_names))
}

/// Generate one grouped bank for a prepared known-data source: read the IPC
/// asset under [`known_data_root`], drop the datetime column, apply the same
/// sentinel cleaning Python's KnownLoadMode.CLEAN default applied
/// ([`clean_known_matrix`], ETT family only), and store the result as
/// `clean [1, T, D]` — a fixed recording has exactly one trajectory draw with
/// `trajectory_seed_base = 0` (the KnownMeta convention the fern-run planner
/// already enforces via SourceDrawKind). Noise draws are generated exactly
/// like synthetic banks.
///
/// Unlike synthetic sources there is NO RNG in the clean channel: the stored
/// clean values are bitwise-identical f32 to what Python
/// `load_data(KnownMeta)` produced (pinned by the known-data parity test).
/// Only the frozen noise stream differs across engines.
///
/// The bank pins its input: metadata gains `source_asset` (asset path
/// relative to the known root) and `source_asset_sha` (sha256[:16] of the
/// IPC bytes), so a silently swapped asset is visible per bank — and per run,
/// once the parent copies it into a `data_assets` run_event.
#[cfg(feature = "known-bank")]
fn generate_known_dgp_bank(
    source_name: &str,
    asset: &KnownSourceAsset,
    trajectory_seed_base: u64,
    noise_seed_base: u64,
    n_trajectory_draws: usize,
    n_noise_draws: usize,
    noise_process: ObservationNoiseProcess,
    out_path: impl AsRef<Path>,
) -> Fallible<()> {
    if n_trajectory_draws != 1 {
        return Err(format!(
            "known data supports exactly n_trajectory_draws=1, got {n_trajectory_draws}; \
             known datasets are fixed recordings, not generative"
        )
        .into());
    }
    if trajectory_seed_base != 0 {
        return Err(format!(
            "known-data DGP banks use trajectory_seed_base=0; got trajectory_seed_base={trajectory_seed_base}"
        )
        .into());
    }

    let asset_path = known_data_root()?.join(asset.asset_subpath);
    let bytes = std::fs::read(&asset_path).map_err(|read_error| {
        format!(
            "known dataset {source_name:?}: cannot read prepared IPC {}: {read_error}. \
             Set SERIE_KNOWN_ROOT to a directory containing prepared known datasets.",
            asset_path.display()
        )
    })?;
    // sha256[:16] of the exact bytes parsed below — the same truncation the
    // Python data_asset_fingerprint uses.
    let source_asset_sha = {
        use sha2::{Digest, Sha256};
        format!("{:x}", Sha256::digest(&bytes))[..16].to_string()
    };

    // Python reference: drop the named date column before numeric cleaning.
    let mut matrix = read_known_ipc(&bytes, asset.date_column)?;
    if let Some(meta) = &asset.sentinel_meta {
        matrix = clean_known_matrix(matrix, meta)?;
    }
    let (flat, timesteps, feature_names) = known_matrix_to_rowmajor_f32(matrix)?;
    let channels = feature_names.len();
    let clean = ArrayD::from_shape_vec(IxDyn(&[1, timesteps, channels]), flat)?;

    let noise_kind = noise_process.innovation_kind();
    let mut noise_flat = unit_noise_draws(noise_kind, noise_seed_base, n_noise_draws, timesteps, channels)?;
    impose_temporal_dependence(&mut noise_flat, noise_process, n_noise_draws, timesteps, channels)?;
    let noise = ArrayD::from_shape_vec(IxDyn(&[n_noise_draws, timesteps, channels]), noise_flat)?;

    let bank = DgpDrawBank::new(
        source_name,
        false,
        trajectory_seed_base,
        noise_seed_base,
        1,
        n_noise_draws,
        timesteps,
        channels,
    )?;
    let mut extra_metadata = serie_bank_metadata(&bank, noise_process)?;
    extra_metadata.push(("source_asset".to_string(), asset.asset_subpath.to_string()));
    extra_metadata.push(("source_asset_sha".to_string(), source_asset_sha));
    write_dgp_bank_safetensors(out_path, &clean, &noise, &[], &[], &feature_names, &extra_metadata)
}

#[cfg(test)]
mod ar1_tests {
    use super::*;

    fn draws(phi: f64) -> Vec<f32> {
        let (n, t, d) = (1usize, 60_000usize, 1usize);
        let process = ObservationNoiseProcess::from_wire(NoiseKind::Gaussian, phi).unwrap();
        let mut flat = unit_noise_draws(process.innovation_kind(), 4242, n, t, d).unwrap();
        impose_temporal_dependence(&mut flat, process, n, t, d).unwrap();
        flat
    }

    /// The whole point of the sqrt(1 - phi^2) factor: a dose must mean the
    /// same multiple of the training standard deviation at every phi, or the
    /// correlation axis silently changes the dose axis too.
    #[test]
    fn ar1_preserves_unit_variance_at_every_phi() {
        for phi in [0.0, 0.5, 0.9, -0.5] {
            let values = draws(phi);
            let mean = values.iter().map(|x| *x as f64).sum::<f64>() / values.len() as f64;
            let variance = values.iter().map(|x| (*x as f64 - mean).powi(2)).sum::<f64>() / values.len() as f64;
            // Correlated samples carry less information than their count
            // suggests: an AR(1) series of length n is worth about
            // n(1-phi)/(1+phi) independent draws, so at phi=0.9 sixty thousand
            // timesteps behave like three thousand. A flat tolerance would
            // pass at phi=0 and fail at phi=0.9 for no reason but that. Allow
            // four standard errors of the variance estimate at the EFFECTIVE
            // sample size.
            let effective = values.len() as f64 * (1.0 - phi) / (1.0 + phi);
            let tolerance = 4.0 * (2.0 / effective).sqrt();
            assert!(
                (variance - 1.0).abs() < tolerance,
                "phi={phi} gave variance {variance}, outside {tolerance:.4} of 1"
            );
        }
    }

    /// Lag-one autocorrelation must come out as the phi that was asked for.
    #[test]
    fn ar1_lag_one_autocorrelation_equals_phi() {
        for phi in [0.5, 0.9, -0.5] {
            let values = draws(phi);
            let mean = values.iter().map(|x| *x as f64).sum::<f64>() / values.len() as f64;
            let centred: Vec<f64> = values.iter().map(|x| *x as f64 - mean).collect();
            let covariance: f64 = centred.windows(2).map(|w| w[0] * w[1]).sum::<f64>() / (centred.len() - 1) as f64;
            let variance: f64 = centred.iter().map(|x| x * x).sum::<f64>() / centred.len() as f64;
            let measured = covariance / variance;
            assert!(
                (measured - phi).abs() < 0.02,
                "phi={phi} measured lag-1 autocorrelation {measured}"
            );
        }
    }

    /// phi = 0 must be the EXISTING gaussian lane, bit for bit, so that rung
    /// is a consistency check rather than a rerun.
    #[test]
    fn ar1_phi_zero_is_bitwise_identical_to_the_plain_gaussian_draw() {
        let (n, t, d) = (2usize, 128usize, 3usize);
        let plain = unit_noise_draws(NoiseKind::Gaussian, 99, n, t, d).unwrap();
        let mut filtered = plain.clone();
        let iid = ObservationNoiseProcess::from_wire(NoiseKind::Gaussian, 0.0).unwrap();
        impose_temporal_dependence(&mut filtered, iid, n, t, d).unwrap();
        assert_eq!(plain, filtered);
    }

    /// Raw config can still be invalid, but the only boundary constructor
    /// rejects it; no AR(1) function accepts a NoiseKind or naked f64.
    #[test]
    fn ar1_process_constructor_refuses_invalid_wire_combinations() {
        assert!(ObservationNoiseProcess::from_wire(NoiseKind::StudentT3, 0.5).is_err());
        assert!(ObservationNoiseProcess::from_wire(NoiseKind::Laplace, -0.5).is_err());
        assert!(ObservationNoiseProcess::from_wire(NoiseKind::Gaussian, 1.0).is_err());
        assert!(ObservationNoiseProcess::from_wire(NoiseKind::Gaussian, f64::NAN).is_err());
        assert_eq!(
            ObservationNoiseProcess::from_wire(NoiseKind::StudentT3, 0.0).unwrap(),
            ObservationNoiseProcess::Iid(NoiseKind::StudentT3)
        );
    }

    #[test]
    fn ar1_metadata_is_explicit_and_iid_metadata_stays_legacy_compatible() {
        let bank = DgpDrawBank::new("ou", true, 42, 1000, 1, 1, 16, 2).unwrap();
        let iid = serie_bank_metadata(&bank, ObservationNoiseProcess::Iid(NoiseKind::Gaussian)).unwrap();
        assert!(!iid.iter().any(|(key, _)| key == "noise_ar1_phi"));
        assert!(!iid.iter().any(|(key, _)| key == "noise_temporal_process"));

        let ar1_process = ObservationNoiseProcess::from_wire(NoiseKind::Gaussian, 0.9).unwrap();
        let ar1 = serie_bank_metadata(&bank, ar1_process).unwrap();
        assert!(ar1.contains(&("noise_temporal_process".to_string(), "ar1".to_string())));
        assert!(ar1.contains(&("noise_ar1_phi".to_string(), "0.9".to_string())));
    }

    /// Each channel is its own series; correlation must not leak across the
    /// row-major stride.
    #[test]
    fn ar1_filters_each_channel_independently() {
        let (n, t, d) = (1usize, 4usize, 2usize);
        // [timestep][channel]. Channel 1 is identically zero, so any nonzero
        // output there proves channel 0 leaked across the stride.
        let mut flat = vec![1.0, 0.0, 2.0, 0.0, 3.0, 0.0, 4.0, 0.0];
        let process = ObservationNoiseProcess::from_wire(NoiseKind::Gaussian, 0.5).unwrap();
        impose_temporal_dependence(&mut flat, process, n, t, d).unwrap();
        assert_eq!(
            flat.iter().skip(1).step_by(2).copied().collect::<Vec<_>>(),
            vec![0.0; t]
        );
        assert_ne!(
            flat.iter().step_by(2).copied().collect::<Vec<_>>(),
            vec![1.0, 2.0, 3.0, 4.0]
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn iid(noise_kind: NoiseKind) -> ObservationNoiseProcess {
        ObservationNoiseProcess::Iid(noise_kind)
    }

    fn le_bytes(values: &[f32]) -> Vec<u8> {
        values.iter().flat_map(|x| x.to_le_bytes()).collect()
    }

    #[test]
    fn source_handler_query_reads_the_real_materializer_tables() {
        assert!(source_handler_is_registered("lorenz"));
        assert!(source_handler_is_registered("level0a_controlled_lgssm"));
        assert!(source_handler_is_registered(
            ActionIndependentGammaAttenuationCanary::SOURCE_NAME
        ));
        assert!(source_handler_is_registered(
            ActionIndependentGaussianAttenuationCanary::SOURCE_NAME
        ));
        assert!(source_handler_is_registered("rossler_ic_holdout_7"));
        assert!(source_handler_is_registered("etth1"));
        assert!(!source_handler_is_registered("ks"));
        assert!(!source_handler_is_registered("world_without_an_implementation"));
    }

    #[test]
    fn radial_attenuation_bank_is_action_free_and_preserves_the_pinned_covariance() {
        let dir = temp_bank_dir("generate_radial_attenuation_gamma4");
        let path = dir.join("dgp_bank.safetensors");
        generate_dgp_bank(
            ActionIndependentGammaAttenuationCanary::SOURCE_NAME,
            42,
            1000,
            1,
            1,
            iid(NoiseKind::Gaussian),
            &path,
        )
        .unwrap();

        let bank = load_dgp_bank_safetensors(&path).unwrap();
        assert_eq!(bank.clean().shape(), &[1, 25_000, 2]);
        assert_eq!(bank.feature_names, vec!["x0", "x1"]);
        assert!(!bank.tensors.contains_key("actions"));
        assert_eq!(
            bank.metadata.get("action_conditioning").map(String::as_str),
            Some("none")
        );
        assert_eq!(
            bank.metadata.get("process_covariance").map(String::as_str),
            Some("0.0225*I2")
        );

        let clean = bank.clean().as_slice().unwrap();
        let mut sum_squares = 0.0_f64;
        let mut sum_fourths = 0.0_f64;
        let mut count = 0usize;
        for timestep in 1..25_000 {
            let previous = &clean[(timestep - 1) * 2..timestep * 2];
            let current = &clean[timestep * 2..(timestep + 1) * 2];
            let standardized_radius_squared = (0..2)
                .map(|coordinate| {
                    let residual = f64::from(current[coordinate])
                        - ActionIndependentGammaAttenuationCanary::DRIFT * f64::from(previous[coordinate]);
                    let standardized = residual / ActionIndependentGammaAttenuationCanary::SCALE;
                    standardized * standardized
                })
                .sum::<f64>();
            sum_squares += standardized_radius_squared;
            sum_fourths += standardized_radius_squared * standardized_radius_squared;
            count += 1;
        }
        let second_moment = sum_squares / count as f64;
        let fourth_moment = sum_fourths / count as f64;
        assert!((second_moment - 2.0).abs() < 0.06, "empirical E[R^2]={second_moment}");
        assert!((fourth_moment - 10.0).abs() < 0.7, "empirical E[R^4]={fourth_moment}");
    }

    #[test]
    fn gaussian_attenuation_control_uses_the_same_process_covariance_without_gamma_tails() {
        let dir = temp_bank_dir("generate_radial_attenuation_gaussian");
        let path = dir.join("dgp_bank.safetensors");
        generate_dgp_bank(
            ActionIndependentGaussianAttenuationCanary::SOURCE_NAME,
            42,
            1000,
            1,
            1,
            iid(NoiseKind::Gaussian),
            &path,
        )
        .unwrap();

        let bank = load_dgp_bank_safetensors(&path).unwrap();
        assert_eq!(bank.clean().shape(), &[1, 25_000, 2]);
        assert!(!bank.tensors.contains_key("actions"));
        assert_eq!(
            bank.metadata.get("process_covariance").map(String::as_str),
            Some("0.0225*I2")
        );
        assert_eq!(
            bank.metadata.get("radial_scale_law").map(String::as_str),
            Some("constant_one")
        );

        let clean = bank.clean().as_slice().unwrap();
        let mut sum_squares = 0.0_f64;
        let mut sum_fourths = 0.0_f64;
        let mut count = 0usize;
        for timestep in 1..25_000 {
            let previous = &clean[(timestep - 1) * 2..timestep * 2];
            let current = &clean[timestep * 2..(timestep + 1) * 2];
            let standardized_radius_squared = (0..2)
                .map(|coordinate| {
                    let residual = f64::from(current[coordinate])
                        - ActionIndependentGaussianAttenuationCanary::DRIFT * f64::from(previous[coordinate]);
                    let standardized = residual / ActionIndependentGaussianAttenuationCanary::SCALE;
                    standardized * standardized
                })
                .sum::<f64>();
            sum_squares += standardized_radius_squared;
            sum_fourths += standardized_radius_squared * standardized_radius_squared;
            count += 1;
        }
        let second_moment = sum_squares / count as f64;
        let fourth_moment = sum_fourths / count as f64;
        assert!((second_moment - 2.0).abs() < 0.06, "empirical E[R^2]={second_moment}");
        assert!((fourth_moment - 8.0).abs() < 0.5, "empirical E[R^4]={fourth_moment}");
    }

    #[test]
    fn rossler_holdout_is_the_complete_balanced_pm_0p01_cube() {
        let initial_conditions = (0..8)
            .map(|ordinal| rossler_initial_condition_holdout(&format!("rossler_ic_holdout_{ordinal}")).unwrap())
            .collect::<Vec<_>>();
        assert_eq!(initial_conditions[0], [0.99, 0.99, 0.99]);
        assert_eq!(initial_conditions[7], [1.01, 1.01, 1.01]);
        for channel in 0..3 {
            assert_eq!(
                initial_conditions
                    .iter()
                    .filter(|initial_condition| initial_condition[channel] == 0.99)
                    .count(),
                4
            );
            assert_eq!(
                initial_conditions
                    .iter()
                    .filter(|initial_condition| initial_condition[channel] == 1.01)
                    .count(),
                4
            );
        }
        assert!(rossler_initial_condition_holdout("rossler_ic_holdout_8").is_none());
        assert!(rossler_initial_condition_holdout("rossler").is_none());
    }

    /// Pin every drawn coordinate. The set must be fixed before any of these
    /// trajectories is scored, or "sixteen random initial conditions" becomes
    /// sixteen chances to find one that preserves the discovery-grid reversal.
    /// Changing the seed here changes the experiment, and this test is what
    /// makes that impossible to do quietly.
    #[test]
    fn rossler_random_initial_conditions_are_predeclared() {
        let drawn = (0..16)
            .map(|ordinal| rossler_initial_condition_random(&format!("rossler_ic_random_{ordinal}")).unwrap())
            .collect::<Vec<_>>();
        // Rounded to 1e-9; the generator is exact but the pin should read.
        let round = |x: f64| (x * 1e9).round() / 1e9;
        let pinned: Vec<[f64; 3]> = drawn.iter().map(|c| [round(c[0]), round(c[1]), round(c[2])]).collect();
        assert_eq!(pinned.len(), 16);
        // Every draw lies strictly inside the cube the eight corners bound,
        // and no two units share an initial condition.
        for coordinate in &drawn {
            for axis in 0..3 {
                assert!(
                    (coordinate[axis] - 1.0).abs() <= ROSSLER_IC_HOLDOUT_DELTA,
                    "draw {coordinate:?} left the +-0.01 cube"
                );
            }
        }
        for i in 0..drawn.len() {
            for j in (i + 1)..drawn.len() {
                assert_ne!(drawn[i], drawn[j], "units {i} and {j} coincide");
            }
        }
        assert_eq!(pinned[0], [0.991358929, 0.996469352, 0.996971415]);
        assert_eq!(pinned[15], [1.003722505, 0.992132128, 1.004818773]);
        assert!(rossler_initial_condition_random("rossler_ic_random_16").is_none());
        assert!(rossler_initial_condition_random("rossler_ic_holdout_0").is_none());
    }

    #[test]
    fn round_trips_clean_noise_and_feature_names() {
        // clean [1,2,2], noise [2,2,2].
        let clean = le_bytes(&[1.0, 2.0, 3.0, 4.0]);
        let noise = le_bytes(&[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]);
        let data: BTreeMap<String, TensorView> = BTreeMap::from([
            (
                "clean".to_string(),
                TensorView::new(Dtype::F32, vec![1, 2, 2], &clean).unwrap(),
            ),
            (
                "noise".to_string(),
                TensorView::new(Dtype::F32, vec![2, 2, 2], &noise).unwrap(),
            ),
        ]);
        let mut meta = HashMap::new();
        meta.insert("feature_names".to_string(), "[\"a\", \"b\"]".to_string());

        let bytes = safetensors::serialize(&data, Some(meta)).unwrap();
        let bank = from_bytes(&bytes, PathBuf::from("dgp_bank.safetensors")).unwrap();

        assert_eq!(bank.feature_names, vec!["a", "b"]);
        assert_eq!(bank.clean().shape(), &[1, 2, 2]);
        assert_eq!(bank.noise().shape(), &[2, 2, 2]);
        assert_eq!(bank.clean()[[0, 0, 0]], 1.0);
        assert!(bank.noisy_tensor_names().is_empty());
    }

    #[test]
    fn defaults_feature_names_when_absent() {
        let clean = le_bytes(&[1.0, 2.0, 3.0]);
        let noise = le_bytes(&[0.0, 0.0, 0.0]);
        let data: BTreeMap<String, TensorView> = BTreeMap::from([
            (
                "clean".to_string(),
                TensorView::new(Dtype::F32, vec![1, 1, 3], &clean).unwrap(),
            ),
            (
                "noise".to_string(),
                TensorView::new(Dtype::F32, vec![1, 1, 3], &noise).unwrap(),
            ),
        ]);
        let bytes = safetensors::serialize(&data, None).unwrap();
        let bank = from_bytes(&bytes, PathBuf::from("x")).unwrap();
        assert_eq!(bank.feature_names, vec!["f0", "f1", "f2"]);
    }

    #[test]
    fn writer_round_trips_through_reader() {
        let clean = ArrayD::from_shape_vec(IxDyn(&[1, 2, 2]), vec![1.0f32, 2.0, 3.0, 4.0]).unwrap();
        let noise = ArrayD::from_shape_vec(IxDyn(&[2, 2, 2]), vec![0.1f32, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]).unwrap();
        let noisy = vec![(
            "noisy_sigma_0.5".to_string(),
            ArrayD::from_shape_vec(IxDyn(&[2, 2]), vec![9.0f32, 8.0, 7.0, 6.0]).unwrap(),
        )];
        let names = vec!["a".to_string(), "b".to_string()];

        let bytes = dgp_bank_to_bytes(
            &clean,
            &noise,
            &noisy,
            &[],
            &names,
            &[("seed".to_string(), "42".to_string())],
        )
        .unwrap();
        let bank = from_bytes(&bytes, PathBuf::from("dgp_bank.safetensors")).unwrap();

        assert_eq!(bank.feature_names, vec!["a", "b"]);
        assert_eq!(bank.clean(), &clean);
        assert_eq!(bank.noise(), &noise);
        assert_eq!(bank.noisy_tensor_names(), vec!["noisy_sigma_0.5"]);
        assert_eq!(
            bank.metadata.get("schema").map(String::as_str),
            Some("serie.dgp_bank_safetensors.v1")
        );
        assert_eq!(bank.metadata.get("seed").map(String::as_str), Some("42"));
    }

    #[test]
    fn controlled_pendulum_bank_keeps_actions_separate_and_transition_aligned() {
        let dir = temp_bank_dir("controlled_pendulum_actions");
        let path = dir.join("bank.safetensors");
        generate_dgp_bank(
            "controlled_pendulum",
            42,
            1000,
            1,
            1,
            ObservationNoiseProcess::Iid(NoiseKind::Gaussian),
            &path,
        )
        .unwrap();
        let bank = load_dgp_bank_safetensors(&path).unwrap();
        let actions = &bank.tensors["actions"];
        assert_eq!(bank.clean().shape(), &[1, 4097, 1]);
        assert_eq!(actions.shape(), &[1, 4096, 1]);
        assert_eq!(
            bank.metadata.get("action_alignment").map(String::as_str),
            Some("actions[t]_drives_clean[t+1]")
        );
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn level0a_bank_is_the_pinned_33_sequence_mean_paths() {
        let dir = temp_bank_dir("level0a_actions");
        let path = dir.join("bank.safetensors");
        generate_dgp_bank(
            "level0a_controlled_lgssm",
            42,
            7,
            33,
            1,
            ObservationNoiseProcess::Iid(NoiseKind::Gaussian),
            &path,
        )
        .unwrap();
        let bank = load_dgp_bank_safetensors(&path).unwrap();
        let actions = &bank.tensors["actions"];
        assert_eq!(bank.clean().shape(), &[33, 9, 2]);
        assert_eq!(actions.shape(), &[33, 8, 2]);
        assert_eq!(
            bank.metadata.get("action_conditioning").map(String::as_str),
            Some("future_only")
        );
        assert_eq!(
            bank.metadata.get("episode_axis").map(String::as_str),
            Some("trajectory_draw")
        );
        let first_sequence = actions.index_axis(ndarray::Axis(0), 0);
        assert!(
            first_sequence.iter().all(|value| *value == 0.0),
            "the first Level 0A action sequence must be all-zero"
        );
        assert_eq!(bank.clean()[[0, 0, 0]], 0.4);
        assert_eq!(bank.clean()[[0, 0, 1]], -0.3);
        // The pinned paths do not depend on the seed bases, but the worker
        // checks that the bank it loaded was built for the cell it is running.
        // Recording zero here refused every Level 0A launch.
        assert_eq!(
            bank.metadata.get("trajectory_seed_base").map(String::as_str),
            Some("42")
        );
        assert_eq!(bank.metadata.get("noise_seed_base").map(String::as_str), Some("7"));
        std::fs::remove_dir_all(dir).unwrap();
    }

    fn temp_bank_dir(name: &str) -> PathBuf {
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let dir = std::env::temp_dir().join(format!("serie_rs_{name}_{}_{}", std::process::id(), nanos));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn student_t3_unit_noise_draws_are_variance_matched_and_heavy_tailed() {
        let flat = unit_noise_draws(NoiseKind::StudentT3, 1000, 1, 200, 50).unwrap();
        let n = flat.len() as f64;
        let mean = flat.iter().map(|v| f64::from(*v)).sum::<f64>() / n;
        let variance = flat.iter().map(|v| (f64::from(*v) - mean).powi(2)).sum::<f64>() / n;
        // sqrt(1/3)-scaled t3 has variance one; the fixed seed makes this one
        // stream deterministic, and t3's infinite fourth moment makes the
        // variance estimator heavy-tailed, hence the loose tolerance.
        assert!(mean.abs() < 0.05, "empirical mean {mean}");
        assert!((variance - 1.0).abs() < 0.3, "empirical variance {variance}");
        // Heavier tails than Gaussian: |draw| > 4 means |t3| > 4*sqrt(3),
        // probability ~6e-3 two-sided (~60 in 10_000) vs ~6e-5 for N(0,1)
        // (~0.6 in 10_000).
        let tail_count = flat.iter().filter(|v| v.abs() > 4.0).count();
        assert!(
            tail_count >= 5,
            "expected heavy tails, got {tail_count} draws with |x|>4"
        );
    }

    /// Extending a bank must append draws, never perturb banks already used by
    /// the four-bank campaign. This is the exact reproducibility contract
    /// behind the eight-bank lane: banks 0..3 may be recomputed under a new
    /// recipe identity, but their frozen noise tensors stay bitwise equal.
    #[test]
    fn extending_noise_bank_preserves_the_existing_draw_prefix_bitwise() {
        let (timesteps, channels) = (37usize, 3usize);
        for noise_kind in [
            NoiseKind::Gaussian,
            NoiseKind::Laplace,
            NoiseKind::StudentT3,
            NoiseKind::Contaminated,
        ] {
            let four = unit_noise_draws(noise_kind, 1000, 4, timesteps, channels).unwrap();
            let eight = unit_noise_draws(noise_kind, 1000, 8, timesteps, channels).unwrap();
            assert_eq!(
                four,
                eight[..four.len()],
                "{noise_kind:?} changed banks 0..3 when the bank was extended"
            );
        }
    }

    #[test]
    fn laplace_unit_noise_draws_are_variance_matched_with_middle_tails() {
        let flat = unit_noise_draws(NoiseKind::Laplace, 1000, 1, 200, 50).unwrap();
        let n = flat.len() as f64;
        let mean = flat.iter().map(|v| f64::from(*v)).sum::<f64>() / n;
        let variance = flat.iter().map(|v| (f64::from(*v) - mean).powi(2)).sum::<f64>() / n;
        assert!(mean.abs() < 0.03, "empirical mean {mean}");
        assert!((variance - 1.0).abs() < 0.08, "empirical variance {variance}");
        // Middle rung of the tail ladder: P(|X| > 4) = exp(-4*sqrt(2)) ~ 3.5e-3
        // (~35 in 10_000) — an order above Gaussian (~0.6), an order below the
        // scaled t3 (~60).
        let tail_count = flat.iter().filter(|v| v.abs() > 4.0).count();
        assert!(
            (5..=120).contains(&tail_count),
            "expected Laplace-weight tails, got {tail_count} draws with |x|>4"
        );
    }

    #[test]
    fn contaminated_unit_noise_draws_are_variance_matched_and_sparse_tailed() {
        let flat = unit_noise_draws(NoiseKind::Contaminated, 1000, 1, 2_000, 100).unwrap();
        let n = flat.len() as f64;
        let mean = flat.iter().map(|value| f64::from(*value)).sum::<f64>() / n;
        let variance = flat.iter().map(|value| (f64::from(*value) - mean).powi(2)).sum::<f64>() / n;
        assert!(mean.abs() < 0.03, "empirical mean {mean}");
        assert!((variance - 1.0).abs() < 0.08, "empirical variance {variance}");

        // At eps=.02 and k=10, the variance-matched mixture has kurtosis
        // 3*((1-eps)+eps*k^4)/((1-eps)+eps*k^2)^2 ~= 67.9. A simple tail
        // count is much more stable than estimating that fourth moment in a
        // unit test, while still distinguishing sparse spikes from N(0,1).
        let tail_count = flat.iter().filter(|value| value.abs() > 4.0).count();
        assert!(
            (1_200..=2_800).contains(&tail_count),
            "expected sparse contaminated tails, got {tail_count}/{} draws with |x|>4",
            flat.len()
        );
    }

    #[test]
    fn student_t3_bank_records_its_noise_law_in_metadata() {
        let dir = temp_bank_dir("generate_dgp_bank_ou_t3");
        let path = dir.join("dgp_bank.safetensors");
        generate_dgp_bank("ou", 42, 1000, 2, 3, iid(NoiseKind::StudentT3), &path).unwrap();

        let bank = load_dgp_bank_safetensors(&path).unwrap();
        assert_eq!(bank.metadata.get("noise_kind").map(String::as_str), Some("student_t3"));
    }

    #[test]
    fn laplace_bank_records_its_noise_law_in_metadata() {
        let dir = temp_bank_dir("generate_dgp_bank_ou_laplace");
        let path = dir.join("dgp_bank.safetensors");
        generate_dgp_bank("ou", 42, 1000, 2, 3, iid(NoiseKind::Laplace), &path).unwrap();

        let bank = load_dgp_bank_safetensors(&path).unwrap();
        assert_eq!(bank.metadata.get("noise_kind").map(String::as_str), Some("laplace"));
    }

    #[test]
    fn contaminated_bank_records_its_noise_law_in_metadata() {
        let dir = temp_bank_dir("generate_dgp_bank_ou_contaminated");
        let path = dir.join("dgp_bank.safetensors");
        generate_dgp_bank("ou", 42, 1000, 2, 3, iid(NoiseKind::Contaminated), &path).unwrap();

        let bank = load_dgp_bank_safetensors(&path).unwrap();
        assert_eq!(
            bank.metadata.get("noise_kind").map(String::as_str),
            Some("contaminated")
        );
    }

    #[test]
    fn generated_stochastic_bank_round_trips_full_serie_header() {
        let dir = temp_bank_dir("generate_dgp_bank_ou");
        let path = dir.join("dgp_bank.safetensors");
        generate_dgp_bank("ou", 42, 1000, 2, 3, iid(NoiseKind::Gaussian), &path).unwrap();

        let bank = load_dgp_bank_safetensors(&path).unwrap();
        assert_eq!(bank.clean().shape()[0], 2);
        assert_eq!(bank.noise().shape()[0], 3);
        assert_eq!(bank.clean().shape()[1..], bank.noise().shape()[1..]);
        assert_eq!(bank.feature_names, vec!["f0"]);
        assert!(bank.noisy_tensor_names().is_empty());

        let meta = &bank.metadata;
        assert_eq!(
            meta.get("schema").map(String::as_str),
            Some("serie.dgp_bank_safetensors.v1")
        );
        assert_eq!(meta.get("source_name").map(String::as_str), Some("ou"));
        assert_eq!(meta.get("noise_kind").map(String::as_str), Some("gaussian"));
        assert_eq!(meta.get("is_stochastic").map(String::as_str), Some("true"));
        assert_eq!(meta.get("draw_design").map(String::as_str), Some("joint"));
        assert_eq!(meta.get("data_engine").map(String::as_str), Some("serie_rs"));
        assert_eq!(meta.get("trajectory_seed_base").map(String::as_str), Some("42"));
        assert_eq!(meta.get("noise_seed_base").map(String::as_str), Some("1000"));
        assert_eq!(
            meta.get("axis_order").map(String::as_str),
            Some("clean/noise=[draw,T,D]; noisy=[T,D]")
        );

        // The spawned draw-seed lists ARE the spawn_draw_seeds output — Python's
        // DGPBank validators re-derive and compare them on load.
        let trajectory_seeds: Vec<u64> = serde_json::from_str(meta.get("trajectory_draw_seeds").unwrap()).unwrap();
        assert_eq!(trajectory_seeds, spawn_draw_seeds(42, 2).unwrap());
        let noise_seeds: Vec<u64> = serde_json::from_str(meta.get("noise_draw_seeds").unwrap()).unwrap();
        assert_eq!(noise_seeds, spawn_draw_seeds(1000, 3).unwrap());

        let shape: serde_json::Value = serde_json::from_str(meta.get("shape").unwrap()).unwrap();
        assert_eq!(shape["clean"][0], 2);
        assert_eq!(shape["noise"][0], 3);
        assert_eq!(shape["clean"][1], shape["noise"][1]);

        // Two stochastic trajectory draws are distinct Brownian paths.
        assert_ne!(
            bank.clean().index_axis(ndarray::Axis(0), 0),
            bank.clean().index_axis(ndarray::Axis(0), 1)
        );

        std::fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn generated_deterministic_bank_marks_noise_only_design() {
        let dir = temp_bank_dir("generate_dgp_bank_lorenz");
        let path = dir.join("dgp_bank.safetensors");
        generate_dgp_bank("lorenz", 42, 1000, 1, 2, iid(NoiseKind::Gaussian), &path).unwrap();

        let bank = load_dgp_bank_safetensors(&path).unwrap();
        assert_eq!(bank.clean().shape()[0], 1);
        assert_eq!(bank.feature_names, vec!["f0", "f1", "f2"]);
        assert_eq!(bank.metadata.get("is_stochastic").map(String::as_str), Some("false"));
        assert_eq!(bank.metadata.get("draw_design").map(String::as_str), Some("noise_only"));

        std::fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn rossler_holdout_banks_materialize_distinct_clean_trajectories_and_initial_conditions() {
        let dir = temp_bank_dir("generate_rossler_initial_condition_holdout");
        let low_path = dir.join("low.safetensors");
        let high_path = dir.join("high.safetensors");
        generate_dgp_bank(
            "rossler_ic_holdout_0",
            42,
            11_000,
            1,
            1,
            iid(NoiseKind::Gaussian),
            &low_path,
        )
        .unwrap();
        generate_dgp_bank(
            "rossler_ic_holdout_7",
            42,
            11_007,
            1,
            1,
            iid(NoiseKind::Gaussian),
            &high_path,
        )
        .unwrap();

        let low = load_dgp_bank_safetensors(&low_path).unwrap();
        let high = load_dgp_bank_safetensors(&high_path).unwrap();
        assert_eq!(
            low.metadata.get("initial_condition").map(String::as_str),
            Some("[0.99,0.99,0.99]")
        );
        assert_eq!(
            high.metadata.get("initial_condition").map(String::as_str),
            Some("[1.01,1.01,1.01]")
        );
        assert_eq!(low.clean().shape(), &[1, 36_000, 3]);
        assert_eq!(high.clean().shape(), &[1, 36_000, 3]);
        assert_ne!(low.clean(), high.clean());
        assert_ne!(low.noise(), high.noise());

        std::fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn generate_dgp_bank_rejects_python_only_sources() {
        // ks/ks_param_shock are the ONLY remaining Python-generated sources
        // (ETDRK4 not ported); known-data sources route to the IPC reader.
        for source in ["ks", "ks_param_shock"] {
            let err = generate_dgp_bank(source, 42, 1000, 1, 1, iid(NoiseKind::Gaussian), "unused.safetensors")
                .unwrap_err()
                .to_string();
            assert!(err.contains("no serie-rs generator preset"), "{source}: {err}");
        }
    }

    #[cfg(not(feature = "known-bank"))]
    #[test]
    fn known_bank_requires_known_bank_feature_in_slim_builds() {
        let err = generate_dgp_bank(
            "hugadb_accel_emg",
            0,
            1000,
            1,
            1,
            iid(NoiseKind::Gaussian),
            "unused.safetensors",
        )
        .unwrap_err()
        .to_string();
        assert!(err.contains("known-data"), "{err}");
    }

    // ── known-data bank tests ──
    //
    // The generation/parity tests need the user-provisioned IPC assets under
    // SERIE_KNOWN_ROOT (or ~/.cache/serie/known); a box without them skips
    // instead of failing, the same guard the fern-cli-rs hugadb e2e uses.

    #[cfg(feature = "known-bank")]
    fn known_asset_or_skip(test_name: &str, subpath: &str) -> Option<PathBuf> {
        let path = known_data_root().unwrap().join(subpath);
        if path.is_file() {
            Some(path)
        } else {
            eprintln!("SKIP {test_name}: missing asset {}", path.display());
            None
        }
    }

    #[cfg(feature = "known-bank")]
    #[test]
    fn known_bank_rejects_multiple_trajectory_draws() {
        // Validated before any file access: no asset needed.
        let err = generate_dgp_bank(
            "hugadb_accel_emg",
            0,
            1000,
            2,
            1,
            iid(NoiseKind::Gaussian),
            "unused.safetensors",
        )
        .unwrap_err()
        .to_string();
        assert!(err.contains("n_trajectory_draws=1"), "{err}");
    }

    #[cfg(feature = "known-bank")]
    #[test]
    fn known_bank_rejects_nonzero_trajectory_seed_base() {
        let err = generate_dgp_bank("etth1", 42, 1000, 1, 1, iid(NoiseKind::Gaussian), "unused.safetensors")
            .unwrap_err()
            .to_string();
        assert!(err.contains("trajectory_seed_base=0"), "{err}");
    }

    #[cfg(feature = "known-bank")]
    #[test]
    fn generated_known_bank_round_trips_serie_header_and_asset_pin() {
        let Some(asset_path) = known_asset_or_skip(
            "generated_known_bank_round_trips_serie_header_and_asset_pin",
            "hugadb/hugadb_accel_emg.ipc",
        ) else {
            return;
        };
        let dir = temp_bank_dir("generate_dgp_bank_hugadb");
        let path = dir.join("dgp_bank.safetensors");
        generate_dgp_bank("hugadb_accel_emg", 0, 1000, 1, 2, iid(NoiseKind::Gaussian), &path).unwrap();

        let bank = load_dgp_bank_safetensors(&path).unwrap();
        assert_eq!(bank.clean().shape(), &[1, 13758, 20]);
        assert_eq!(bank.noise().shape(), &[2, 13758, 20]);
        assert_eq!(bank.feature_names[0], "RF_acc_x");
        assert_eq!(bank.feature_names[19], "L_EMG");

        let meta = &bank.metadata;
        assert_eq!(meta.get("source_name").map(String::as_str), Some("hugadb_accel_emg"));
        assert_eq!(meta.get("is_stochastic").map(String::as_str), Some("false"));
        assert_eq!(meta.get("draw_design").map(String::as_str), Some("noise_only"));
        assert_eq!(meta.get("data_engine").map(String::as_str), Some("serie_rs"));
        assert_eq!(meta.get("trajectory_seed_base").map(String::as_str), Some("0"));
        assert_eq!(
            meta.get("source_asset").map(String::as_str),
            Some("hugadb/hugadb_accel_emg.ipc")
        );
        // The pin is sha256[:16] of the asset bytes the generator parsed.
        let expected_sha = {
            use sha2::{Digest, Sha256};
            format!("{:x}", Sha256::digest(std::fs::read(&asset_path).unwrap()))[..16].to_string()
        };
        assert_eq!(meta.get("source_asset_sha"), Some(&expected_sha));
        let trajectory_seeds: Vec<u64> = serde_json::from_str(meta.get("trajectory_draw_seeds").unwrap()).unwrap();
        assert_eq!(trajectory_seeds, spawn_draw_seeds(0, 1).unwrap());

        // The header-only reader returns the same metadata map without
        // touching tensor data.
        let header_metadata = load_dgp_bank_metadata(&path).unwrap();
        assert_eq!(header_metadata.get("source_asset_sha"), Some(&expected_sha));
        assert_eq!(
            header_metadata.get("source_name").map(String::as_str),
            Some("hugadb_accel_emg")
        );

        std::fs::remove_dir_all(dir).ok();
    }

    /// Cross-language clean-data parity: unlike synthetic sources, known-data
    /// clean channels involve NO RNG, so the Rust-read clean matrix must be
    /// EXACTLY (bitwise f32) what Python's worker load produced. Pins below
    /// were dumped 2026-07-02 from Python (KnownLoadMode.CLEAN defaults — the
    /// exact worker load path, including ETT sentinel drops/imputation) with:
    ///
    /// ```text
    /// SERIE_KNOWN_ROOT=... uv run python - <<'EOF'
    /// import hashlib, numpy as np
    /// from serie.data import load_known_data
    /// from serie.pipeline import KnownMeta
    /// arr = load_known_data(KnownMeta.etth1()).values  # per source
    /// print(arr.shape, hashlib.sha256(arr.tobytes()).hexdigest())
    /// EOF
    /// ```
    ///
    /// If a pin mismatches, STOP and investigate the load-path difference —
    /// do not re-pin: accepting drift here silently changes every known-data
    /// run's training data.
    #[cfg(feature = "known-bank")]
    #[test]
    fn known_clean_channels_match_python_load_bitwise() {
        use sha2::{Digest, Sha256};

        // (source, T, D, sha256 of the row-major [T, D] f32 LE bytes).
        let pins = [
            (
                "etth1",
                17420,
                7,
                "a2e852e83a3fec3c53cab3ee81e0c83a106c6f80bf58c0080b26b2c031df750a",
            ),
            (
                "etth2",
                17420,
                4,
                "2169dc3679cd2f3ef0db0a32de4ec6710931f4310faa75a49b07091beeff3882",
            ),
            (
                "ettm1",
                69680,
                7,
                "0d0df325a8cf1d799afe608fd3f489512a7c25425f7bfb728562b89ef6a116c8",
            ),
            (
                "ettm2",
                69680,
                4,
                "d72b59a80ecc59d89fa927ccdaa53a9bef478801c71135d810c49fc89d851660",
            ),
            (
                "solar_energy",
                52560,
                137,
                "81ec08c014d67dbecf612b64d6aa577f08913cc30713b476c286da2d60f1383b",
            ),
            (
                "solar_energy_subset",
                52560,
                16,
                "5b61dd630c7b90d6307c1265b9ff1ca3819f65a62921a6f1235d62aa7df97661",
            ),
            (
                "hugadb_accel_emg",
                13758,
                20,
                "280c5485828c6edca9935589f2d7e86202532f3ea2f4b648afa6c475593d3aa6",
            ),
            (
                "hugadb_walk_bicycling_switch",
                13758,
                20,
                "4b3fef3e0edafe45491a497bb9b5323cf10b61a26d66d15cf8bf0309daaf3664",
            ),
        ];
        let dir = temp_bank_dir("known_clean_parity");
        for (source, timesteps, channels, expected_sha) in pins {
            let subpath = known_source_asset(source).unwrap().asset_subpath;
            if known_asset_or_skip("known_clean_channels_match_python_load_bitwise", subpath).is_none() {
                continue;
            }
            let path = dir.join(format!("{source}.safetensors"));
            generate_dgp_bank(source, 0, 1000, 1, 1, iid(NoiseKind::Gaussian), &path).unwrap();
            let bank = load_dgp_bank_safetensors(&path).unwrap();
            assert_eq!(bank.clean().shape(), &[1, timesteps, channels], "{source}");

            // clean is [1, T, D] in standard (row-major) layout, so iteration
            // order equals Python's C-order tobytes().
            let mut hasher = Sha256::new();
            for value in bank.clean().iter() {
                hasher.update(value.to_le_bytes());
            }
            assert_eq!(format!("{:x}", hasher.finalize()), expected_sha, "{source}");
        }

        // Verify the same 40-value prefix without redistributing raw sensor values.
        // The full-array SHA-256 pin above is unchanged; this additional digest
        // preserves a compact prefix check, subject to SHA-256 collision risk.
        let hugadb_path = dir.join("hugadb_accel_emg.safetensors");
        if hugadb_path.is_file() {
            let bank = load_dgp_bank_safetensors(&hugadb_path).unwrap();
            let actual_bits: Vec<u32> = bank.clean().iter().take(40).map(|v| v.to_bits()).collect();
            assert_eq!(actual_bits.len(), 40);
            let mut prefix_hasher = Sha256::new();
            for value in actual_bits {
                prefix_hasher.update(value.to_le_bytes());
            }
            assert_eq!(
                format!("{:x}", prefix_hasher.finalize()),
                "f8436012e0942611b9dafa859f2d502e9a189e14fc8fe8d4b28672ff2e372e9e"
            );
            eprintln!("verified HuGaDB 40-value prefix digest");
        }

        std::fs::remove_dir_all(dir).ok();
    }
}
