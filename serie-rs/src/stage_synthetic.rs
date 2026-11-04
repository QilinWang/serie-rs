#[cfg(feature = "pipeline")]
use crate::numeric_frame::{NumericColumn, NumericFrame, NumericFrameError};
use crate::stage_generate::{self, ShockEvent as NativeShockEvent};
#[cfg(feature = "pipeline")]
use crate::stage_noise::{NoiseConfig, NoiseRecipe, inject_noise};
use crate::stage_params::{
    ChuaParams, DiagonalParams, DoubleWellParams, GarchParams, KsParams, LgSsmParams, Lorenz96Params, LorenzParams,
    OuParams, RosslerParams, SeasonalArParams, SldsParams, SystemParams,
};
#[cfg(feature = "pipeline")]
use crate::stage_partition::{PartitionConfig, partition};
#[cfg(feature = "pipeline")]
use crate::stage_ready::{ReadyBundle, ReadyConfig, ready};
use std::error::Error;
use std::fmt::{Display, Formatter};

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// bundle transitions (stage fn change)
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// generate_timestep_major(config): TrajectorySpec -> SyntheticTrajectory
// generate(config): TrajectorySpec -> SyntheticTrajectory -> SyntheticBundle
// generate_ready_bundle(config, noise_config, partition_config, ready_config): TrajectorySpec -> ReadyBundle
// shock/system compile helpers: internal details (no extra bundle boundary change)
//
// flow
//   TrajectorySpec
//   -> s01_validate_generate_config
//   -> s01_compile_native_run_plan
//   -> stage_generate::generate_* / *_switch
//   -> post-burn-in slice of the generated timestep stream
//   -> SyntheticTrajectory
//
// flow (optional NumericFrame pipeline adapter)
//   SyntheticTrajectory
//   -> s01_timestep_major_values_to_frame
//   -> SyntheticBundle
//
// flow (full synthetic pipeline helper)
//   TrajectorySpec
//   -> generate
//   -> inject_noise
//   -> partition
//   -> ready
//   -> ReadyBundle
//
// flow (train-std-scaled noise helper)
//   TrajectorySpec
//   -> generate
//   -> partition clean data to compute train std
//   -> inject_noise with NoiseRecipe::Scaled { channel_std: train_std }
//   -> partition noisy data
//   -> ready
//   -> ReadyBundle

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// synthetic
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// synthetic structs

/// One post-burn-in trajectory in timestep-major `[T, D]` order.
///
/// This is the synthetic generation boundary shared by DGP-bank production
/// and the optional NumericFrame pipeline. Values remain f64 until the consumer
/// chooses its storage dtype, matching the native integrators.
#[derive(Debug, Clone)]
pub struct SyntheticTrajectory {
    pub config: TrajectorySpec,
    pub timestep_major_values: Vec<f64>,
    pub timesteps: usize,
    pub channels: usize,
    pub feature_names: Vec<String>,
}

/// Validation or native-generation failure before any NumericFrame is built.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SyntheticGenerationError {
    message: String,
}

impl Display for SyntheticGenerationError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(&self.message)
    }
}

impl Error for SyntheticGenerationError {}

#[derive(Debug, Clone)]
#[cfg(feature = "pipeline")]
pub struct SyntheticBundle {
    pub config: TrajectorySpec,
    pub frame: NumericFrame,
    pub feature_names: Vec<String>,
    pub channels: usize,
}

/// Generate the post-burn-in timestep-major trajectory without a table engine.
///
/// ```
/// use serie_rs::stage_synthetic::{generate_timestep_major, TrajectorySpec};
///
/// let mut spec = TrajectorySpec::lorenz();
/// spec.steps = 8;
/// let trajectory = generate_timestep_major(spec)?;
/// assert_eq!((trajectory.timesteps, trajectory.channels), (9, 3));
/// # Ok::<(), serie_rs::stage_synthetic::SyntheticGenerationError>(())
/// ```
pub fn generate_timestep_major(config: TrajectorySpec) -> Result<SyntheticTrajectory, SyntheticGenerationError> {
    s01_validate_generate_config(&config)?;

    let generated_values = s01_generate_timestep_major_f64(&config)?;
    // stage_generate returns timestep-major data including the initial condition row.
    // The post-burn-in count here is an output-row count, independent of a table adapter.
    let kept_output_rows = config.steps + 1 - config.burn_in;
    let start_value = config
        .burn_in
        .checked_mul(config.dim)
        .ok_or_else(|| s01_compute_error("synthetic generate: burn-in offset overflow"))?;
    let kept_value_count = kept_output_rows
        .checked_mul(config.dim)
        .ok_or_else(|| s01_compute_error("synthetic generate: output shape overflow"))?;
    let end_value = start_value
        .checked_add(kept_value_count)
        .ok_or_else(|| s01_compute_error("synthetic generate: output range overflow"))?;
    let timestep_major_values = generated_values.get(start_value..end_value).ok_or_else(|| {
        s01_compute_error(format!(
            "synthetic generate: native output has {} values, expected at least {end_value}",
            generated_values.len()
        ))
    })?;
    let feature_names = (0..config.dim).map(|channel| format!("f{channel}")).collect();

    Ok(SyntheticTrajectory {
        timestep_major_values: timestep_major_values.to_vec(),
        timesteps: kept_output_rows,
        channels: config.dim,
        feature_names,
        config,
    })
}

// synthetic NumericFrame stage fn
#[cfg(feature = "pipeline")]
pub fn generate(config: TrajectorySpec) -> Result<SyntheticBundle, NumericFrameError> {
    let trajectory = generate_timestep_major(config).map_err(s01_numeric_frame_error)?;
    let frame = s01_timestep_major_values_to_frame(
        &trajectory.timestep_major_values,
        0,
        trajectory.timesteps,
        trajectory.channels,
    )?;
    let feature_names = frame.column_names();

    Ok(SyntheticBundle {
        feature_names,
        channels: frame.width(),
        config: trajectory.config,
        frame,
    })
}

// synthetic full-pipeline stage helper
#[cfg(feature = "pipeline")]
pub fn generate_ready_bundle(
    config: TrajectorySpec,
    noise_config: &NoiseConfig,
    partition_config: PartitionConfig,
    ready_config: ReadyConfig,
) -> Result<ReadyBundle, NumericFrameError> {
    let synthetic_bundle = generate(config)?;
    let noisy_bundle = inject_noise(synthetic_bundle, noise_config)?;
    let partitioned_bundle = partition(noisy_bundle, partition_config)?;
    ready(partitioned_bundle, ready_config)
}

#[cfg(feature = "pipeline")]
pub fn generate_ready_bundle_train_std_noise(
    config: TrajectorySpec,
    sigma: f64,
    noise_seed: u64,
    partition_config: PartitionConfig,
    ready_config: ReadyConfig,
) -> Result<ReadyBundle, NumericFrameError> {
    let synthetic_bundle = generate(config)?;
    let clean_like = inject_noise(
        synthetic_bundle.clone(),
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
    let noisy_bundle = inject_noise(synthetic_bundle, &noisy_config)?;
    let noisy_partitioned = partition(noisy_bundle, partition_config)?;
    ready(noisy_partitioned, ready_config)
}

#[derive(Debug, Clone)]
enum NativeRunPlan {
    // One generator call, with optional regime switches/state shocks inline.
    OneRun {
        param_sets: Vec<SystemParams>,
        shocks: Vec<NativeShockEvent>,
    },
    // Two independent generator calls, then splice trajectories at one step.
    TwoRun {
        params_b: SystemParams,
        x0_b: Vec<f64>,
        splice_step: usize,
    },
}

fn s01_compute_error(message: impl Into<String>) -> SyntheticGenerationError {
    SyntheticGenerationError {
        message: message.into(),
    }
}

#[cfg(feature = "pipeline")]
fn s01_numeric_frame_error(error: SyntheticGenerationError) -> NumericFrameError {
    NumericFrameError::new(error.to_string())
}

fn s01_validate_generate_config(config: &TrajectorySpec) -> Result<(), SyntheticGenerationError> {
    if config.steps == 0 {
        return Err(s01_compute_error("synthetic generate: steps must be > 0"));
    }
    if config.burn_in >= config.steps {
        return Err(s01_compute_error(format!(
            "synthetic generate: burn_in ({}) must be < steps ({})",
            config.burn_in, config.steps
        )));
    }
    if config.dt <= 0.0
        && !matches!(
            config.system,
            ODESystem::Henon | ODESystem::Logistic | ODESystem::SLDS | ODESystem::SeasonalAR | ODESystem::GARCH
        )
    {
        return Err(s01_compute_error(
            "synthetic generate: dt must be > 0 for continuous systems",
        ));
    }

    let expected_dim = s01_expected_dim(&config.params)?;
    if config.dim != expected_dim {
        return Err(s01_compute_error(format!(
            "synthetic generate: dim mismatch for {:?}: config.dim={}, expected {}",
            config.system, config.dim, expected_dim
        )));
    }
    if config.x0.len() != expected_dim {
        return Err(s01_compute_error(format!(
            "synthetic generate: x0 length mismatch for {:?}: got {}, expected {}",
            config.system,
            config.x0.len(),
            expected_dim
        )));
    }
    if !s01_params_match_system(config.system, &config.params) {
        return Err(s01_compute_error(format!(
            "synthetic generate: params variant does not match system {:?}",
            config.system
        )));
    }

    match config.system {
        ODESystem::Lorenz
        | ODESystem::Rossler
        | ODESystem::HyperRossler
        | ODESystem::Chua
        | ODESystem::Duffing
        | ODESystem::VanDerPol
        | ODESystem::Lorenz96 => {
            if config.method != IntegrationMethod::RK4 {
                return Err(s01_compute_error(format!(
                    "synthetic generate: {:?} currently requires RK4 in stage_generate",
                    config.system
                )));
            }
        }
        ODESystem::Henon | ODESystem::Logistic => {
            if config.method != IntegrationMethod::Map {
                return Err(s01_compute_error(format!(
                    "synthetic generate: {:?} currently requires Map integration",
                    config.system
                )));
            }
        }
        ODESystem::OU | ODESystem::DoubleWell => {
            if config.method != IntegrationMethod::Euler {
                return Err(s01_compute_error(format!(
                    "synthetic generate: {:?} currently requires Euler integration",
                    config.system
                )));
            }
        }
        ODESystem::SLDS | ODESystem::SeasonalAR | ODESystem::GARCH => {
            // Discrete statistical systems: method and dt are schema
            // placeholders (the presets carry Euler/dt=1.0 for schema compat,
            // like Python's slds_spec "dt=1.0 is a dummy"). Python's
            // generate_from_trajectory_spec validates method only for
            // ode/map/pde kinds; the discrete simulators step on the integer
            // index, so nothing to check here.
        }
        ODESystem::LgSsm => {
            // Runs on the discrete recurrence path but dt is a REAL sampling
            // interval (A = exp(F·dt)), so LgSsm is deliberately NOT in the
            // dt-exempt list above; method stays the Euler schema placeholder
            // like the other discrete-path systems.
        }
        ODESystem::KS
        | ODESystem::Linear
        | ODESystem::Diagonal
        | ODESystem::JordanBlock
        | ODESystem::Rotation2D
        | ODESystem::Thomas
        | ODESystem::Chen
        | ODESystem::LotkaVolterra => {
            return Err(s01_compute_error(format!(
                "synthetic generate: {:?} is cataloged in stage_synthetic but not yet wired into stage_generate",
                config.system
            )));
        }
    }

    Ok(())
}

fn s01_generate_timestep_major_f64(config: &TrajectorySpec) -> Result<Vec<f64>, SyntheticGenerationError> {
    match config.system {
        ODESystem::Lorenz => s01_generate_ode_like(config, "lorenz"),
        ODESystem::Rossler => s01_generate_ode_like(config, "rossler"),
        ODESystem::HyperRossler => s01_generate_ode_like(config, "hyper_rossler"),
        ODESystem::Chua => s01_generate_ode_like(config, "chua"),
        ODESystem::Duffing => s01_generate_ode_like(config, "duffing"),
        ODESystem::VanDerPol => s01_generate_ode_like(config, "vanderpol"),
        ODESystem::Lorenz96 => s01_generate_ode_like(config, "lorenz96"),
        ODESystem::Henon => s01_generate_map_like(config, "henon"),
        ODESystem::Logistic => s01_generate_map_like(config, "logistic"),
        ODESystem::OU => s01_generate_sde_like(config, "ou"),
        ODESystem::DoubleWell => s01_generate_sde_like(config, "doublewell"),
        ODESystem::LgSsm => s01_generate_discrete_like(config, "lg_ssm"),
        ODESystem::SLDS => s01_generate_discrete_like(config, "slds"),
        ODESystem::SeasonalAR => s01_generate_discrete_like(config, "seasonal_ar"),
        ODESystem::GARCH => s01_generate_discrete_like(config, "garch"),
        _ => Err(s01_compute_error(format!(
            "synthetic generate: {:?} is not yet backed by a native generator",
            config.system
        ))),
    }
}

fn s01_generate_ode_like(config: &TrajectorySpec, system_key: &str) -> Result<Vec<f64>, SyntheticGenerationError> {
    match s01_compile_native_run_plan(config)? {
        NativeRunPlan::OneRun { param_sets, shocks } => {
            let (param_matrix, n_params) = s01_flatten_param_sets(&param_sets)?;
            stage_generate::generate_ode(
                system_key,
                &config.x0,
                config.steps,
                config.dt,
                &shocks,
                &param_matrix,
                n_params,
            )
            .map_err(s01_compute_error)
        }
        NativeRunPlan::TwoRun {
            params_b,
            x0_b,
            splice_step,
        } => {
            let params_a = s01_flatten_one_regime_params(&config.params)?;
            let params_b = s01_flatten_one_regime_params(&params_b)?;
            stage_generate::generate_ode_switch(
                system_key,
                &config.x0,
                &x0_b,
                config.steps,
                config.dt,
                splice_step,
                &params_a,
                &params_b,
            )
            .map_err(s01_compute_error)
        }
    }
}

fn s01_generate_map_like(config: &TrajectorySpec, system_key: &str) -> Result<Vec<f64>, SyntheticGenerationError> {
    match s01_compile_native_run_plan(config)? {
        NativeRunPlan::OneRun { param_sets, shocks } => {
            if !shocks.is_empty() {
                return Err(s01_compute_error(format!(
                    "synthetic generate: {} map shocks are not wired in stage_generate yet",
                    system_key
                )));
            }
            let (param_matrix, n_params) = s01_flatten_param_sets(&param_sets)?;
            stage_generate::generate_map(system_key, &config.x0, config.steps, &param_matrix, n_params)
                .map_err(s01_compute_error)
        }
        NativeRunPlan::TwoRun {
            params_b,
            x0_b,
            splice_step,
        } => {
            let params_a = s01_flatten_one_regime_params(&config.params)?;
            let params_b = s01_flatten_one_regime_params(&params_b)?;
            stage_generate::generate_map_switch(
                system_key,
                &config.x0,
                &x0_b,
                config.steps,
                splice_step,
                &params_a,
                &params_b,
            )
            .map_err(s01_compute_error)
        }
    }
}

fn s01_generate_sde_like(config: &TrajectorySpec, system_key: &str) -> Result<Vec<f64>, SyntheticGenerationError> {
    match s01_compile_native_run_plan(config)? {
        NativeRunPlan::OneRun { param_sets, shocks } => {
            let (param_matrix, n_params) = s01_flatten_param_sets(&param_sets)?;
            let sigma_per_regime: Result<Vec<f64>, SyntheticGenerationError> =
                param_sets.iter().map(s01_sigma_noise).collect();
            stage_generate::generate_sde_seeded(
                system_key,
                &config.x0,
                config.steps,
                config.dt,
                &sigma_per_regime?,
                config.seed,
                &shocks,
                &param_matrix,
                n_params,
            )
            .map_err(s01_compute_error)
        }
        NativeRunPlan::TwoRun {
            params_b,
            x0_b,
            splice_step,
        } => {
            // Same two-run composition as the ODE/map splice; the diffusion
            // coefficient for each run comes from that run's own params
            // (Python: sigma = active_params.sigma_noise per trajectory).
            let sigma_a = s01_sigma_noise(&config.params)?;
            let sigma_b = s01_sigma_noise(&params_b)?;
            let params_a = s01_flatten_one_regime_params(&config.params)?;
            let params_b = s01_flatten_one_regime_params(&params_b)?;
            stage_generate::generate_sde_switch_seeded(
                system_key,
                &config.x0,
                &x0_b,
                config.steps,
                config.dt,
                sigma_a,
                sigma_b,
                splice_step,
                config.seed,
                &params_a,
                &params_b,
            )
            .map_err(s01_compute_error)
        }
    }
}

fn s01_generate_discrete_like(config: &TrajectorySpec, system_key: &str) -> Result<Vec<f64>, SyntheticGenerationError> {
    match s01_compile_native_run_plan(config)? {
        NativeRunPlan::OneRun { param_sets, shocks } => {
            let (param_matrix, n_params) = s01_flatten_param_sets(&param_sets)?;
            stage_generate::generate_discrete_seeded(
                system_key,
                &config.x0,
                config.steps,
                config.dt,
                config.seed,
                &shocks,
                &param_matrix,
                n_params,
            )
            .map_err(s01_compute_error)
        }
        NativeRunPlan::TwoRun {
            params_b,
            x0_b,
            splice_step,
        } => {
            let params_a = s01_flatten_one_regime_params(&config.params)?;
            let params_b = s01_flatten_one_regime_params(&params_b)?;
            stage_generate::generate_discrete_switch_seeded(
                system_key,
                &config.x0,
                &x0_b,
                config.steps,
                config.dt,
                splice_step,
                config.seed,
                &params_a,
                &params_b,
            )
            .map_err(s01_compute_error)
        }
    }
}

fn s01_compile_native_run_plan(config: &TrajectorySpec) -> Result<NativeRunPlan, SyntheticGenerationError> {
    if config.shocks.is_stationary() {
        return Ok(NativeRunPlan::OneRun {
            param_sets: vec![config.params.clone()],
            shocks: vec![],
        });
    }

    if config.shocks.n_shocks() != 1 {
        return Err(s01_compute_error(
            "synthetic generate: multi-event ShockSchedule is not wired yet; pass exactly one event",
        ));
    }

    let event = &config.shocks.events()[0];
    let usable_steps = config.usable_steps();
    // CHANGED: shock times are interpreted over the usable post-burn-in segment.
    // So t_start=0.35 means "35% into the returned trajectory", not "35% into
    // the pre-burn-in raw integration prefix".
    let first_change_step = config.burn_in + (event.t_start() * usable_steps as f64) as usize;
    let last_change_step = config.burn_in + (event.timing().t_end * usable_steps as f64) as usize;

    match event {
        ShockEvent::ParamsReplace { timing, params_after } => {
            // One transition model:
            // compile the requested number of scheduled parameter updates inside
            // the transition window. One update is the abrupt case. More updates
            // add intermediate regimes between before and after.
            s01_compile_param_transition_updates(
                &config.params,
                params_after,
                first_change_step,
                last_change_step,
                timing.n_transition_updates,
                config.steps,
                "parameter transition",
            )
        }
        ShockEvent::StateKick { timing, state_idx, eps } => {
            if !timing.uses_single_change_step() {
                return Err(s01_compute_error(
                    "synthetic generate: StateKick must use exactly one change step",
                ));
            }
            if !(1 <= first_change_step && first_change_step <= config.steps) {
                return Err(s01_compute_error(format!(
                    "synthetic generate: state kick step {} not in [1, {}]",
                    first_change_step, config.steps
                )));
            }
            if *state_idx >= config.dim {
                return Err(s01_compute_error(format!(
                    "synthetic generate: state_idx {} out of range [0, {})",
                    state_idx, config.dim
                )));
            }
            Ok(NativeRunPlan::OneRun {
                param_sets: vec![config.params.clone()],
                shocks: vec![NativeShockEvent::StatePerturbation {
                    step: first_change_step,
                    state_idx: *state_idx,
                    eps: *eps,
                }],
            })
        }
        ShockEvent::Splice {
            timing,
            ic_b,
            params_after,
        } => {
            if !timing.uses_single_change_step() {
                return Err(s01_compute_error(
                    "synthetic generate: Splice must use exactly one change step",
                ));
            }
            if !(first_change_step <= config.steps) {
                return Err(s01_compute_error(format!(
                    "synthetic generate: splice step {} not in [0, {}]",
                    first_change_step, config.steps
                )));
            }
            if ic_b.len() != config.dim {
                return Err(s01_compute_error(format!(
                    "synthetic generate: splice ic_b length mismatch: got {}, expected {}",
                    ic_b.len(),
                    config.dim
                )));
            }
            Ok(NativeRunPlan::TwoRun {
                params_b: params_after.clone().unwrap_or_else(|| config.params.clone()),
                x0_b: ic_b.clone(),
                splice_step: first_change_step,
            })
        }
    }
}

fn s01_expected_dim(params: &SystemParams) -> Result<usize, SyntheticGenerationError> {
    match params {
        SystemParams::Lorenz(_) => Ok(3),
        SystemParams::Rossler(_) => Ok(3),
        SystemParams::HyperRossler(_) => Ok(4),
        SystemParams::Chua(_) => Ok(3),
        SystemParams::Duffing(_) => Ok(2),
        SystemParams::VanDerPol(_) => Ok(2),
        SystemParams::Lorenz96(params) => Ok(params.dim),
        SystemParams::Henon(_) => Ok(2),
        SystemParams::Logistic(_) => Ok(1),
        SystemParams::OU(_) => Ok(1),
        SystemParams::DoubleWell(_) => Ok(1),
        SystemParams::LgSsm(params) => Ok(2 * params.gammas.len()),
        SystemParams::SLDS(_) => Ok(1),
        SystemParams::SeasonalAR(_) => Ok(1),
        SystemParams::GARCH(_) => Ok(2),
        SystemParams::KS(params) => Ok(params.nx),
        SystemParams::Diagonal(params) => Ok(params.eigenvalues.len()),
    }
}

fn s01_params_match_system(system: ODESystem, params: &SystemParams) -> bool {
    matches!(
        (system, params),
        (ODESystem::Lorenz, SystemParams::Lorenz(_))
            | (ODESystem::Rossler, SystemParams::Rossler(_))
            | (ODESystem::HyperRossler, SystemParams::HyperRossler(_))
            | (ODESystem::Chua, SystemParams::Chua(_))
            | (ODESystem::Duffing, SystemParams::Duffing(_))
            | (ODESystem::VanDerPol, SystemParams::VanDerPol(_))
            | (ODESystem::Lorenz96, SystemParams::Lorenz96(_))
            | (ODESystem::Henon, SystemParams::Henon(_))
            | (ODESystem::Logistic, SystemParams::Logistic(_))
            | (ODESystem::OU, SystemParams::OU(_))
            | (ODESystem::DoubleWell, SystemParams::DoubleWell(_))
            | (ODESystem::LgSsm, SystemParams::LgSsm(_))
            | (ODESystem::SLDS, SystemParams::SLDS(_))
            | (ODESystem::SeasonalAR, SystemParams::SeasonalAR(_))
            | (ODESystem::GARCH, SystemParams::GARCH(_))
            | (ODESystem::KS, SystemParams::KS(_))
            | (ODESystem::Diagonal, SystemParams::Diagonal(_))
    )
}

// Flatten one regime's typed params into the ordered numeric vector that
// stage_generate expects for a single regime.
fn s01_flatten_one_regime_params(params: &SystemParams) -> Result<Vec<f64>, SyntheticGenerationError> {
    match params {
        SystemParams::Lorenz(_)
        | SystemParams::Rossler(_)
        | SystemParams::HyperRossler(_)
        | SystemParams::Chua(_)
        | SystemParams::Duffing(_)
        | SystemParams::VanDerPol(_)
        | SystemParams::Lorenz96(_)
        | SystemParams::Henon(_)
        | SystemParams::Logistic(_)
        | SystemParams::OU(_)
        | SystemParams::DoubleWell(_)
        | SystemParams::LgSsm(_)
        | SystemParams::SLDS(_)
        | SystemParams::SeasonalAR(_)
        | SystemParams::GARCH(_) => Ok(params.as_f64_vec()),
        _ => Err(s01_compute_error(format!(
            "synthetic generate: stage_generate parameter flattening is not wired for {:?}",
            params
        ))),
    }
}

fn s01_flatten_param_sets(param_sets: &[SystemParams]) -> Result<(Vec<f64>, usize), SyntheticGenerationError> {
    if param_sets.is_empty() {
        return Err(s01_compute_error("synthetic generate: param_sets cannot be empty"));
    }

    let first_regime_values = s01_flatten_one_regime_params(&param_sets[0])?;
    let n_params = first_regime_values.len();
    let mut flat = Vec::with_capacity(param_sets.len() * n_params);
    flat.extend_from_slice(&first_regime_values);

    for params in &param_sets[1..] {
        let regime_values = s01_flatten_one_regime_params(params)?;
        if regime_values.len() != n_params {
            return Err(s01_compute_error(
                "synthetic generate: inconsistent flattened parameter counts across regimes",
            ));
        }
        flat.extend_from_slice(&regime_values);
    }

    Ok((flat, n_params))
}

fn s01_sigma_noise(params: &SystemParams) -> Result<f64, SyntheticGenerationError> {
    match params {
        SystemParams::OU(params) => Ok(params.sigma_noise),
        SystemParams::DoubleWell(params) => Ok(params.sigma_noise),
        _ => Err(s01_compute_error(format!(
            "synthetic generate: {:?} does not expose sigma_noise for SDE generation",
            params
        ))),
    }
}

fn s01_compile_param_transition_updates(
    before: &SystemParams,
    after: &SystemParams,
    first_change_step: usize,
    last_change_step: usize,
    n_transition_updates: usize,
    max_valid_step: usize,
    transition_name: &str,
) -> Result<NativeRunPlan, SyntheticGenerationError> {
    if n_transition_updates == 0 {
        return Err(s01_compute_error(format!(
            "synthetic generate: {} requires at least one transition update",
            transition_name
        )));
    }
    if !(1 <= first_change_step && first_change_step <= max_valid_step) {
        return Err(s01_compute_error(format!(
            "synthetic generate: {} first_change_step {} not in [1, {}]",
            transition_name, first_change_step, max_valid_step
        )));
    }
    if n_transition_updates > 1 && !(first_change_step < last_change_step && last_change_step <= max_valid_step) {
        return Err(s01_compute_error(format!(
            "synthetic generate: {} requires 1 <= first_change_step < last_change_step <= {}, got {}..{}",
            transition_name, max_valid_step, first_change_step, last_change_step
        )));
    }

    let transition_window_step_count = last_change_step - first_change_step;
    // CHG: exempt the abrupt case from the window-gap check. With
    // n_transition_updates == 1 and t_start == t_end the window is zero steps
    // and the single update fires at first_change_step, matching the Python
    // instantaneous branch (one ParamChangeEvent at shock_step,
    // serie/integrators.py generate_from_trajectory_spec). Only multi-update
    // transitions need step gaps to spread across.
    if n_transition_updates > 1 && n_transition_updates > transition_window_step_count {
        return Err(s01_compute_error(format!(
            "synthetic generate: {} asks for {} transition updates across only {} step gaps",
            transition_name, n_transition_updates, transition_window_step_count
        )));
    }

    let transition_update_count = n_transition_updates;
    let mut param_sets = Vec::with_capacity(transition_update_count + 1);
    param_sets.push(before.clone());
    let mut shocks = Vec::with_capacity(transition_update_count);

    for update_index in 0..transition_update_count {
        let change_progress = (update_index + 1) as f64 / transition_update_count as f64;
        let scheduled_change_step = if transition_update_count == 1 {
            first_change_step
        } else {
            first_change_step + ((update_index + 1) * transition_window_step_count) / transition_update_count
        };
        // CHG: the single-update case is the abrupt regime replacement —
        // Python's instantaneous branch uses event.params_after verbatim, so
        // take it directly (this is also the one path where structural fields
        // such as Seasonal-AR S may change, since no interpolation happens).
        // Windowed transitions interpolate every scheduled update: the lerp
        // form lands transition fields exactly on `after` at α = 1.0 while
        // structural fields stay fixed (see SystemParams::interpolate_toward).
        let interpolated_params = if transition_update_count == 1 {
            after.clone()
        } else {
            before
                .interpolate_toward(after, change_progress)
                .map_err(s01_compute_error)?
        };
        param_sets.push(interpolated_params);
        shocks.push(NativeShockEvent::ParamChange {
            step: scheduled_change_step,
            param_set_idx: param_sets.len() - 1,
        });
    }

    Ok(NativeRunPlan::OneRun { param_sets, shocks })
}

#[cfg(feature = "pipeline")]
fn s01_timestep_major_values_to_frame(
    timestep_major_values: &[f64],
    start_row: usize,
    n_rows: usize,
    dim: usize,
) -> Result<NumericFrame, NumericFrameError> {
    let end_row = start_row + n_rows;
    let mut columns: Vec<NumericColumn> = Vec::with_capacity(dim);

    for d in 0..dim {
        let mut values = Vec::with_capacity(n_rows);
        for t in start_row..end_row {
            values.push(timestep_major_values[t * dim + d]);
        }
        columns.push(NumericColumn::new(format!("f{}", d).into(), values.as_slice()));
    }

    NumericFrame::new(n_rows, columns)
}

// ── Integration method ──

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IntegrationMethod {
    RK4,
    Euler,  // also used for SDE Euler-Maruyama
    Map,    // discrete iteration, no dt
    ETDRK4, // KS PDE spectral method
}

// ── System enum ──
// Every system Python knew about, including commented-out and future ones.

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ODESystem {
    // Continuous ODEs
    Lorenz,
    Rossler,
    HyperRossler,
    Chua,
    Duffing,
    VanDerPol,
    Lorenz96,

    // Discrete maps
    Henon,
    Logistic,

    // SDEs
    OU,
    DoubleWell,
    // Continuous-time linear SDE sampled EXACTLY at dt (matrix OU / linear
    // Gaussian state-space); generated by the discrete recurrence path.
    LgSsm,

    // Discrete statistical
    SLDS,
    SeasonalAR,
    GARCH,

    // PDEs
    KS,

    // Designed (eigenvalue recovery tests)
    Linear,
    Diagonal,
    JordanBlock,
    Rotation2D,

    // Future / catalog
    Thomas,
    Chen,
    LotkaVolterra,
}

impl ODESystem {
    pub fn is_stochastic(&self) -> bool {
        matches!(
            self,
            Self::OU | Self::DoubleWell | Self::LgSsm | Self::SLDS | Self::SeasonalAR | Self::GARCH
        )
    }
}

#[derive(Debug, Clone)]
pub struct TrajectorySpec {
    pub system: ODESystem,
    pub params: SystemParams,
    pub dt: f64,
    pub steps: usize,
    pub dim: usize,
    pub burn_in: usize,
    pub method: IntegrationMethod,
    pub x0: Vec<f64>,
    pub seed: u64,
    pub shocks: ShockSchedule, // empty = stationary
}

impl TrajectorySpec {
    pub fn total_time(&self) -> f64 {
        self.steps as f64 * self.dt
    }
    pub fn usable_steps(&self) -> usize {
        self.steps - self.burn_in
    }

    /// Stable identifier for DB/filenames. Includes shock type if any.
    pub fn tag(&self) -> String {
        let base = format!("{:?}", self.system).to_lowercase();
        if self.shocks.is_stationary() {
            return base;
        }
        if self.shocks.n_shocks() == 1 {
            return match &self.shocks.events()[0] {
                ShockEvent::Splice { .. } => format!("{}_switch", base),
                ShockEvent::ParamsReplace { .. } => format!("{}_param_shock", base),
                ShockEvent::StateKick { .. } => format!("{}_state_shock", base),
            };
        }
        format!("{}_shock_x{}", base, self.shocks.n_shocks())
    }

    // ══════════════════════════════════════════════════════════════════
    //  Continuous ODEs
    // ══════════════════════════════════════════════════════════════════

    pub fn lorenz() -> Self {
        Self {
            system: ODESystem::Lorenz,
            params: SystemParams::Lorenz(LorenzParams {
                sigma: 10.0,
                rho: 28.0,
                beta: 8.0 / 3.0,
            }),
            dt: 1e-2,
            steps: 35999,
            dim: 3,
            burn_in: 0,
            method: IntegrationMethod::RK4,
            x0: vec![1.0, 0.98, 1.1],
            seed: 0,
            shocks: ShockSchedule::stationary(),
        }
    }

    pub fn lorenz_param_shock() -> Self {
        let mut spec = Self::lorenz();
        spec.shocks = ShockSchedule::single(ShockEvent::params_replace(
            SystemParams::Lorenz(LorenzParams {
                sigma: 10.1,
                rho: 28.1,
                beta: 8.1 / 3.0,
            }),
            0.35,
        ));
        spec
    }

    pub fn lorenz_state_shock() -> Self {
        let mut spec = Self::lorenz();
        spec.shocks = ShockSchedule::single(ShockEvent::state_kick(0, 0.9, 0.35));
        spec
    }

    pub fn lorenz_switch() -> Self {
        let mut spec = Self::lorenz();
        spec.shocks = ShockSchedule::single(ShockEvent::trajectory_splice(
            vec![1.002, 0.982, 1.102],
            0.35,
            Some(SystemParams::Lorenz(LorenzParams {
                sigma: 10.0,
                rho: 28.1,
                beta: 8.0 / 3.0,
            })),
        ));
        spec
    }

    pub fn rossler() -> Self {
        Self {
            system: ODESystem::Rossler,
            params: SystemParams::Rossler(RosslerParams { a: 0.2, b: 0.2, c: 5.7 }),
            dt: 1e-2,
            steps: 35999,
            dim: 3,
            burn_in: 0,
            method: IntegrationMethod::RK4,
            x0: vec![1.0, 1.0, 1.0],
            seed: 0,
            shocks: ShockSchedule::stationary(),
        }
    }

    pub fn rossler_param_shock() -> Self {
        let mut spec = Self::rossler();
        spec.shocks = ShockSchedule::single(ShockEvent::params_replace(
            SystemParams::Rossler(RosslerParams {
                a: 0.25,
                b: 0.25,
                c: 5.75,
            }),
            0.35,
        ));
        spec
    }

    // ── Commented-out in Python, preserved as comments ──

    // pub fn hyper_rossler() -> Self {
    //     Self {
    //         system: ODESystem::HyperRossler,
    //         params: SystemParams::HyperRossler(params! {
    //             "a" => 0.25, "b" => 3.0, "c" => 0.5, "d" => 0.05
    //         }),
    //         dt: 1e-2, steps: 35999, dim: 4, burn_in: 0,
    //         method: IntegrationMethod::RK4,
    //         x0: vec![0.1, 0.0, 0.0, 0.0],
    //         seed: 0,
    //         shocks: ShockSchedule::stationary(),
    //     }
    // }

    pub fn chua() -> Self {
        Self {
            system: ODESystem::Chua,
            params: SystemParams::Chua(ChuaParams {
                alpha: 15.6,
                beta: 28.0,
                m1_inner_le1: -8.0 / 7.0,
                m0_outer_gt1: -5.0 / 7.0,
            }),
            dt: 5e-3,
            steps: 35999,
            dim: 3,
            burn_in: 0,
            method: IntegrationMethod::RK4,
            x0: vec![0.1, 0.0, 0.0],
            seed: 0,
            shocks: ShockSchedule::stationary(),
        }
    }

    pub fn chua_param_shock() -> Self {
        let mut spec = Self::chua();
        spec.shocks = ShockSchedule::single(ShockEvent::params_replace(
            SystemParams::Chua(ChuaParams {
                alpha: 15.9,
                beta: 28.5,
                m1_inner_le1: -8.1 / 7.0,
                m0_outer_gt1: -5.2 / 7.0,
            }),
            0.35,
        ));
        spec
    }

    pub fn chua_switch() -> Self {
        let mut spec = Self::chua();
        spec.shocks = ShockSchedule::single(ShockEvent::trajectory_splice(vec![0.11, 0.01, 0.02], 0.35, None));
        spec
    }

    // pub fn duffing() -> Self {
    //     Self {
    //         system: ODESystem::Duffing,
    //         params: SystemParams::Duffing(params! {
    //             "alpha" => 1.0, "beta" => -1.0, "delta" => 0.2,
    //             "gamma" => 0.65, "omega" => 1.2
    //         }),
    //         dt: 0.01, steps: 3600, dim: 2, burn_in: 0,
    //         method: IntegrationMethod::RK4,
    //         x0: vec![0.1, 0.0],
    //         seed: 0,
    //         shocks: ShockSchedule::stationary(),
    //     }
    // }

    // pub fn vanderpol() -> Self {
    //     Self {
    //         system: ODESystem::VanDerPol,
    //         params: SystemParams::VanDerPol(params! { "mu" => 2.0 }),
    //         dt: 0.01, steps: 36000, dim: 2, burn_in: 0,
    //         method: IntegrationMethod::RK4,
    //         x0: vec![0.1, 0.0],
    //         seed: 0,
    //         shocks: ShockSchedule::stationary(),
    //     }
    // }

    pub fn lorenz96() -> Self {
        Self::lorenz96_with_dim(6)
    }

    pub fn lorenz96_with_dim(dim: usize) -> Self {
        let mut x0 = vec![1.0; dim];
        x0[0] = 1.01;
        Self {
            system: ODESystem::Lorenz96,
            params: SystemParams::Lorenz96(Lorenz96Params { forcing: 8.0, dim }),
            dt: 7e-3,
            steps: 55000,
            dim,
            burn_in: 0,
            method: IntegrationMethod::RK4,
            x0,
            seed: 0,
            shocks: ShockSchedule::stationary(),
        }
    }

    pub fn lorenz96_switch() -> Self {
        let mut spec = Self::lorenz96();
        spec.shocks = ShockSchedule::single(ShockEvent::trajectory_splice(
            vec![0.99, 1.02, 1.02, 1.03, 1.01, 1.01],
            0.35,
            Some(SystemParams::Lorenz96(Lorenz96Params {
                forcing: 9.0,
                dim: spec.dim,
            })),
        ));
        spec
    }

    // ══════════════════════════════════════════════════════════════════
    //  Maps
    // ══════════════════════════════════════════════════════════════════

    // pub fn henon() -> Self {
    //     Self {
    //         system: ODESystem::Henon,
    //         params: SystemParams::Henon(params! { "a" => 1.4, "b" => 0.3 }),
    //         dt: 1.0, steps: 36000, dim: 2, burn_in: 0,
    //         method: IntegrationMethod::Map,
    //         x0: vec![0.1, 0.0],
    //         seed: 0,
    //         shocks: ShockSchedule::stationary(),
    //     }
    // }

    // pub fn logistic() -> Self {
    //     Self {
    //         system: ODESystem::Logistic,
    //         params: SystemParams::Logistic(params! { "r" => 3.9 }),
    //         dt: 1.0, steps: 36000, dim: 1, burn_in: 0,
    //         method: IntegrationMethod::Map,
    //         x0: vec![0.4],
    //         seed: 0,
    //         shocks: ShockSchedule::stationary(),
    //     }
    // }

    // ══════════════════════════════════════════════════════════════════
    //  SDEs
    // ══════════════════════════════════════════════════════════════════

    pub fn ou() -> Self {
        Self {
            system: ODESystem::OU,
            params: SystemParams::OU(OuParams {
                theta: 0.2,
                mu: 0.0,
                sigma_noise: 0.3,
            }),
            dt: 0.5,
            steps: 25000,
            dim: 1,
            burn_in: 0,
            method: IntegrationMethod::Euler,
            x0: vec![0.0],
            seed: 0,
            shocks: ShockSchedule::stationary(),
        }
    }

    pub fn ou_param_shock() -> Self {
        let mut spec = Self::ou();
        spec.shocks = ShockSchedule::single(ShockEvent::params_replace(
            SystemParams::OU(OuParams {
                theta: 0.2,
                mu: 0.5,
                sigma_noise: 0.3,
            }),
            0.35,
        ));
        spec
    }

    pub fn doublewell() -> Self {
        Self {
            system: ODESystem::DoubleWell,
            params: SystemParams::DoubleWell(DoubleWellParams {
                a: 1.5,
                sigma_noise: 0.25,
            }),
            dt: 0.5,
            steps: 25000,
            dim: 1,
            burn_in: 0,
            method: IntegrationMethod::Euler,
            x0: vec![0.0],
            seed: 0,
            shocks: ShockSchedule::stationary(),
        }
    }

    pub fn doublewell_param_shock() -> Self {
        let mut spec = Self::doublewell();
        spec.shocks = ShockSchedule::single(ShockEvent::params_replace(
            SystemParams::DoubleWell(DoubleWellParams {
                a: 1.0,
                sigma_noise: 0.35,
            }),
            0.35,
        ));
        spec
    }

    /// Hard one-axis Double Well shift for the H64 paper expansion.
    ///
    /// The latent SDE is `dx = (a*x - x^3)dt + sigma_noise*dW`, with
    /// potential `U(x) = x^4/4 - a*x^2/2`. At 35% of the trajectory this
    /// changes only `a: 1.5 -> 0.5` while keeping `sigma_noise = 0.25`.
    /// The barrier height `a^2/4` falls from 0.5625 to 0.0625 and the
    /// Kramers-like exponent `2*barrier/sigma_noise^2` falls from 18 to 2.
    /// Observation-noise titration remains a separate downstream operation.
    pub fn doublewell_barrier_collapse() -> Self {
        let mut spec = Self::doublewell();
        spec.shocks = ShockSchedule::single(ShockEvent::params_replace(
            SystemParams::DoubleWell(DoubleWellParams {
                a: 0.5,
                sigma_noise: 0.25,
            }),
            0.35,
        ));
        spec
    }

    pub fn doublewell_switch() -> Self {
        let mut spec = Self::doublewell();
        spec.shocks = ShockSchedule::single(ShockEvent::trajectory_splice(
            vec![0.0],
            0.35,
            Some(SystemParams::DoubleWell(DoubleWellParams {
                a: 1.0,
                sigma_noise: 0.35,
            })),
        ));
        spec
    }

    /// Main-benchmark LG-SSM preset, mirroring lg_ssm_spec (serie/catalog.py):
    /// D = 8 latent dims (roadmap §2.5 main-benchmark state dimension 8–32),
    /// one damped oscillatory block per period from the §3.6 menu
    /// (12/24/48/96 steps), equal damping γ = -ln(0.95)/dt so ρ(A) = 0.95
    /// (§3.7 scenario 1) stays fixed while kappa_v varies, kappa_v = 1
    /// (normal drift), and q_c = 2γ standardizing the stationary covariance
    /// to the identity (§2.5).
    pub fn lg_ssm() -> Self {
        Self::lg_ssm_with_periods(&[12.0, 24.0, 48.0, 96.0])
    }

    /// Main-benchmark LG-SSM with the same known abrupt frequency switch as
    /// `serie.catalog.lg_ssm_switch_spec`: at 50% of the trajectory every
    /// angular frequency is multiplied by 1.5 while damping, q_c, kappa_v,
    /// dimension, and the injected innovation stream remain unchanged.
    pub fn lg_ssm_switch() -> Self {
        let mut spec = Self::lg_ssm();
        let params_after = match &spec.params {
            SystemParams::LgSsm(params) => {
                let mut changed = params.clone();
                changed.omegas.iter_mut().for_each(|omega| *omega *= 1.5);
                SystemParams::LgSsm(changed)
            }
            _ => unreachable!("lg_ssm preset owns LgSsm parameters"),
        };
        spec.shocks = ShockSchedule::single(ShockEvent::params_replace(params_after, 0.5));
        spec
    }

    /// Main-benchmark LG-SSM with a stronger, known multi-axis parameter
    /// regime switch, mirroring `serie.catalog.lg_ssm_regime_switch_spec`.
    ///
    /// At 50% of the trajectory, angular frequencies are multiplied by 1.5,
    /// the per-step spectral radius changes from 0.95 to 0.90, kappa_v changes
    /// from 1 to 5, and q_c becomes 2*gamma_after. The state remains continuous
    /// and the same innovation stream continues across the boundary. This is
    /// therefore a piecewise parameter-regime switch inside one exactly
    /// solvable linear-Gaussian family, not a splice of unrelated systems.
    pub fn lg_ssm_regime_switch() -> Self {
        let mut spec = Self::lg_ssm();
        let spectral_radius_after: f64 = 0.90;
        let gamma_after = -spectral_radius_after.ln() / spec.dt;
        let params_after = match &spec.params {
            SystemParams::LgSsm(params) => {
                let mut changed = params.clone();
                changed.gammas.fill(gamma_after);
                changed.omegas.iter_mut().for_each(|omega| *omega *= 1.5);
                changed.kappa_v = 5.0;
                changed.q_c = 2.0 * gamma_after;
                SystemParams::LgSsm(changed)
            }
            _ => unreachable!("lg_ssm preset owns LgSsm parameters"),
        };
        spec.shocks = ShockSchedule::single(ShockEvent::params_replace(params_after, 0.5));
        spec
    }

    /// Small/debug LG-SSM preset, mirroring lg_ssm_small_spec: D = 4
    /// (roadmap §2.5 small/debug state dimension 2–4), periods 12 and 24.
    pub fn lg_ssm_small() -> Self {
        Self::lg_ssm_with_periods(&[12.0, 24.0])
    }

    fn lg_ssm_with_periods(periods: &[f64]) -> Self {
        let dt = 1.0;
        let spectral_radius: f64 = 0.95;
        let gamma = -spectral_radius.ln() / dt;
        let dim = 2 * periods.len();
        Self {
            system: ODESystem::LgSsm,
            params: SystemParams::LgSsm(LgSsmParams {
                gammas: vec![gamma; periods.len()],
                omegas: periods.iter().map(|p| 2.0 * std::f64::consts::PI / (p * dt)).collect(),
                kappa_v: 1.0,
                q_c: 2.0 * gamma,
            }),
            dt,
            steps: 25000,
            dim,
            burn_in: 0,
            method: IntegrationMethod::Euler, // schema placeholder; stepping is exact
            x0: vec![0.0; dim],
            seed: 0,
            shocks: ShockSchedule::stationary(),
        }
    }

    // ══════════════════════════════════════════════════════════════════
    //  Discrete statistical
    // ══════════════════════════════════════════════════════════════════

    pub fn slds() -> Self {
        Self {
            system: ODESystem::SLDS,
            params: SystemParams::SLDS(SldsParams {
                a1: 0.9,
                q1: 0.05,
                a2: 0.98,
                q2: 0.35,
                p11: 0.94,
                p22: 0.95,
            }),
            dt: 1.0,
            steps: 25000,
            dim: 1,
            burn_in: 0,
            method: IntegrationMethod::Euler, // ignored for discrete
            x0: vec![0.0],
            seed: 0,
            shocks: ShockSchedule::stationary(),
        }
    }

    pub fn slds_param_shock() -> Self {
        let mut spec = Self::slds();
        spec.shocks = ShockSchedule::single(ShockEvent::params_replace(
            SystemParams::SLDS(SldsParams {
                a1: 0.83,
                q1: 0.50,
                a2: 0.97,
                q2: 0.30,
                p11: 0.96,
                p22: 0.92,
            }),
            0.35,
        ));
        spec
    }

    pub fn slds_switch() -> Self {
        let mut spec = Self::slds();
        spec.shocks = ShockSchedule::single(ShockEvent::trajectory_splice(
            vec![0.0],
            0.35,
            Some(SystemParams::SLDS(SldsParams {
                a1: 0.87,
                q1: 0.07,
                a2: 0.99,
                q2: 0.45,
                p11: 0.90,
                p22: 0.95,
            })),
        ));
        spec
    }

    pub fn seasonal_ar() -> Self {
        Self {
            system: ODESystem::SeasonalAR,
            params: SystemParams::SeasonalAR(SeasonalArParams {
                s: 24.0,
                phi: 0.5,
                sigma: 0.2,
                a0: 1.0,
                amp_drift: 0.0,
            }),
            dt: 1.0,
            steps: 25000,
            dim: 1,
            burn_in: 0,
            method: IntegrationMethod::Euler,
            x0: vec![0.0],
            seed: 0,
            shocks: ShockSchedule::stationary(),
        }
    }

    pub fn seasonal_ar_param_shock() -> Self {
        let mut spec = Self::seasonal_ar();
        spec.shocks = ShockSchedule::single(ShockEvent::params_replace(
            SystemParams::SeasonalAR(SeasonalArParams {
                s: 24.0,
                phi: 0.8,
                sigma: 0.35,
                a0: 1.4,
                amp_drift: 0.0,
            }),
            0.35,
        ));
        spec
    }

    pub fn garch() -> Self {
        // sigma2_0 = omega / (1 - alpha - beta) when alpha+beta < 1
        let omega = 0.01_f64;
        let alpha = 0.06_f64;
        let beta = 0.90_f64;
        let sigma2_0 = if alpha + beta < 1.0 {
            omega / (1.0 - alpha - beta).max(1e-9)
        } else {
            1.0
        };
        Self {
            system: ODESystem::GARCH,
            params: SystemParams::GARCH(GarchParams { omega, alpha, beta }),
            dt: 1.0,
            steps: 25000,
            dim: 2,
            burn_in: 0,
            method: IntegrationMethod::Euler,
            x0: vec![0.0, sigma2_0],
            seed: 0,
            shocks: ShockSchedule::stationary(),
        }
    }

    pub fn garch_param_shock() -> Self {
        let mut spec = Self::garch();
        spec.shocks = ShockSchedule::single(ShockEvent::params_replace(
            SystemParams::GARCH(GarchParams {
                omega: 0.03,
                alpha: 0.15,
                beta: 0.70,
            }),
            0.35,
        ));
        spec
    }

    // ══════════════════════════════════════════════════════════════════
    //  PDEs
    // ══════════════════════════════════════════════════════════════════

    pub fn ks() -> Self {
        let nx = 64;
        let lx = 22.0;
        // CHG (2026-07-01, user-approved scientific fix, mirrors serie
        // catalog.py ks_spec): x0 was zeros(nx), a fixed point of the KS flow
        // — the preset generated an identically zero series. Seed field is
        // the standard Kassam–Trefethen (2005) initial condition generalized
        // to domain length Lx:
        //     u0(x) = cos(2πx/Lx) · (1 + sin(2πx/Lx)),   x_j = j·Lx/nx
        // burn_in = 2500 (t = 25) discards the smooth-IC transient; steps
        // grew by the same 2500 so usable output stays 25001 rows (t ≈ 250).
        let x0 = (0..nx)
            .map(|j| {
                let x = j as f64 * (lx / nx as f64);
                let phase = 2.0 * std::f64::consts::PI * x / lx;
                phase.cos() * (1.0 + phase.sin())
            })
            .collect();
        Self {
            system: ODESystem::KS,
            params: SystemParams::KS(KsParams {
                nu: 1.0,
                lx,
                nx,
                dealias: true,
            }),
            dt: 1e-2,
            steps: 27500,
            dim: nx,
            burn_in: 2500,
            method: IntegrationMethod::ETDRK4,
            x0,
            seed: 0,
            shocks: ShockSchedule::stationary(),
        }
    }

    pub fn ks_param_shock() -> Self {
        let mut spec = Self::ks();
        spec.shocks = ShockSchedule::single(ShockEvent::params_replace(
            SystemParams::KS(KsParams {
                nx: 64,
                lx: 22.0,
                nu: 0.80,
                dealias: true,
            }),
            0.35,
        ));
        spec
    }

    // ══════════════════════════════════════════════════════════════════
    //  Designed / eigenvalue recovery
    // ══════════════════════════════════════════════════════════════════

    pub fn diagonal(eigenvalues: &[f64]) -> Self {
        let dim = eigenvalues.len();
        Self {
            system: ODESystem::Diagonal,
            params: SystemParams::Diagonal(DiagonalParams {
                eigenvalues: eigenvalues.to_vec(),
            }),
            dt: 0.01,
            steps: 10000,
            dim,
            burn_in: 0,
            method: IntegrationMethod::RK4,
            x0: vec![1.0; dim],
            seed: 0,
            shocks: ShockSchedule::stationary(),
        }
    }
}

/// Composed timeline of shocks. Empty = stationary baseline.
/// Events must be time-ordered (enforced in new()).

/// Timing shared by all shocks.
/// Times are FRACTIONS of usable T: 0.0 = start, 1.0 = end.
/// For parameter transitions, [t_start, t_end] is the transition window and
/// n_transition_updates says how many discrete regime updates to schedule in it.
/// n_transition_updates = 1 is the abrupt one-step case at t_start.
#[derive(Debug, Clone)]
pub struct ShockTiming {
    pub name: String,
    pub t_start: f64, // 0.0..=1.0
    pub t_end: f64,   // 0.0..=1.0
    pub n_transition_updates: usize,
}

impl ShockTiming {
    pub fn uses_single_change_step(&self) -> bool {
        self.n_transition_updates == 1 && self.t_start == self.t_end
    }
}

/// Each variant carries ONLY the fields it needs.
/// Python version: 14 fields on every ShockEvent, 10 unused per variant.
/// Rust version: each variant has 2-4 fields. No Option spam, no dead weight.
#[derive(Debug, Clone)]
pub enum ShockEvent {
    /// All parameters move toward a new parameter set across scheduled transition updates.
    ParamsReplace {
        timing: ShockTiming,
        params_after: SystemParams,
    },

    /// Additive kick: x[state_idx] += eps at t_start.
    StateKick {
        timing: ShockTiming,
        state_idx: usize,
        eps: f64,
    },

    /// Same-system trajectory splice: run A and B independently,
    /// glue B's trajectory onto A's at t_start.
    Splice {
        timing: ShockTiming,
        ic_b: Vec<f64>,
        params_after: Option<SystemParams>, // None = same params as A
    },
}

impl ShockEvent {
    // Every variant has timing — extract it uniformly
    pub fn timing(&self) -> &ShockTiming {
        match self {
            Self::ParamsReplace { timing, .. } => timing,
            Self::StateKick { timing, .. } => timing,
            Self::Splice { timing, .. } => timing,
        }
    }

    pub fn t_start(&self) -> f64 {
        self.timing().t_start
    }

    pub fn name(&self) -> &str {
        &self.timing().name
    }

    // ── Named constructors (replace @classmethod presets) ──

    pub fn params_replace(params_after: SystemParams, t_start: f64) -> Self {
        Self::ParamsReplace {
            timing: ShockTiming {
                name: format!("params_replace_at_{}", t_start),
                t_start,
                t_end: t_start,
                n_transition_updates: 1,
            },
            params_after,
        }
    }

    pub fn params_transition(
        params_after: SystemParams,
        t_start: f64,
        t_end: f64,
        n_transition_updates: usize,
    ) -> Self {
        Self::ParamsReplace {
            timing: ShockTiming {
                name: format!("params_transition_to_{}", t_end),
                t_start,
                t_end,
                n_transition_updates,
            },
            params_after,
        }
    }

    pub fn state_kick(state_idx: usize, eps: f64, t_start: f64) -> Self {
        Self::StateKick {
            timing: ShockTiming {
                name: format!("state_kick_idx{}_eps{}", state_idx, eps),
                t_start,
                t_end: t_start,
                n_transition_updates: 1,
            },
            state_idx,
            eps,
        }
    }

    pub fn trajectory_splice(ic_b: Vec<f64>, t_start: f64, params_after: Option<SystemParams>) -> Self {
        Self::Splice {
            timing: ShockTiming {
                name: format!("splice_at_{}", t_start),
                t_start,
                t_end: t_start,
                n_transition_updates: 1,
            },
            ic_b,
            params_after,
        }
    }
}

#[derive(Debug, Clone)]
pub struct ShockSchedule {
    events: Vec<ShockEvent>, // sorted by t_start, validated
}

impl ShockSchedule {
    pub fn new(mut events: Vec<ShockEvent>) -> Self {
        events.sort_by(|a, b| a.t_start().partial_cmp(&b.t_start()).unwrap());
        // Validate no overlap for same target
        for w in events.windows(2) {
            assert!(
                w[0].t_start() <= w[1].t_start(),
                "ShockSchedule: '{}' at t={} after '{}' at t={}",
                w[0].timing().name,
                w[0].t_start(),
                w[1].timing().name,
                w[1].t_start(),
            );
        }
        Self { events }
    }

    pub fn stationary() -> Self {
        Self { events: vec![] }
    }
    pub fn single(event: ShockEvent) -> Self {
        Self::new(vec![event])
    }

    pub fn is_stationary(&self) -> bool {
        self.events.is_empty()
    }
    pub fn n_shocks(&self) -> usize {
        self.events.len()
    }
    pub fn shock_times(&self) -> Vec<f64> {
        self.events.iter().map(|e| e.t_start()).collect()
    }
    pub fn events(&self) -> &[ShockEvent] {
        &self.events
    }
}

/*
A single designed perturbation on the timeline.
Consumer: SyntheticDataGenerator (applies shock during integration).
Times are in FRACTION of usable T (0.0 = start of usable data, 1.0 = end).
This makes shocks independent of absolute T and transient settings.
*/

/// Tests for the synthetic generation stage: trajectory specs, burn-in, and shocks.
///
/// Why these tests exist:
/// - Burn-in must shorten the output by exactly burn_in rows — the transient
///   approach to the attractor is discarded so the training data starts on (or
///   near) the attractor. Wrong burn-in math means training on transient data.
/// - Feature names must match the system dimension (f0, f1, f2 for 3D) because
///   downstream stages use feature names for column selection and reporting.
/// - Seed determinism: same TrajectorySpec + same seed → identical trajectory.
///   Without this, experiment reruns are not reproducible.
/// - Different seeds must produce different trajectories for stochastic systems
///   (OU, DoubleWell) — if they don't, the "multiple draws" in a DGP bank are
///   all identical and the noise titration has no statistical power.
/// - The full-pipeline integration test verifies that synthetic → noise →
///   partition → ready composes without panics.
#[cfg(test)]
mod array_core_tests {
    use super::*;

    #[test]
    fn timestep_major_core_applies_burn_in_and_names_channels() {
        let mut config = TrajectorySpec::lorenz();
        config.steps = 8;
        config.burn_in = 2;

        let trajectory = generate_timestep_major(config).unwrap();

        assert_eq!(trajectory.timesteps, 7);
        assert_eq!(trajectory.channels, 3);
        assert_eq!(trajectory.timestep_major_values.len(), 21);
        assert_eq!(trajectory.feature_names, ["f0", "f1", "f2"]);
    }
}

#[cfg(all(test, feature = "pipeline"))]
mod tests {
    use super::*;
    #[cfg(feature = "pipeline")]
    use crate::stage_noise::NoiseConfig;
    #[cfg(feature = "pipeline")]
    use crate::stage_partition::PartitionConfig;
    #[cfg(feature = "pipeline")]
    use crate::stage_ready::ReadyConfig;

    #[test]
    fn generate_lorenz_bundle_shape_and_names() {
        let mut config = TrajectorySpec::lorenz();
        config.steps = 8;

        let bundle = generate(config).unwrap();

        assert_eq!(bundle.channels, 3);
        assert_eq!(bundle.frame.height(), 9);
        assert_eq!(
            bundle
                .feature_names
                .iter()
                .map(|name| name.as_str())
                .collect::<Vec<_>>(),
            vec!["f0", "f1", "f2"]
        );

        let f0 = bundle.frame.column("f0").unwrap().numeric().unwrap();
        let f1 = bundle.frame.column("f1").unwrap().numeric().unwrap();
        let f2 = bundle.frame.column("f2").unwrap().numeric().unwrap();
        assert_eq!(f0.get(0).unwrap(), 1.0);
        assert_eq!(f1.get(0).unwrap(), 0.98);
        assert_eq!(f2.get(0).unwrap(), 1.1);
    }

    #[test]
    fn generate_applies_burn_in_to_output_rows() {
        let mut config = TrajectorySpec::ou();
        config.steps = 10;
        config.burn_in = 3;

        let bundle = generate(config).unwrap();

        assert_eq!(bundle.channels, 1);
        assert_eq!(bundle.frame.height(), 8); // steps + 1 - burn_in
    }

    #[test]
    fn generate_rejects_multi_event_schedule_for_now() {
        let mut config = TrajectorySpec::lorenz();
        config.steps = 16;
        config.shocks = ShockSchedule::new(vec![
            ShockEvent::state_kick(0, 0.5, 0.2),
            ShockEvent::state_kick(1, 0.25, 0.4),
        ]);

        let err = generate(config).unwrap_err();
        assert!(err.to_string().contains("multi-event"));
    }

    #[test]
    fn compile_native_run_plan_expands_params_replace_into_scheduled_regimes() {
        let mut config = TrajectorySpec::lorenz();
        config.steps = 20;
        config.shocks = ShockSchedule::single(ShockEvent::ParamsReplace {
            timing: ShockTiming {
                name: "rho_replace_transition".to_string(),
                t_start: 0.25,
                t_end: 0.5,
                n_transition_updates: 2,
            },
            params_after: SystemParams::Lorenz(LorenzParams {
                sigma: 10.0,
                rho: 35.0,
                beta: 8.0 / 3.0,
            }),
        });

        let plan = s01_compile_native_run_plan(&config).unwrap();
        match plan {
            NativeRunPlan::OneRun { param_sets, shocks } => {
                assert_eq!(param_sets.len(), 3);
                assert_eq!(shocks.len(), 2);

                match &param_sets[1] {
                    SystemParams::Lorenz(params) => {
                        assert!((params.rho - 31.5).abs() < 1e-12);
                    }
                    other => panic!("expected Lorenz regime, got {:?}", other),
                }

                match &param_sets[2] {
                    SystemParams::Lorenz(params) => {
                        assert!((params.rho - 35.0).abs() < 1e-12);
                    }
                    other => panic!("expected Lorenz regime, got {:?}", other),
                }

                let drift_steps = shocks
                    .iter()
                    .map(|shock| match shock {
                        NativeShockEvent::ParamChange { step, .. } => *step,
                        other => panic!("expected ParamChange shock, got {:?}", other),
                    })
                    .collect::<Vec<_>>();
                assert_eq!(drift_steps, vec![7, 10]);
            }
            other => panic!("expected OneRun plan, got {:?}", other),
        }
    }

    #[test]
    fn compile_native_run_plan_treats_one_transition_update_as_the_abrupt_case() {
        let mut config = TrajectorySpec::lorenz();
        config.steps = 20;
        config.shocks = ShockSchedule::single(ShockEvent::ParamsReplace {
            timing: ShockTiming {
                name: "one_step_param_transition".to_string(),
                t_start: 0.25,
                t_end: 0.5,
                n_transition_updates: 1,
            },
            params_after: SystemParams::Lorenz(LorenzParams {
                sigma: 10.0,
                rho: 35.0,
                beta: 8.0 / 3.0,
            }),
        });

        let plan = s01_compile_native_run_plan(&config).unwrap();
        match plan {
            NativeRunPlan::OneRun { param_sets, shocks } => {
                assert_eq!(param_sets.len(), 2);
                assert_eq!(shocks.len(), 1);

                match &param_sets[1] {
                    SystemParams::Lorenz(params) => {
                        assert!((params.rho - 35.0).abs() < 1e-12);
                    }
                    other => panic!("expected Lorenz regime, got {:?}", other),
                }

                let change_steps = shocks
                    .iter()
                    .map(|shock| match shock {
                        NativeShockEvent::ParamChange { step, .. } => *step,
                        other => panic!("expected ParamChange shock, got {:?}", other),
                    })
                    .collect::<Vec<_>>();
                assert_eq!(change_steps, vec![5]);
            }
            other => panic!("expected OneRun plan, got {:?}", other),
        }
    }

    #[test]
    fn windowed_seasonal_ar_transition_keeps_period_fixed_mid_window() {
        // Structural/integer params never interpolate: a windowed transition
        // that names a different S must leave S at its before-value in every
        // scheduled update (Python _transition_field_names excludes S).
        let mut config = TrajectorySpec::seasonal_ar();
        config.steps = 20;
        config.shocks = ShockSchedule::single(ShockEvent::ParamsReplace {
            timing: ShockTiming {
                name: "seasonal_ar_windowed_transition".to_string(),
                t_start: 0.25,
                t_end: 0.75,
                n_transition_updates: 2,
            },
            params_after: SystemParams::SeasonalAR(SeasonalArParams {
                s: 48.0, // structural: must NOT move
                phi: 0.8,
                sigma: 0.35,
                a0: 1.4,
                amp_drift: 0.0,
            }),
        });

        let plan = s01_compile_native_run_plan(&config).unwrap();
        match plan {
            NativeRunPlan::OneRun { param_sets, .. } => {
                assert_eq!(param_sets.len(), 3);
                match (&param_sets[1], &param_sets[2]) {
                    (SystemParams::SeasonalAR(mid), SystemParams::SeasonalAR(landed)) => {
                        assert_eq!(mid.s, 24.0, "mid-window S must stay at the before-value");
                        assert_eq!(mid.phi, 0.5 * 0.5 + 0.8 * 0.5); // α = 0.5 lerp
                        assert_eq!(landed.s, 24.0, "windowed transitions never move S");
                        assert_eq!(landed.phi, 0.8);
                    }
                    other => panic!("expected SeasonalAR regimes, got {:?}", other),
                }
            }
            other => panic!("expected OneRun plan, got {:?}", other),
        }
    }

    #[test]
    fn compile_native_run_plan_accepts_abrupt_params_replace_at_single_step() {
        // Regression: t_start == t_end with one transition update is the
        // abrupt case (ShockTiming: "n_transition_updates = 1 is the abrupt
        // one-step case at t_start"); the window-gap check used to reject it
        // for every *_param_shock preset.
        let mut config = TrajectorySpec::lorenz_param_shock();
        config.steps = 40;
        let params_after = match &config.shocks.events()[0] {
            ShockEvent::ParamsReplace { params_after, .. } => params_after.clone(),
            other => panic!("expected ParamsReplace event, got {:?}", other),
        };

        let plan = s01_compile_native_run_plan(&config).unwrap();
        match plan {
            NativeRunPlan::OneRun { param_sets, shocks } => {
                assert_eq!(param_sets.len(), 2);
                match (&param_sets[1], &params_after) {
                    (SystemParams::Lorenz(got), SystemParams::Lorenz(want)) => {
                        // Lands on params_after exactly (Python parity: the
                        // instantaneous branch uses event.params_after as-is).
                        assert_eq!(got.sigma, want.sigma);
                        assert_eq!(got.rho, want.rho);
                        assert_eq!(got.beta, want.beta);
                    }
                    other => panic!("expected Lorenz regimes, got {:?}", other),
                }
                assert_eq!(shocks.len(), 1);
                match &shocks[0] {
                    NativeShockEvent::ParamChange { step, param_set_idx } => {
                        assert_eq!(*step, (0.35_f64 * 40.0) as usize); // 14
                        assert_eq!(*param_set_idx, 1);
                    }
                    other => panic!("expected ParamChange shock, got {:?}", other),
                }
            }
            other => panic!("expected OneRun plan, got {:?}", other),
        }
    }

    #[cfg(feature = "pipeline")]
    #[test]
    fn generate_ready_bundle_runs_full_synthetic_pipeline() {
        let mut config = TrajectorySpec::lorenz();
        config.steps = 59;

        let noise_config = NoiseConfig {
            skip_noise: true,
            ..NoiseConfig::new(0.0, 7)
        };
        let ready_bundle = generate_ready_bundle(
            config,
            &noise_config,
            PartitionConfig::p70_10_20(),
            ReadyConfig::new(4, 2),
        )
        .unwrap();

        assert_eq!(ready_bundle.channels, 3);
        assert_eq!(ready_bundle.seq_len, 4);
        assert_eq!(ready_bundle.pred_len, 2);
        assert!(ready_bundle.train.n_windows > 0);
        assert!(ready_bundle.val.n_windows > 0);
        assert!(ready_bundle.test.n_windows > 0);
    }

    // ── Gap tests (mirroring Python test_contracts.py) ────────────────

    #[test]
    fn burn_in_shortens_output() {
        let mut config = TrajectorySpec::lorenz();
        config.steps = 200;
        config.burn_in = 50;
        let bundle = generate(config).unwrap();
        // output = steps - burn_in + 1 = 200 - 50 + 1 = 151
        assert_eq!(bundle.frame.height(), 151);
    }

    #[test]
    fn burn_in_zero_gives_steps_plus_one() {
        let mut config = TrajectorySpec::lorenz();
        config.steps = 100;
        config.burn_in = 0;
        let bundle = generate(config).unwrap();
        assert_eq!(bundle.frame.height(), 101);
    }

    #[test]
    fn burn_in_greater_than_steps_is_rejected() {
        let mut config = TrajectorySpec::lorenz();
        config.steps = 100;
        config.burn_in = 100; // must be < steps
        assert!(generate(config).is_err());
    }

    #[test]
    fn feature_names_are_f0_f1_f2_for_3d_systems() {
        let mut config = TrajectorySpec::lorenz();
        config.steps = 10;
        let bundle = generate(config).unwrap();
        let names: Vec<&str> = bundle.feature_names.iter().map(|n| n.as_str()).collect();
        assert_eq!(names, vec!["f0", "f1", "f2"]);
    }

    #[test]
    fn feature_names_are_f0_for_1d_systems() {
        let mut config = TrajectorySpec::ou();
        config.steps = 10;
        let bundle = generate(config).unwrap();
        let names: Vec<&str> = bundle.feature_names.iter().map(|n| n.as_str()).collect();
        assert_eq!(names, vec!["f0"]);
    }

    #[test]
    fn same_seed_produces_identical_trajectories() {
        let mut config = TrajectorySpec::lorenz();
        config.steps = 50;
        let a = generate(config.clone()).unwrap();
        let b = generate(config).unwrap();
        // Deterministic system: same config → bit-identical output
        let a_vals: Vec<f64> = a
            .frame
            .column("f0")
            .unwrap()
            .numeric()
            .unwrap()
            .into_iter()
            .filter_map(|v| v)
            .collect();
        let b_vals: Vec<f64> = b
            .frame
            .column("f0")
            .unwrap()
            .numeric()
            .unwrap()
            .into_iter()
            .filter_map(|v| v)
            .collect();
        assert_eq!(a_vals, b_vals);
    }

    #[test]
    fn initial_condition_is_preserved_in_output() {
        let mut config = TrajectorySpec::lorenz();
        config.steps = 10;
        let bundle = generate(config).unwrap();
        let f0 = bundle.frame.column("f0").unwrap().numeric().unwrap();
        let f1 = bundle.frame.column("f1").unwrap().numeric().unwrap();
        let f2 = bundle.frame.column("f2").unwrap().numeric().unwrap();
        // Lorenz default IC: (1.0, 0.98, 1.1)
        assert!((f0.get(0).unwrap() - 1.0).abs() < 1e-10);
        assert!((f1.get(0).unwrap() - 0.98).abs() < 1e-10);
        assert!((f2.get(0).unwrap() - 1.1).abs() < 1e-10);
    }

    // ── Discrete statistical systems (SLDS, Seasonal AR, GARCH) ──────

    fn column_values(bundle: &SyntheticBundle, name: &str) -> Vec<f64> {
        bundle
            .frame
            .column(name)
            .unwrap()
            .numeric()
            .unwrap()
            .into_iter()
            .flatten()
            .collect()
    }

    #[test]
    fn slds_generate_applies_burn_in_and_single_channel() {
        let mut config = TrajectorySpec::slds();
        config.steps = 10;
        config.burn_in = 3;
        let bundle = generate(config).unwrap();
        assert_eq!(bundle.channels, 1);
        assert_eq!(bundle.frame.height(), 8); // steps + 1 - burn_in
        let names: Vec<&str> = bundle.feature_names.iter().map(|n| n.as_str()).collect();
        assert_eq!(names, vec!["f0"]);
    }

    #[test]
    fn garch_generate_emits_return_and_variance_channels() {
        let mut config = TrajectorySpec::garch();
        config.steps = 10;
        let bundle = generate(config).unwrap();
        assert_eq!(bundle.channels, 2);
        assert_eq!(bundle.frame.height(), 11);
        let names: Vec<&str> = bundle.feature_names.iter().map(|n| n.as_str()).collect();
        assert_eq!(names, vec!["f0", "f1"]);
        // Row 0 carries the preset x0: r_0 = 0, σ²_0 = ω/(1−α−β) = 0.25.
        let f0 = column_values(&bundle, "f0");
        let f1 = column_values(&bundle, "f1");
        assert_eq!(f0[0], 0.0);
        assert!((f1[0] - 0.25).abs() < 1e-15);
    }

    #[test]
    fn lg_ssm_presets_generate_expected_dims_and_finite_values() {
        // Main preset: D = 8 (roadmap §2.5 main benchmark); small: D = 4.
        for (spec, expected_dim) in [
            (TrajectorySpec::lg_ssm(), 8),
            (TrajectorySpec::lg_ssm_switch(), 8),
            (TrajectorySpec::lg_ssm_regime_switch(), 8),
            (TrajectorySpec::lg_ssm_small(), 4),
        ] {
            let mut config = spec;
            config.steps = 300;
            let bundle = generate(config).unwrap();
            assert_eq!(bundle.channels, expected_dim);
            assert_eq!(bundle.frame.height(), 301);
            for d in 0..expected_dim {
                let values = column_values(&bundle, &format!("f{d}"));
                assert!(values.iter().all(|v| v.is_finite()));
            }
            let f0 = column_values(&bundle, "f0");
            assert_eq!(f0[0], 0.0, "preset x0 is the origin");
            assert!(f0.iter().any(|v| v.abs() > 0.1), "process noise must excite the state");
        }
    }

    #[test]
    fn discrete_systems_same_seed_reproduces_and_different_seed_differs() {
        for spec in [
            TrajectorySpec::slds(),
            TrajectorySpec::seasonal_ar(),
            TrajectorySpec::garch(),
            TrajectorySpec::lg_ssm_small(),
        ] {
            let mut config_a = spec.clone();
            config_a.steps = 100;
            config_a.seed = 7;
            let config_b = config_a.clone();
            let mut config_c = config_a.clone();
            config_c.seed = 8;

            let a = generate(config_a).unwrap();
            let b = generate(config_b).unwrap();
            let c = generate(config_c).unwrap();
            let (a, b, c) = (
                column_values(&a, "f0"),
                column_values(&b, "f0"),
                column_values(&c, "f0"),
            );
            assert_eq!(a, b, "{:?}: same seed must reproduce identical output", spec.system);
            assert_ne!(a, c, "{:?}: different seed must change the output", spec.system);
        }
    }

    #[test]
    fn discrete_param_shock_presets_generate() {
        for spec in [
            TrajectorySpec::slds_param_shock(),
            TrajectorySpec::seasonal_ar_param_shock(),
            TrajectorySpec::garch_param_shock(),
        ] {
            let mut config = spec;
            config.steps = 40;
            let bundle = generate(config).unwrap();
            assert_eq!(bundle.frame.height(), 41);
        }
    }

    #[test]
    fn doublewell_barrier_collapse_changes_only_a_at_the_frozen_boundary() {
        let config = TrajectorySpec::doublewell_barrier_collapse();
        match &config.params {
            SystemParams::DoubleWell(params) => {
                assert_eq!(params.a, 1.5);
                assert_eq!(params.sigma_noise, 0.25);
            }
            other => panic!("expected DoubleWell base parameters, got {other:?}"),
        }
        match &config.shocks.events()[0] {
            ShockEvent::ParamsReplace { timing, params_after } => {
                assert_eq!(timing.t_start, 0.35);
                match params_after {
                    SystemParams::DoubleWell(params) => {
                        assert_eq!(params.a, 0.5);
                        assert_eq!(params.sigma_noise, 0.25);
                    }
                    other => panic!("expected DoubleWell shifted parameters, got {other:?}"),
                }
            }
            other => panic!("expected ParamsReplace event, got {other:?}"),
        }
    }

    #[test]
    fn slds_switch_splices_at_computed_step_with_seed_plus_one() {
        let mut config = TrajectorySpec::slds_switch();
        config.steps = 20;
        config.seed = 5;
        // Same step arithmetic as s01_compile_native_run_plan:
        // burn_in + trunc(t_start · usable_steps). The f64 product 0.35·20
        // rounds to exactly 7.0 (Python int(0.35*20) is likewise 7).
        let splice_step = (0.35_f64 * 20.0) as usize;
        assert_eq!(splice_step, 7);

        let params_a = config.params.as_f64_vec();
        let params_b = match &config.shocks.events()[0] {
            ShockEvent::Splice { params_after, .. } => params_after.clone().unwrap().as_f64_vec(),
            other => panic!("expected Splice event, got {:?}", other),
        };

        let bundle = generate(config).unwrap();
        let switched = column_values(&bundle, "f0");

        let traj_a =
            stage_generate::generate_discrete_seeded("slds", &[0.0], 20, 1.0, 5, &[], &params_a, params_a.len())
                .unwrap();
        let traj_b =
            stage_generate::generate_discrete_seeded("slds", &[0.0], 20, 1.0, 6, &[], &params_b, params_b.len())
                .unwrap();

        assert_eq!(&switched[..=splice_step], &traj_a[..=splice_step]);
        assert_eq!(&switched[splice_step + 1..], &traj_b[splice_step + 1..]);
    }

    #[test]
    fn doublewell_switch_splices_at_computed_step_with_seed_plus_one() {
        let mut config = TrajectorySpec::doublewell_switch();
        config.steps = 20;
        config.seed = 5;
        let splice_step = (0.35_f64 * 20.0) as usize; // 7, as above

        let params_a = config.params.as_f64_vec();
        let params_b = match &config.shocks.events()[0] {
            ShockEvent::Splice { params_after, .. } => params_after.clone().unwrap().as_f64_vec(),
            other => panic!("expected Splice event, got {:?}", other),
        };

        let bundle = generate(config).unwrap();
        let switched = column_values(&bundle, "f0");

        // A runs with (a=1.5, σ=0.25, seed), B with (a=1.0, σ=0.35, seed+1).
        let traj_a =
            stage_generate::generate_sde_seeded("doublewell", &[0.0], 20, 0.5, &[0.25], 5, &[], &params_a, 2).unwrap();
        let traj_b =
            stage_generate::generate_sde_seeded("doublewell", &[0.0], 20, 0.5, &[0.35], 6, &[], &params_b, 2).unwrap();

        assert_eq!(&switched[..=splice_step], &traj_a[..=splice_step]);
        assert_eq!(&switched[splice_step + 1..], &traj_b[splice_step + 1..]);
    }

    #[test]
    fn ou_different_seeds_produce_different_trajectories() {
        let mut config_a = TrajectorySpec::ou();
        config_a.steps = 100;
        config_a.seed = 7;
        let mut config_b = TrajectorySpec::ou();
        config_b.steps = 100;
        config_b.seed = 8;
        let a = generate(config_a).unwrap();
        let b = generate(config_b).unwrap();
        let a_vals: Vec<f64> = a
            .frame
            .column("f0")
            .unwrap()
            .numeric()
            .unwrap()
            .into_iter()
            .filter_map(|v| v)
            .collect();
        let b_vals: Vec<f64> = b
            .frame
            .column("f0")
            .unwrap()
            .numeric()
            .unwrap()
            .into_iter()
            .filter_map(|v| v)
            .collect();
        // SDE: different seeds → different trajectories
        assert_ne!(a_vals, b_vals);
    }
}
