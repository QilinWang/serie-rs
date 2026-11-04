//! `serie-rs` — the standalone CLI for the serie data layer.
//!
//! Mirrors the Python `serie` command surface for the Rust engine:
//!
//! ```text
//! serie-rs list-systems        list the synthetic systems you can build
//! serie-rs generate <system>   generate a DGP bank (clean + noise [+ noisy]) → files
//! serie-rs demo <system>       bake attractor_data.json for the web viewer
//! serie-rs diagnose <system>   partition statistics as a text table or a TUI panel
//! ```
//!
//! This is the binary; the reusable engine is the `serie_rs::stage_*` library.
//! `generate` writes the same grouped `dgp_bank.safetensors` that the Python
//! `serie` package writes (and that `serie_rs::dgp_bank` reads back).

use std::error::Error;
use std::path::PathBuf;

use clap::{Args, Parser, Subcommand, ValueEnum};
use ndarray::{ArrayD, IxDyn};

use serie_rs::dgp_bank::{NoiseKind, unit_noise_draws, write_dgp_bank_safetensors};
use serie_rs::stage_diag_cli::{DiagnoseConfig, run_diagnostics};
use serie_rs::stage_noise::{NoiseConfig, inject_noise};
use serie_rs::stage_partition::{PartitionConfig, partition};
use serie_rs::stage_synthetic::{TrajectorySpec, generate};

type Fallible<T> = Result<T, Box<dyn Error + Send + Sync>>;

/// Base synthetic systems with a constructable preset.
const SYSTEMS: &[&str] = &[
    "lorenz",
    "rossler",
    "chua",
    "lorenz96",
    "ou",
    "doublewell",
    "slds",
    "seasonal_ar",
    "garch",
];
/// The three 3-D attractors the `demo shapes` bundle shows.
const DEMO_SHAPES: &[&str] = &["lorenz", "rossler", "chua"];

// ── CLI surface ─────────────────────────────────────────────────────────────

#[derive(Parser)]
#[command(
    name = "serie-rs",
    about = "Rust engine for the serie time-series robustness lab",
    version
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// List the synthetic systems you can build.
    ListSystems,
    /// Generate a DGP bank (clean + noise [+ noisy]) and write it to files.
    Generate(Generate),
    /// Bake attractor_data.json for the interactive web viewer.
    Demo(Demo),
    /// Show partition statistics as a text table.
    Diagnose(Diagnose),
}

/// Intervention preset applied to the clean source before noise.
#[derive(Copy, Clone, Debug, ValueEnum)]
enum Shock {
    None,
    Param,
    State,
    Switch,
}

#[derive(Args)]
struct Generate {
    /// System name (see `serie-rs list-systems`).
    system: String,
    /// Intervention preset to apply before noise.
    #[arg(long, value_enum, default_value = "none")]
    shock: Shock,
    /// Observation-noise dose. 0.0 = clean (no `noisy_sigma_*` tensor written).
    #[arg(long, default_value_t = 0.0)]
    sigma: f64,
    /// Root seed for the trajectory.
    #[arg(long, default_value_t = 42)]
    seed: u64,
    /// Root seed for the frozen noise draws.
    #[arg(long, default_value_t = 1000)]
    noise_seed: u64,
    /// Number of frozen unit-noise draws to bank.
    #[arg(long, default_value_t = 8)]
    n_noise_draws: usize,
    /// Output directory. Writes `dgp_bank.safetensors` + `manifest.json`.
    /// Without it, only the JSON summary is printed.
    #[arg(long)]
    out: Option<PathBuf>,
}

#[derive(Args)]
struct Demo {
    /// System name, or "shapes" for the lorenz+rossler+chua bundle.
    #[arg(default_value = "shapes")]
    system: String,
    /// Root seed for the trajectory.
    #[arg(long, default_value_t = 42)]
    seed: u64,
    /// Root seed for the frozen noise draws.
    #[arg(long, default_value_t = 1000)]
    noise_seed: u64,
    /// Frozen unit-noise draws the slider can pick.
    #[arg(long, default_value_t = 30)]
    n_draws: usize,
    /// Subsample each trajectory to this many points (keeps the JSON small).
    #[arg(long, default_value_t = 4000)]
    n_points: usize,
    /// Output JSON path.
    #[arg(long, default_value = "attractor_data.json")]
    out: PathBuf,
}

#[derive(Args)]
struct Diagnose {
    /// System name (see `serie-rs list-systems`).
    system: String,
    /// Intervention preset to apply before noise.
    #[arg(long, value_enum, default_value = "none")]
    shock: Shock,
    /// Observation-noise dose.
    #[arg(long, default_value_t = 0.0)]
    sigma: f64,
    /// Root seed for the trajectory.
    #[arg(long, default_value_t = 42)]
    seed: u64,
    /// Root seed for the noise draw.
    #[arg(long, default_value_t = 1000)]
    noise_seed: u64,
    /// Train fraction of the split.
    #[arg(long, default_value_t = 0.7)]
    train_frac: f64,
    /// Validation fraction of the split (test gets the remainder).
    #[arg(long, default_value_t = 0.2)]
    val_frac: f64,
}

fn main() {
    if let Err(e) = run(Cli::parse()) {
        eprintln!("serie-rs: {e}");
        std::process::exit(1);
    }
}

fn run(cli: Cli) -> Fallible<()> {
    match cli.command {
        Command::ListSystems => {
            run_list_systems();
            Ok(())
        }
        Command::Generate(args) => run_generate(args),
        Command::Demo(args) => run_demo(args),
        Command::Diagnose(args) => run_diagnose(args),
    }
}

// ── commands ────────────────────────────────────────────────────────────────

fn run_list_systems() {
    println!("Synthetic systems (serie-rs):");
    for system in SYSTEMS {
        println!("  {system}");
    }
    println!("\nShock variants via --shock: none|param|state|switch (availability varies by system).");
    println!("Demo bundle: `serie-rs demo shapes` bakes lorenz+rossler+chua together.");
}

fn run_generate(args: Generate) -> Fallible<()> {
    let mut spec = spec_for(&args.system, args.shock)?;
    spec.seed = args.seed;
    let bundle = generate(spec)?;
    let feature_names: Vec<String> = bundle.feature_names.iter().map(|name| name.to_string()).collect();
    let timesteps = bundle.frame.height();
    let channels = bundle.frame.width();
    let clean_flat: Vec<f32> = bundle
        .frame
        .row_major_f64()
        .into_iter()
        .map(|value| value as f32)
        .collect();

    let clean = ArrayD::from_shape_vec(IxDyn(&[1, timesteps, channels]), clean_flat.clone())?;
    let noise_flat = unit_noise_draws(
        NoiseKind::Gaussian,
        args.noise_seed,
        args.n_noise_draws,
        timesteps,
        channels,
    )?;
    let noise = ArrayD::from_shape_vec(IxDyn(&[args.n_noise_draws, timesteps, channels]), noise_flat.clone())?;

    // Optional materialized cell: noisy = clean[0] + sigma * noise[0].
    let mut noisy: Vec<(String, ArrayD<f32>)> = Vec::new();
    if args.sigma > 0.0 && args.n_noise_draws > 0 {
        let sigma = args.sigma as f32;
        let cell: Vec<f32> = (0..timesteps * channels)
            .map(|i| clean_flat[i] + sigma * noise_flat[i])
            .collect();
        let name = format!("noisy_sigma_{}", format_sigma(args.sigma));
        noisy.push((name, ArrayD::from_shape_vec(IxDyn(&[timesteps, channels]), cell)?));
    }

    let summary = serde_json::json!({
        "system": args.system,
        "shock": format!("{:?}", args.shock).to_lowercase(),
        "seed": args.seed,
        "noise_seed": args.noise_seed,
        "sigma": args.sigma,
        "timesteps": timesteps,
        "channels": channels,
        "n_noise_draws": args.n_noise_draws,
        "feature_names": feature_names,
    });

    if let Some(dir) = &args.out {
        std::fs::create_dir_all(dir)?;
        let extra = vec![
            ("system".to_string(), args.system.clone()),
            ("seed".to_string(), args.seed.to_string()),
            ("noise_seed".to_string(), args.noise_seed.to_string()),
        ];
        write_dgp_bank_safetensors(
            &dir.join("dgp_bank.safetensors"),
            &clean,
            &noise,
            &noisy,
            &[],
            &feature_names,
            &extra,
        )?;

        let manifest = serde_json::json!({
            "schema": "serie.dgp_bank_safetensors.v1",
            "summary": summary,
            "files": ["dgp_bank.safetensors", "manifest.json"],
        });
        std::fs::write(
            dir.join("manifest.json"),
            serde_json::to_string_pretty(&manifest)? + "\n",
        )?;
        println!("wrote {} (dgp_bank.safetensors + manifest.json)", dir.display());
    }

    println!("{}", serde_json::to_string_pretty(&summary)?);
    Ok(())
}

fn run_demo(args: Demo) -> Fallible<()> {
    let systems: Vec<String> = if args.system == "shapes" {
        DEMO_SHAPES.iter().map(|name| name.to_string()).collect()
    } else {
        vec![args.system.clone()]
    };

    let baked: Vec<serde_json::Value> = systems
        .iter()
        .map(|system| bake_one(system, args.seed, args.noise_seed, args.n_draws, args.n_points))
        .collect::<Fallible<_>>()?;

    let payload = if baked.len() == 1 {
        baked.into_iter().next().expect("one system baked")
    } else {
        serde_json::json!({
            "kind": "serie_demo_bundle",
            "default_system": systems[0],
            "systems": baked,
        })
    };

    if let Some(parent) = args.out.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)?;
        }
    }
    std::fs::write(&args.out, serde_json::to_string(&payload)?)?;
    println!("baked {} -> {}", systems.join(", "), args.out.display());
    Ok(())
}

fn run_diagnose(args: Diagnose) -> Fallible<()> {
    let mut spec = spec_for(&args.system, args.shock)?;
    spec.seed = args.seed;
    let bundle = generate(spec)?;

    let noisy = inject_noise(bundle, &NoiseConfig::new(args.sigma, args.noise_seed))?;

    let train_end = args.train_frac;
    let val_end = args.train_frac + args.val_frac;
    let partitioned = partition(
        noisy,
        PartitionConfig::new(0.0, train_end, train_end, val_end, val_end, 1.0),
    )?;

    run_diagnostics(
        &partitioned.train_stats,
        &partitioned.val_stats,
        &partitioned.test_stats,
        DiagnoseConfig::default(),
    );
    Ok(())
}

// ── one baked demo system (mirrors serie/demo/bake_demo_data.py) ─────────────

fn bake_one(system: &str, seed: u64, noise_seed: u64, n_draws: usize, n_points: usize) -> Fallible<serde_json::Value> {
    let mut spec = spec_for(system, Shock::None)?;
    spec.seed = seed;
    let dt = spec.dt;
    let bundle = generate(spec)?;
    let feature_names: Vec<String> = bundle.feature_names.iter().map(|name| name.to_string()).collect();
    let timesteps = bundle.frame.height();
    let channels = bundle.frame.width();
    let clean_flat: Vec<f32> = bundle
        .frame
        .row_major_f64()
        .into_iter()
        .map(|value| value as f32)
        .collect();
    let noise_flat = unit_noise_draws(NoiseKind::Gaussian, noise_seed, n_draws, timesteps, channels)?;

    // Subsample timestep indices down to n_points (matches numpy linspace indices).
    let idx: Vec<usize> = if timesteps > n_points && n_points > 1 {
        (0..n_points).map(|i| i * (timesteps - 1) / (n_points - 1)).collect()
    } else {
        (0..timesteps).collect()
    };

    let (clean3, draws3, names3, projection) = project_to_3d(
        &clean_flat,
        &noise_flat,
        &idx,
        timesteps,
        channels,
        n_draws,
        &feature_names,
    );
    let points = clean3.len();

    Ok(serde_json::json!({
        "system": system,
        "seed": seed,
        "source_dim": channels,
        "projection": projection,
        "feature_names": names3,
        "dt": dt,
        "n_points": points,
        "shape": { "clean": [points, 3], "draws": [n_draws, points, 3] },
        "clean": clean3,
        "draws": draws3,
    }))
}

/// Project a clean trajectory + its unit-noise draws into the 3-D points the
/// viewer renders, mirroring `bake_demo_data.py::_project_arrays`.
fn project_to_3d(
    clean_flat: &[f32],
    noise_flat: &[f32],
    idx: &[usize],
    timesteps: usize,
    channels: usize,
    n_draws: usize,
    names: &[String],
) -> (Vec<Vec<f32>>, Vec<Vec<Vec<f32>>>, Vec<String>, &'static str) {
    let clean_at = |ti: usize, c: usize| clean_flat[ti * channels + c];
    let draw_at = |draw: usize, ti: usize, c: usize| noise_flat[draw * timesteps * channels + ti * channels + c];
    let points = idx.len();

    if channels >= 3 {
        let clean3 = idx
            .iter()
            .map(|&ti| vec![clean_at(ti, 0), clean_at(ti, 1), clean_at(ti, 2)])
            .collect();
        let draws3 = (0..n_draws)
            .map(|draw| {
                idx.iter()
                    .map(|&ti| vec![draw_at(draw, ti, 0), draw_at(draw, ti, 1), draw_at(draw, ti, 2)])
                    .collect()
            })
            .collect();
        let names3 = vec![names[0].clone(), names[1].clone(), names[2].clone()];
        let projection = if channels > 3 {
            "first_three_channels"
        } else {
            "native_3d"
        };
        (clean3, draws3, names3, projection)
    } else if channels == 2 {
        let clean3 = idx
            .iter()
            .map(|&ti| vec![clean_at(ti, 0), clean_at(ti, 1), 0.0])
            .collect();
        let draws3 = (0..n_draws)
            .map(|draw| {
                idx.iter()
                    .map(|&ti| vec![draw_at(draw, ti, 0), draw_at(draw, ti, 1), 0.0])
                    .collect()
            })
            .collect();
        let names3 = vec![names[0].clone(), names[1].clone(), "zero".to_string()];
        (clean3, draws3, names3, "2d_padded_z0")
    } else {
        // channels == 1: spread over a time axis on x, value on y, zero on z.
        let clean3 = (0..points)
            .map(|j| {
                let time = if points > 1 {
                    -1.0 + 2.0 * j as f32 / (points as f32 - 1.0)
                } else {
                    0.0
                };
                vec![time, clean_at(idx[j], 0), 0.0]
            })
            .collect();
        let draws3 = (0..n_draws)
            .map(|draw| (0..points).map(|j| vec![0.0, draw_at(draw, idx[j], 0), 0.0]).collect())
            .collect();
        let names3 = vec!["time".to_string(), names[0].clone(), "zero".to_string()];
        (clean3, draws3, names3, "1d_time_value_z0")
    }
}

// ── shared helpers ──────────────────────────────────────────────────────────

/// Map a system name + shock preset to a `TrajectorySpec`.
fn spec_for(system: &str, shock: Shock) -> Fallible<TrajectorySpec> {
    let spec = match (system, shock) {
        ("lorenz", Shock::None) => TrajectorySpec::lorenz(),
        ("lorenz", Shock::Param) => TrajectorySpec::lorenz_param_shock(),
        ("lorenz", Shock::State) => TrajectorySpec::lorenz_state_shock(),
        ("lorenz", Shock::Switch) => TrajectorySpec::lorenz_switch(),
        ("rossler", Shock::None) => TrajectorySpec::rossler(),
        ("rossler", Shock::Param) => TrajectorySpec::rossler_param_shock(),
        ("chua", Shock::None) => TrajectorySpec::chua(),
        ("chua", Shock::Param) => TrajectorySpec::chua_param_shock(),
        ("chua", Shock::Switch) => TrajectorySpec::chua_switch(),
        ("lorenz96", Shock::None) => TrajectorySpec::lorenz96(),
        ("lorenz96", Shock::Switch) => TrajectorySpec::lorenz96_switch(),
        ("ou", Shock::None) => TrajectorySpec::ou(),
        ("ou", Shock::Param) => TrajectorySpec::ou_param_shock(),
        ("doublewell", Shock::None) => TrajectorySpec::doublewell(),
        ("doublewell", Shock::Param) => TrajectorySpec::doublewell_param_shock(),
        ("doublewell", Shock::Switch) => TrajectorySpec::doublewell_switch(),
        ("slds", Shock::None) => TrajectorySpec::slds(),
        ("slds", Shock::Param) => TrajectorySpec::slds_param_shock(),
        ("slds", Shock::Switch) => TrajectorySpec::slds_switch(),
        ("seasonal_ar", Shock::None) => TrajectorySpec::seasonal_ar(),
        ("seasonal_ar", Shock::Param) => TrajectorySpec::seasonal_ar_param_shock(),
        ("garch", Shock::None) => TrajectorySpec::garch(),
        ("garch", Shock::Param) => TrajectorySpec::garch_param_shock(),
        (other, shock) => {
            return Err(
                format!("no preset for system {other:?} with shock {shock:?}; run `serie-rs list-systems`").into(),
            );
        }
    };
    Ok(spec)
}

/// Format a sigma for a `noisy_sigma_*` tensor name (trailing zeros trimmed).
fn format_sigma(sigma: f64) -> String {
    let mut text = format!("{sigma:.6}");
    while text.contains('.') && text.ends_with('0') {
        text.pop();
    }
    if text.ends_with('.') {
        text.pop();
    }
    text
}
