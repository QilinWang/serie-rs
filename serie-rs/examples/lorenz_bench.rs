use serde_json::json;
use serie_rs::stage_generate::generate_ode;
use std::hint::black_box;
use std::time::Instant;

fn main() -> Result<(), String> {
    let (steps, repetitions) = arguments()?;
    let initial = [1.0, 0.98, 1.1];
    let params = [10.0, 28.0, 8.0 / 3.0];

    black_box(generate_ode("lorenz", &initial, steps, 1e-2, &[], &params, 3)?);

    let started = Instant::now();
    let mut trajectory = Vec::new();
    for _ in 0..repetitions {
        trajectory = black_box(generate_ode("lorenz", &initial, steps, 1e-2, &[], &params, 3)?);
    }
    let elapsed = started.elapsed();
    let last = trajectory.len() - 3;

    println!(
        "{}",
        json!({
            "engine": "rust",
            "steps": steps,
            "repetitions": repetitions,
            "elapsed_ns": elapsed.as_nanos(),
            "ns_per_run": elapsed.as_nanos() as f64 / repetitions as f64,
            "final_state": [trajectory[last], trajectory[last + 1], trajectory[last + 2]],
            "trajectory_fnv64": format!("{:016x}", fnv64(&trajectory)),
        })
    );
    Ok(())
}

fn fnv64(values: &[f64]) -> u64 {
    const OFFSET: u64 = 14_695_981_039_346_656_037;
    const PRIME: u64 = 1_099_511_628_211;
    let mut hash = OFFSET;
    for value in values {
        for byte in value.to_bits().to_le_bytes() {
            hash ^= u64::from(byte);
            hash = hash.wrapping_mul(PRIME);
        }
    }
    hash
}

fn arguments() -> Result<(usize, usize), String> {
    let args = std::env::args().skip(1).collect::<Vec<_>>();
    if args.len() != 2 {
        return Err("usage: lorenz_bench <steps> <repetitions>".to_string());
    }
    let steps = args[0]
        .parse::<usize>()
        .map_err(|_| "steps must be a non-negative integer".to_string())?;
    let repetitions = args[1]
        .parse::<usize>()
        .map_err(|_| "repetitions must be a positive integer".to_string())?;
    if repetitions == 0 {
        return Err("repetitions must be a positive integer".to_string());
    }
    Ok((steps, repetitions))
}
