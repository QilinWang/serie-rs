//! serie_rs::stage_generate — Rust core for dynamical systems trajectory generation.
//!
//! Architecture (maps 1:1 to the Python refactoring target):
//!
//!   - Each system is a struct carrying its parameters
//!   - The struct `impl`s OdeSystem (continuous), MapSystem (discrete), or SdeDrift (stochastic)
//!   - Integrators are generic over the trait — one RK4 for all ODEs, one EM for all SDEs
//!   - Shock scheduling is a separate concern (list of ShockEvents)
//!   - Public Rust entry points at the bottom use native ShockEvent lists
//!
//! This replaces the Python pattern of:
//!   SYSTEMS dict → string dispatch → _params_as_args bridge → _run_ode_py
//! with:
//!   struct LorenzParams → impl OdeSystem → integrate_rk4<LorenzParams>
use crate::stage_params::{
    ChuaParams, DoubleWellParams, DuffingParams, GarchParams, HenonParams, HyperRosslerParams, LgSsmParams,
    LogisticParams, Lorenz96Params, LorenzParams, OuParams, RosslerParams, SeasonalArParams, SldsParams,
    VanDerPolParams,
};
use nalgebra::{DMatrix, DVector, linalg::Cholesky};
use rand::RngExt; // brings random::<f64>() (uniform [0,1)) into scope
use rand::SeedableRng;
use rand_distr::{Distribution, StandardNormal};
use rand_pcg::Pcg64;

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// §1  Traits
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
//
// Three integration families, three traits. Each system struct implements
// exactly one. The integrators are generic over the trait, so adding a
// new system means: define a struct, impl the trait, add a match arm.

/// Continuous-time dynamical system: dx/dt = f(x, t; params).
///
/// Implementors carry their own parameters and compute the vector field
/// in-place into `out`, avoiding allocation in the integration hot loop.
trait OdeSystem {
    fn dim(&self) -> usize;
    /// Compute dx/dt at state `x`, time `t`. Write result into `out`.
    /// Both `x` and `out` are slices of length `dim()`.
    fn derivative(&self, x: &[f64], t: f64, out: &mut [f64]);
}

/// Discrete-time map: x_{n+1} = f(x_n; params).
///
/// Implementors carry parameters and compute the next iterate in-place.
trait MapSystem {
    fn dim(&self) -> usize;
    /// Compute x_{n+1} from x_n. Write result into `out`.
    fn step(&self, x: &[f64], out: &mut [f64]);
}

/// SDE drift function: the deterministic part of dx = f(x,t)dt + σdW.
///
/// Separate from OdeSystem because SDE integration uses Euler-Maruyama
/// (not RK4), and the noise term is handled by the integrator, not the
/// system. Drift functions receive ALL params from the Python param class
/// via from_slice, but only read the ones relevant to the deterministic
/// dynamics — sigma_noise is in the slice but ignored here (Python
/// extracts it into sigma_per_regime to avoid field-index coupling).
trait SdeDrift {
    fn dim(&self) -> usize;
    /// Compute the drift f(x, t) of the SDE. Write result into `out`.
    fn drift(&self, x: &[f64], t: f64, out: &mut [f64]);
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// §2  ODE Systems
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
//
// Parameter structs live in stage_params; this module adds the dynamics
// impls plus from_slice helpers used by the flat-array generation path.
// Adding a new system means adding the typed params there and the behavior here.
//
// NOTE on from_slice: Rust allows adding methods to a struct from
// anywhere in the file via `impl StructName { ... }`. Think of it
// like adding a @classmethod to a Python class after the class body.
// p: &[f64] means "a borrowed read-only view into a list of floats."
// No copy. Like getting a numpy view.
// -> Self means "returns an instance of this struct."

impl OdeSystem for LorenzParams {
    fn dim(&self) -> usize {
        3
    }
    fn derivative(&self, x: &[f64], _t: f64, out: &mut [f64]) {
        // dx/dt = σ(y - x)
        out[0] = self.sigma * (x[1] - x[0]);
        // dy/dt = x(ρ - z) - y
        out[1] = x[0] * (self.rho - x[2]) - x[1];
        // dz/dt = xy - βz
        out[2] = x[0] * x[1] - self.beta * x[2];
    }
}

impl LorenzParams {
    fn from_slice(p: &[f64]) -> Self {
        Self {
            sigma: p[0],
            rho: p[1],
            beta: p[2],
        }
    }
}

impl OdeSystem for RosslerParams {
    fn dim(&self) -> usize {
        3
    }
    fn derivative(&self, x: &[f64], _t: f64, out: &mut [f64]) {
        out[0] = -x[1] - x[2]; // dx/dt = -y - z
        out[1] = x[0] + self.a * x[1]; // dy/dt = x + ay
        out[2] = self.b + x[2] * (x[0] - self.c); // dz/dt = b + z(x - c)
    }
}

impl RosslerParams {
    fn from_slice(p: &[f64]) -> Self {
        Self {
            a: p[0],
            b: p[1],
            c: p[2],
        }
    }
}

impl OdeSystem for ChuaParams {
    fn dim(&self) -> usize {
        3
    }
    fn derivative(&self, x: &[f64], _t: f64, out: &mut [f64]) {
        /*
        Chua circuit with piecewise-linear nonlinearity h(x).

        h(x) = m0_outer_gt1 * x
            + 0.5 * (m1_inner_le1 - m0_outer_gt1) * (|x+1| - |x-1|)

        This gives:
        slope = m1_inner_le1 for |x| <= 1
        slope = m0_outer_gt1 for |x| > 1

        ODE:
        dx/dt = alpha * (y - x - h(x))
        dy/dt = x - y + z
        dz/dt = -beta * y
        https://link.springer.com/content/pdf/10.1007/s11071-022-08078-y.pdf
        https://www.physics.smu.edu/~olness/quarknet/chaoscircuit.pdf
        https://acoustique.ec-lyon.fr/chaos/Chua_from-Alligood.pdf
        */
        let h = self.m0_outer_gt1 * x[0]
            + 0.5 * (self.m1_inner_le1 - self.m0_outer_gt1) * ((x[0] + 1.0).abs() - (x[0] - 1.0).abs());
        out[0] = self.alpha * (x[1] - x[0] - h); // dx/dt = α(y - x - h(x))
        out[1] = x[0] - x[1] + x[2]; // dy/dt = x - y + z
        out[2] = -self.beta * x[1]; // dz/dt = -βy
    }
}

impl ChuaParams {
    fn from_slice(p: &[f64]) -> Self {
        Self {
            alpha: p[0],
            beta: p[1],
            m1_inner_le1: p[2],
            m0_outer_gt1: p[3],
        }
    }
}

impl OdeSystem for VanDerPolParams {
    fn dim(&self) -> usize {
        2
    }
    fn derivative(&self, x: &[f64], _t: f64, out: &mut [f64]) {
        out[0] = x[1]; // dx/dt = y
        out[1] = self.mu * (1.0 - x[0] * x[0]) * x[1] - x[0]; // dy/dt = μ(1-x²)y - x
    }
}

impl VanDerPolParams {
    fn from_slice(p: &[f64]) -> Self {
        Self { mu: p[0] }
    }
}

impl OdeSystem for DuffingParams {
    fn dim(&self) -> usize {
        2
    }
    fn derivative(&self, x: &[f64], t: f64, out: &mut [f64]) {
        out[0] = x[1]; // dx/dt = v
        // dv/dt = γcos(ωt) - δv - αx - βx³
        out[1] = self.gamma * (self.omega * t).cos()
            - self.delta * x[1]
            - self.alpha * x[0]
            - self.beta * x[0] * x[0] * x[0];
    }
}

impl DuffingParams {
    fn from_slice(p: &[f64]) -> Self {
        Self {
            alpha: p[0],
            beta: p[1],
            delta: p[2],
            gamma: p[3],
            omega: p[4],
        }
    }
}

impl OdeSystem for HyperRosslerParams {
    fn dim(&self) -> usize {
        4
    }
    fn derivative(&self, x: &[f64], _t: f64, out: &mut [f64]) {
        out[0] = -x[1] - x[2]; // dx/dt = -y - z
        out[1] = x[0] + self.a * x[1] + x[3]; // dy/dt = x + ay + w
        out[2] = self.b + x[0] * x[2]; // dz/dt = b + xz
        out[3] = -self.c * x[2] + self.d * x[3]; // dw/dt = -cz + dw
    }
}

impl HyperRosslerParams {
    fn from_slice(p: &[f64]) -> Self {
        Self {
            a: p[0],
            b: p[1],
            c: p[2],
            d: p[3],
        }
    }
}

impl OdeSystem for Lorenz96Params {
    fn dim(&self) -> usize {
        self.dim
    }
    fn derivative(&self, x: &[f64], _t: f64, out: &mut [f64]) {
        let n = self.dim;
        for i in 0..n {
            let im2 = (i + n - 2) % n; // x_{i-2} (periodic)
            let im1 = (i + n - 1) % n; // x_{i-1} (periodic)
            let ip1 = (i + 1) % n; // x_{i+1} (periodic)
            // dx_i/dt = (x_{i+1} - x_{i-2}) · x_{i-1} - x_i + F
            out[i] = (x[ip1] - x[im2]) * x[im1] - x[i] + self.forcing;
        }
    }
}

///Lorenz96 is special: it takes an extra dim argument because dim comes from the initial
/// condition length, not from the param array. Python only sends [forcing] as params.
/// The dim shorthand on the right side — Self { forcing: p[0], dim } — is Rust's field
/// init shorthand: when variable name matches field name, you can skip the colon.
/// Same as Python's dict(x=x) but terser.
impl Lorenz96Params {
    fn from_slice(p: &[f64], dim: usize) -> Self {
        Self { forcing: p[0], dim }
    }
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// §3  Map Systems
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

impl MapSystem for HenonParams {
    fn dim(&self) -> usize {
        2
    }
    fn step(&self, x: &[f64], out: &mut [f64]) {
        out[0] = 1.0 - self.a * x[0] * x[0] + x[1]; // x_{n+1} = 1 - ax² + y
        out[1] = self.b * x[0]; // y_{n+1} = bx
    }
}

impl HenonParams {
    fn from_slice(p: &[f64]) -> Self {
        Self { a: p[0], b: p[1] }
    }
}

impl MapSystem for LogisticParams {
    fn dim(&self) -> usize {
        1
    }
    fn step(&self, x: &[f64], out: &mut [f64]) {
        out[0] = self.r * x[0] * (1.0 - x[0]); // x_{n+1} = rx(1-x)
    }
}

impl LogisticParams {
    fn from_slice(p: &[f64]) -> Self {
        Self { r: p[0] }
    }
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// §4  SDE Systems
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
//
// SDE systems define only the DRIFT (deterministic part). The noise
// term σ·√dt·ε is handled generically by the Euler-Maruyama integrator.
// sigma_noise is present in the Python param class (and therefore in
// param_matrix), but the drift impl ignores it — Python extracts it
// into sigma_per_regime before calling Rust, avoiding field-index coupling.

impl SdeDrift for OuParams {
    fn dim(&self) -> usize {
        1
    }
    fn drift(&self, x: &[f64], _t: f64, out: &mut [f64]) {
        // dx = θ(μ - x)dt
        out[0] = self.theta * (self.mu - x[0]);
    }
}

impl OuParams {
    /// OUParams (Python) serializes as [theta, mu, sigma_noise].
    /// Drift only needs theta and mu; sigma_noise is at p[2] but ignored.
    fn from_slice(p: &[f64]) -> Self {
        Self {
            theta: p[0],
            mu: p[1],
            sigma_noise: p[2],
        }
    }
}

impl SdeDrift for DoubleWellParams {
    fn dim(&self) -> usize {
        1
    }
    fn drift(&self, x: &[f64], _t: f64, out: &mut [f64]) {
        // dx = (-x³ + ax)dt
        let x0 = x[0];
        out[0] = -(x0 * x0 * x0) + self.a * x0;
    }
}

impl DoubleWellParams {
    /// DoubleWellParams (Python) serializes as [a, sigma_noise].
    /// Drift only needs a; sigma_noise is at p[1] but ignored.
    fn from_slice(p: &[f64]) -> Self {
        Self {
            a: p[0],
            sigma_noise: p[1],
        }
    }
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// §5  Shock Events
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

/// A perturbation event applied during integration at a specific timestep.
///
/// Shocks are the mechanism for testing forecasting robustness:
///   - ParamChange: abrupt regime shift (e.g., ρ: 28→35 in Lorenz)
///   - StatePerturbation: instantaneous state kick (sensitivity to IC)
///
/// Multiple shocks can be composed in a list, sorted by step.
#[derive(Clone, Debug)]
pub enum ShockEvent {
    /// Design note:
    /// Python's multi_shock concatenates param sets schedule-by-schedule,
    /// then sorts events by time. That means the first chronological
    /// ParamChange may legitimately target regime 3 rather than regime 1.
    /// So Rust must honor the explicit index, not just "advance to next".
    ParamChange { step: usize, param_set_idx: usize },
    /// Additive state perturbation: x[state_idx] += eps at `step`.
    StatePerturbation { step: usize, state_idx: usize, eps: f64 },
}

/// Native Rust callers pass ShockEvent values directly.
/// We still normalize ordering here so call sites do not need to pre-sort.
fn sorted_shocks(shocks: &[ShockEvent]) -> Vec<ShockEvent> {
    let mut sorted = shocks.to_vec();
    sorted.sort_by_key(|shock| match shock {
        ShockEvent::ParamChange { step, .. } => *step,
        ShockEvent::StatePerturbation { step, .. } => *step,
    });
    sorted
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// §6  Integrators
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

// ── §6a  ODE: 4th-order Runge-Kutta ─────────────────────────────────

/// Classical 4th-order Runge-Kutta integration. No shocks.
///
/// Returns flat Vec of (steps+1)*dim f64 values, row-major:
///   [x0_0, x0_1, ..., x0_d, x1_0, x1_1, ..., x1_d, ...]
///
/// This is the workhorse: generic over any OdeSystem, monomorphized at
/// compile time so each system gets its own optimized machine code.
/// Zero allocation in the hot loop (scratch buffers pre-allocated).
///
/// The RK4 Math Logic
///
/// The RK4 method works by taking four different "probes" of the derivative to find the best average slope.
///
/// k1: The slope at the beginning of the step.
///
/// k2 & k3: Slopes at the midpoint (using the previous slopes to "guess" where the system will be).
///
/// k4: The slope at the end of the step.
///

fn integrate_rk4<S: OdeSystem>(system: &S, x0: &[f64], steps: usize, dt: f64) -> Vec<f64> {
    let dim = system.dim();
    let mut traj = vec![0.0f64; (steps + 1) * dim];
    traj[..dim].copy_from_slice(x0);

    let mut x = x0.to_vec();
    let mut k1 = vec![0.0; dim];
    let mut k2 = vec![0.0; dim];
    let mut k3 = vec![0.0; dim];
    let mut k4 = vec![0.0; dim];
    let mut tmp = vec![0.0; dim];
    let mut t = 0.0f64;

    for i in 1..=steps {
        // k1
        system.derivative(&x, t, &mut k1);
        // k2
        for d in 0..dim {
            tmp[d] = x[d] + 0.5 * dt * k1[d];
        }
        system.derivative(&tmp, t + 0.5 * dt, &mut k2);
        // k3
        for d in 0..dim {
            tmp[d] = x[d] + 0.5 * dt * k2[d];
        }
        system.derivative(&tmp, t + 0.5 * dt, &mut k3);
        // k4
        for d in 0..dim {
            tmp[d] = x[d] + dt * k3[d];
        }
        system.derivative(&tmp, t + dt, &mut k4);
        // combine
        for d in 0..dim {
            x[d] += (dt / 6.0) * (k1[d] + 2.0 * k2[d] + 2.0 * k3[d] + k4[d]);
        }
        t += dt;

        // Check finite
        for d in 0..dim {
            if !x[d].is_finite() {
                // Fill rest with NaN and return
                for j in i * dim..traj.len() {
                    traj[j] = f64::NAN;
                }
                return traj;
            }
        }

        traj[i * dim..(i + 1) * dim].copy_from_slice(&x);
    }
    traj
}

/// RK4 with shock event support.
///
/// `systems` is indexed by `param_set_idx`, with systems[0] active
/// initially. ShockEvents must be sorted by step. State perturbations
/// modify x in-place without changing the active system.
/// We do not use "next regime" semantics here because merged schedules
/// may jump to a later row first.
///
/// Design: shocks are a *separate concern* from the dynamics. The system
/// structs know nothing about shocks — the integrator handles scheduling.
/// This maps cleanly to the Python refactoring goal: extract shock logic
/// from the 7 copy-pasted shock blocks in _run_*_py.
fn integrate_rk4_shocks<S: OdeSystem>(
    systems: &[S], // systems[param_set_idx] = active regime for that event.
    x0: &[f64],
    steps: usize,
    dt: f64,
    shocks: &[ShockEvent],
) -> Vec<f64> {
    let dim = systems[0].dim();
    let mut traj = vec![0.0f64; (steps + 1) * dim];
    traj[..dim].copy_from_slice(x0);

    let mut x = x0.to_vec();
    let mut k1 = vec![0.0; dim];
    let mut k2 = vec![0.0; dim];
    let mut k3 = vec![0.0; dim];
    let mut k4 = vec![0.0; dim];
    let mut tmp = vec![0.0; dim];
    let mut t = 0.0f64;

    let mut active_sys: usize = 0; // index into systems[]
    let mut shock_idx: usize = 0;

    for i in 1..=steps {
        // Check shocks at this step
        while shock_idx < shocks.len() {
            match &shocks[shock_idx] {
                ShockEvent::ParamChange { step, param_set_idx } if *step == i => {
                    active_sys = *param_set_idx;
                    shock_idx += 1;
                }
                ShockEvent::StatePerturbation { step, state_idx, eps } if *step == i => {
                    x[*state_idx] += eps;
                    shock_idx += 1;
                }
                _ => break,
            }
        }
        // if someone passes more ParamChange shocks than there are entries in systems
        //, .min() clamps to the last one instead of panicking with an out-of-bounds index.
        let sys = &systems[active_sys.min(systems.len() - 1)];

        // RK4 step
        sys.derivative(&x, t, &mut k1);
        for d in 0..dim {
            tmp[d] = x[d] + 0.5 * dt * k1[d];
        }
        sys.derivative(&tmp, t + 0.5 * dt, &mut k2);
        for d in 0..dim {
            tmp[d] = x[d] + 0.5 * dt * k2[d];
        }
        sys.derivative(&tmp, t + 0.5 * dt, &mut k3);
        for d in 0..dim {
            tmp[d] = x[d] + dt * k3[d];
        }
        sys.derivative(&tmp, t + dt, &mut k4);
        for d in 0..dim {
            x[d] += (dt / 6.0) * (k1[d] + 2.0 * k2[d] + 2.0 * k3[d] + k4[d]);
        }
        t += dt;

        for d in 0..dim {
            if !x[d].is_finite() {
                for j in i * dim..traj.len() {
                    traj[j] = f64::NAN;
                }
                return traj;
            }
        }
        traj[i * dim..(i + 1) * dim].copy_from_slice(&x);
    }
    traj
}

// ── §6b  Maps: discrete iteration ───────────────────────────────────

/// Discrete map iteration: x_{n+1} = f(x_n). No shocks.
///
/// Returns flat Vec of (steps+1)*dim values, same layout as integrate_rk4.
fn iterate_map<S: MapSystem>(system: &S, x0: &[f64], steps: usize) -> Vec<f64> {
    let dim = system.dim();
    let mut traj = vec![0.0f64; (steps + 1) * dim];
    traj[..dim].copy_from_slice(x0);

    let mut x = x0.to_vec();
    let mut next = vec![0.0; dim];

    for i in 1..=steps {
        system.step(&x, &mut next);
        for d in 0..dim {
            if !next[d].is_finite() {
                // This range starts exactly where the "failure" happened and goes to the very
                // end of your pre-allocated trajectory data.
                for j in i * dim..traj.len() {
                    traj[j] = f64::NAN;
                }
                return traj;
            }
        }
        x.copy_from_slice(&next);
        traj[i * dim..(i + 1) * dim].copy_from_slice(&x);
    }
    traj
}

// ── §6c  SDE: Euler-Maruyama ────────────────────────────────────────

/// Euler-Maruyama integration for SDEs. No shocks.
///
/// x_{n+1} = x_n + f(x_n, t_n) * dt + σ * √dt * ε_n
///
/// `noise` is a pre-generated flat array of unit normals N(0,1),
/// length = steps * dim. Layout: [step0_dim0, step0_dim1, ..., step1_dim0, ...]
/// Python generates this via np.random.default_rng(seed).standard_normal((steps, dim)).ravel()
///
/// Returns flat Vec of (steps+1)*dim f64 values, same row-major layout as integrate_rk4.
fn integrate_em<S: SdeDrift>(system: &S, x0: &[f64], steps: usize, dt: f64, sigma: f64, noise: &[f64]) -> Vec<f64> {
    let dim = system.dim();
    let sqrt_dt = dt.sqrt();
    let mut traj = vec![0.0f64; (steps + 1) * dim];
    traj[..dim].copy_from_slice(x0);

    let mut x = x0.to_vec();
    let mut drift_buf = vec![0.0; dim];
    let mut t = 0.0f64;

    for i in 1..=steps {
        system.drift(&x, t, &mut drift_buf);

        // EM step: x += f(x)*dt + σ*√dt*ε
        // noise[(i-1)*dim .. i*dim] are the unit normals for this step
        let noise_offset = (i - 1) * dim;
        for d in 0..dim {
            x[d] += drift_buf[d] * dt + sigma * sqrt_dt * noise[noise_offset + d];
        }
        t += dt;

        // NaN guard
        for d in 0..dim {
            if !x[d].is_finite() {
                for j in i * dim..traj.len() {
                    traj[j] = f64::NAN;
                }
                return traj;
            }
        }
        traj[i * dim..(i + 1) * dim].copy_from_slice(&x);
    }
    traj
}

fn standard_normal_noise(seed: u64, steps: usize, dim: usize) -> Vec<f64> {
    let n = steps.checked_mul(dim).expect("steps * dim overflow");
    let mut rng = Pcg64::seed_from_u64(seed);
    let dist = StandardNormal;
    (0..n).map(|_| dist.sample(&mut rng)).collect()
}

fn integrate_em_seeded<S: SdeDrift>(system: &S, x0: &[f64], steps: usize, dt: f64, sigma: f64, seed: u64) -> Vec<f64> {
    let noise = standard_normal_noise(seed, steps, system.dim());
    integrate_em(system, x0, steps, dt, sigma, &noise)
}

/// Euler-Maruyama with shock event support.
///
/// `systems` is indexed by `param_set_idx`, with systems[0] active initially.
///
/// `sigma_per_regime` maps regime index → diffusion coefficient.
/// When a ParamChange fires, we switch to that explicit regime index
/// and update sigma to match.
/// This is the key difference from the ODE shock integrator, where sigma
/// doesn't exist.
///
/// Shock timing convention: shocks fire when step == i (1-indexed),
/// matching integrate_rk4_shocks and the Python _run_sde_py loop
/// `for i in range(1, steps+1): if i == shock_step: ...`
fn integrate_em_shocks<S: SdeDrift>(
    systems: &[S],
    x0: &[f64],
    steps: usize,
    dt: f64,
    sigma_per_regime: &[f64],
    noise: &[f64],
    shocks: &[ShockEvent],
) -> Vec<f64> {
    let dim = systems[0].dim();
    let sqrt_dt = dt.sqrt();
    let mut traj = vec![0.0f64; (steps + 1) * dim];
    traj[..dim].copy_from_slice(x0);

    let mut x = x0.to_vec();
    let mut drift_buf = vec![0.0; dim];
    let mut t = 0.0f64;

    let mut active_regime: usize = 0;
    let mut sigma = sigma_per_regime[0];
    let mut shock_idx: usize = 0;

    for i in 1..=steps {
        // Process shocks at this step (same convention as integrate_rk4_shocks)
        while shock_idx < shocks.len() {
            match &shocks[shock_idx] {
                ShockEvent::ParamChange { step, param_set_idx } if *step == i => {
                    active_regime = *param_set_idx;
                    sigma = sigma_per_regime[active_regime.min(sigma_per_regime.len() - 1)];
                    shock_idx += 1;
                }
                ShockEvent::StatePerturbation { step, state_idx, eps } if *step == i => {
                    x[*state_idx] += eps;
                    shock_idx += 1;
                }
                _ => break,
            }
        }

        let sys = &systems[active_regime.min(systems.len() - 1)];
        sys.drift(&x, t, &mut drift_buf);

        let noise_offset = (i - 1) * dim;
        for d in 0..dim {
            x[d] += drift_buf[d] * dt + sigma * sqrt_dt * noise[noise_offset + d];
        }
        t += dt;

        for d in 0..dim {
            if !x[d].is_finite() {
                for j in i * dim..traj.len() {
                    traj[j] = f64::NAN;
                }
                return traj;
            }
        }
        traj[i * dim..(i + 1) * dim].copy_from_slice(&x);
    }
    traj
}

fn integrate_em_shocks_seeded<S: SdeDrift>(
    systems: &[S],
    x0: &[f64],
    steps: usize,
    dt: f64,
    sigma_per_regime: &[f64],
    seed: u64,
    shocks: &[ShockEvent],
) -> Vec<f64> {
    let noise = standard_normal_noise(seed, steps, systems[0].dim());
    integrate_em_shocks(systems, x0, steps, dt, sigma_per_regime, &noise, shocks)
}

// ── §6d  Discrete statistical systems (SLDS, Seasonal AR, GARCH) ────
//
// These are stepwise stochastic recurrences, not ODE/SDE integrations:
// the noise law is part of the system definition (regime-dependent
// variance, conditional variance), so each system gets its own runner
// instead of a shared integrator — exactly like the Python side, where
// _run_slds / _run_seasonal_ar / _run_garch in serie/integrators.py are
// bespoke loops next to the generic _run_ode_py/_run_sde_py.
//
// Each runner takes injected draws so the deterministic recurrence is
// parity-testable (same pattern as integrate_em taking a `noise` slice);
// the *_seeded wrappers draw them from one Pcg64 stream. Python
// interleaves its draws step-by-step from one numpy Generator; Rust
// pre-draws blocks from one Pcg64 stream. The noise LAWS match; the
// stream order does not — cross-engine byte parity is explicitly not a
// goal (`data_engine` carries the difference).
//
// Shock timing convention: shocks fire when step == t (1-indexed),
// matching integrate_rk4_shocks and the Python loops
// `for t in range(1, steps + 1): while events[event_idx].step == t: ...`.
//
// Like the Python discrete runners (and unlike _run_ode_py/_run_sde_py),
// there is no finiteness guard here — Python's discrete loops have no
// isfinite check, so none is added on this side either.

impl SldsParams {
    /// SLDSParams (Python) serializes as [A1, Q1, A2, Q2, p11, p22]
    /// (SystemParams::as_f64_vec order).
    fn from_slice(p: &[f64]) -> Self {
        Self {
            a1: p[0],
            q1: p[1],
            a2: p[2],
            q2: p[3],
            p11: p[4],
            p22: p[5],
        }
    }
}

impl SeasonalArParams {
    /// SeasonalARParams (Python) serializes as [S, phi, sigma, a0, amp_drift].
    fn from_slice(p: &[f64]) -> Self {
        Self {
            s: p[0],
            phi: p[1],
            sigma: p[2],
            a0: p[3],
            amp_drift: p[4],
        }
    }
}

impl GarchParams {
    /// GARCHParams (Python) serializes as [omega, alpha, beta].
    fn from_slice(p: &[f64]) -> Self {
        Self {
            omega: p[0],
            alpha: p[1],
            beta: p[2],
        }
    }
}

/// Switching linear dynamical system (K=2 regimes). Port of _run_slds
/// (serie/integrators.py).
///
/// Markov-switching AR(1) on a scalar state:
///   z_t ∈ {1,2},  P(z_t=1 | z_{t-1}=1) = p11,  P(z_t=2 | z_{t-1}=2) = p22
///   x_t = A_{z_t}·x_{t-1} + √Q_{z_t}·ε_t
///
/// Regime timing: at each step, z_t is drawn from the Markov transition
/// given z_{t-1}, THEN the step runs with the NEWLY drawn regime's (A, Q) —
/// a switch drawn at step t drives the dynamics at step t itself, matching
/// the SLDSParams docstring formula above.
/// CHG (user-approved 2026-07-01): both engines previously stepped with the
/// pre-transition regime's (A, Q); the standard convention now applies in
/// Python _run_slds and here, so slds / slds_param_shock / slds_switch
/// trajectories differ from all pre-fix runs.
///
/// `regime_uniforms` has length steps+1: [0] decides the initial regime
/// (z = 1 if u < 0.5 else 2, matching `z = 1 if rng.random() < 0.5 else 2`),
/// and [t] is the transition draw at step t.
/// `unit_normals` has length steps: unit_normals[t-1] is ε_t, scaled by
/// √Q here (Python draws rng.normal(0, sqrt(Q)) directly — same law).
fn run_slds(
    systems: &[SldsParams],
    x0: &[f64],
    steps: usize,
    regime_uniforms: &[f64],
    unit_normals: &[f64],
    shocks: &[ShockEvent],
) -> Vec<f64> {
    let mut traj = vec![0.0f64; steps + 1];
    let mut x = x0[0];
    traj[0] = x;

    // Start regime randomly
    let mut z: u8 = if regime_uniforms[0] < 0.5 { 1 } else { 2 };

    let mut active_params: usize = 0;
    let mut shock_idx: usize = 0;

    for t in 1..=steps {
        // Apply all events scheduled for this step.
        while shock_idx < shocks.len() {
            match &shocks[shock_idx] {
                ShockEvent::ParamChange { step, param_set_idx } if *step == t => {
                    active_params = *param_set_idx;
                    shock_idx += 1;
                }
                ShockEvent::StatePerturbation { step, eps, .. } if *step == t => {
                    // Python _run_slds: `x += e.eps` — the state is scalar,
                    // so state_idx is ignored there and here.
                    x += eps;
                    shock_idx += 1;
                }
                _ => break,
            }
        }
        let p = &systems[active_params.min(systems.len() - 1)];

        // Transition regime, then evolve with the NEWLY drawn regime.
        z = match z {
            1 => {
                if regime_uniforms[t] < p.p11 {
                    1
                } else {
                    2
                }
            }
            _ => {
                if regime_uniforms[t] < p.p22 {
                    2
                } else {
                    1
                }
            }
        };
        let (a, q) = if z == 1 { (p.a1, p.q1) } else { (p.a2, p.q2) };

        // Evolve: x = A_{z_t}·x + N(0, Q_{z_t})
        x = a * x + q.sqrt() * unit_normals[t - 1];
        traj[t] = x;
    }
    traj
}

/// One Pcg64 stream per SLDS trajectory: the regime-uniform block first
/// (steps+1 draws), then the process-normal block (steps draws).
fn slds_draws(seed: u64, steps: usize) -> (Vec<f64>, Vec<f64>) {
    let mut rng = Pcg64::seed_from_u64(seed);
    let regime_uniforms: Vec<f64> = (0..=steps).map(|_| rng.random::<f64>()).collect();
    let dist = StandardNormal;
    let unit_normals: Vec<f64> = (0..steps).map(|_| dist.sample(&mut rng)).collect();
    (regime_uniforms, unit_normals)
}

fn run_slds_seeded(systems: &[SldsParams], x0: &[f64], steps: usize, seed: u64, shocks: &[ShockEvent]) -> Vec<f64> {
    let (regime_uniforms, unit_normals) = slds_draws(seed, steps);
    run_slds(systems, x0, steps, &regime_uniforms, &unit_normals, shocks)
}

/// Seasonal AR(1) with optionally drifting seasonal amplitude. Port of
/// _run_seasonal_ar (serie/integrators.py).
///
///   a(t) = a0 + amp_drift·t
///   y_t  = a(t)·sin(2πt/S) + φ·y_{t-1} + σ·ε_t
///
/// t is the raw 1-indexed integration step (burn-in is sliced off
/// afterwards by the caller), so the seasonal phase is anchored to the
/// integration index exactly like the Python loop
/// `for t in range(1, steps + 1)`.
///
/// `unit_normals` has length steps: unit_normals[t-1] is ε_t, scaled by
/// the active regime's σ here (Python draws rng.normal(0, sigma) — same law).
fn run_seasonal_ar(
    systems: &[SeasonalArParams],
    x0: &[f64],
    steps: usize,
    unit_normals: &[f64],
    shocks: &[ShockEvent],
) -> Vec<f64> {
    let mut traj = vec![0.0f64; steps + 1];
    let mut y = x0[0];
    traj[0] = y;

    let mut active_params: usize = 0;
    let mut shock_idx: usize = 0;

    for t in 1..=steps {
        // Apply all events scheduled for this step.
        while shock_idx < shocks.len() {
            match &shocks[shock_idx] {
                ShockEvent::ParamChange { step, param_set_idx } if *step == t => {
                    active_params = *param_set_idx;
                    shock_idx += 1;
                }
                ShockEvent::StatePerturbation { step, eps, .. } if *step == t => {
                    // Python _run_seasonal_ar: `y += e.eps` (scalar state).
                    y += eps;
                    shock_idx += 1;
                }
                _ => break,
            }
        }
        let p = &systems[active_params.min(systems.len() - 1)];

        let t_f = t as f64;
        let a_t = p.a0 + p.amp_drift * t_f;
        let seasonal = a_t * (2.0 * std::f64::consts::PI * t_f / p.s).sin();
        y = seasonal + p.phi * y + p.sigma * unit_normals[t - 1];
        traj[t] = y;
    }
    traj
}

/// GARCH(1,1). Port of _run_garch (serie/integrators.py).
///
///   σ²_t = ω + α·r²_{t-1} + β·σ²_{t-1}   (clamped at 0)
///   r_t  = σ_t·ε_t,  ε_t ~ N(0,1)
///
/// Output is 2 channels per timestep: [r, σ²]. x0 supplies both initial
/// values: x0[0] = r_0, x0[1] = σ²_0 (the preset computes
/// σ²_0 = ω/(1-α-β); this runner takes it as given, matching Python's
/// "init is fully supplied by x0 (no sigma2_0 logic here)").
///
/// State perturbations mirror Python exactly: `traj[t-1, 0] += eps` —
/// the STORED previous return is perturbed, so the kick is visible in
/// output row t-1 and feeds the σ² recursion; state_idx is ignored.
///
/// `unit_normals` has length steps: unit_normals[t-1] is ε_t
/// (Python draws rng.standard_normal() directly).
fn run_garch(
    systems: &[GarchParams],
    x0: &[f64],
    steps: usize,
    unit_normals: &[f64],
    shocks: &[ShockEvent],
) -> Vec<f64> {
    let mut traj = vec![0.0f64; (steps + 1) * 2];
    traj[0] = x0[0]; // r_0
    traj[1] = x0[1]; // store initial sigma2
    let mut sigma2 = x0[1];

    let mut active_params: usize = 0;
    let mut shock_idx: usize = 0;

    for t in 1..=steps {
        // Apply all events scheduled for this step.
        while shock_idx < shocks.len() {
            match &shocks[shock_idx] {
                ShockEvent::ParamChange { step, param_set_idx } if *step == t => {
                    active_params = *param_set_idx;
                    shock_idx += 1;
                }
                ShockEvent::StatePerturbation { step, eps, .. } if *step == t => {
                    // Python _run_garch: `traj[t - 1, 0] += e.eps`.
                    traj[(t - 1) * 2] += eps;
                    shock_idx += 1;
                }
                _ => break,
            }
        }
        let p = &systems[active_params.min(systems.len() - 1)];

        let r_prev = traj[(t - 1) * 2]; // read r from traj
        sigma2 = p.omega + p.alpha * (r_prev * r_prev) + p.beta * sigma2;
        sigma2 = sigma2.max(0.0);

        let r_t = sigma2.sqrt() * unit_normals[t - 1];
        traj[t * 2] = r_t;
        traj[t * 2 + 1] = sigma2; // store sigma2 state
    }
    traj
}

fn run_seasonal_ar_seeded(
    systems: &[SeasonalArParams],
    x0: &[f64],
    steps: usize,
    seed: u64,
    shocks: &[ShockEvent],
) -> Vec<f64> {
    let unit_normals = standard_normal_noise(seed, steps, 1);
    run_seasonal_ar(systems, x0, steps, &unit_normals, shocks)
}

fn run_garch_seeded(systems: &[GarchParams], x0: &[f64], steps: usize, seed: u64, shocks: &[ShockEvent]) -> Vec<f64> {
    let unit_normals = standard_normal_noise(seed, steps, 1);
    run_garch(systems, x0, steps, &unit_normals, shocks)
}

// ── §6e  LG-SSM (matrix OU, exact discretization) ───────────────────
//
// Port of the LG-SSM block in serie/catalog.py (LGSSMParams,
// _lg_ssm_mixing_matrix, lg_ssm_drift_matrix, lg_ssm_discretize_exact) and
// _run_lg_ssm in serie/integrators.py. Roadmap:
// docs/lab/directions/controlled_model_families_roadmap.md §15.2 (matrix OU,
// Van Loan), §15.3 (damped oscillatory blocks), §3.5/§3.6 (controlled
// non-normality κ(V)).
//
// The generator is an exact stepwise recurrence — x_{k+1} = A·x_k +
// chol(Q)·ε_k with A = exp(F·dt) and Q from the Van Loan block exponential —
// so it lives with the discrete statistical runners, takes injected draws
// for parity-testable determinism, and does NO Euler stepping. Python uses
// scipy.linalg.expm; here nalgebra's `exp` (both scaling-and-squaring Padé).
// Deterministic linear algebra (A, Q) is pinned cross-language to 1e-12;
// RNG streams differ by design (`data_engine` carries the difference).

impl LgSsmParams {
    /// LGSSMParams (Python) flattens as [gammas…, omegas…, kappa_v, q_c]
    /// (SystemParams::as_f64_vec order). Block count = (len - 2) / 2.
    fn from_slice(p: &[f64]) -> Self {
        let n_blocks = (p.len() - 2) / 2;
        Self {
            gammas: p[..n_blocks].to_vec(),
            omegas: p[n_blocks..2 * n_blocks].to_vec(),
            kappa_v: p[2 * n_blocks],
            q_c: p[2 * n_blocks + 1],
        }
    }
}

/// Orthonormal DCT-II matrix W, shape [dim, dim] — the deterministic
/// orthogonal mixing factor of V = diag(σ)·W. Mirrors _lg_ssm_mixing_matrix:
/// W[k, n] = a_k·cos(π·(n + 1/2)·k / dim), a_0 = √(1/dim), a_k = √(2/dim).
fn lg_ssm_mixing_matrix(dim: usize) -> DMatrix<f64> {
    let mut w = DMatrix::<f64>::zeros(dim, dim);
    for k in 0..dim {
        let scale = if k == 0 {
            (1.0 / dim as f64).sqrt()
        } else {
            (2.0 / dim as f64).sqrt()
        };
        for n in 0..dim {
            w[(k, n)] = scale * (std::f64::consts::PI * (n as f64 + 0.5) * k as f64 / dim as f64).cos();
        }
    }
    w
}

/// Build the LG-SSM continuous-time drift F = V B_c V⁻¹, shape [D, D].
/// Mirrors lg_ssm_drift_matrix (serie/catalog.py): B_c is the §15.3
/// block-diagonal of [[-γ_i, -ω_i], [ω_i, -γ_i]]; V = diag(σ)·W with
/// σ_j = kappa_v^{j/(D-1)}, so cond₂(V) = kappa_v (§3.5) and
/// V⁻¹ = Wᵀ·diag(1/σ) in closed form — no numerical inversion.
fn lg_ssm_drift_matrix(params: &LgSsmParams) -> Result<DMatrix<f64>, String> {
    if params.gammas.len() != params.omegas.len() || params.gammas.is_empty() {
        return Err(format!(
            "lg_ssm requires one omega per gamma with at least one block, got {} gammas and {} omegas",
            params.gammas.len(),
            params.omegas.len()
        ));
    }
    if params.kappa_v < 1.0 {
        return Err(format!(
            "kappa_v is a condition number and must be >= 1, got {}",
            params.kappa_v
        ));
    }
    let dim = 2 * params.gammas.len();
    let mut b = DMatrix::<f64>::zeros(dim, dim);
    for (block_idx, (gamma, omega)) in params.gammas.iter().zip(params.omegas.iter()).enumerate() {
        // §15.3: F_i = [[-γ_i, -ω_i], [ω_i, -γ_i]], eigenvalues -γ_i ± iω_i.
        b[(2 * block_idx, 2 * block_idx)] = -gamma;
        b[(2 * block_idx, 2 * block_idx + 1)] = -omega;
        b[(2 * block_idx + 1, 2 * block_idx)] = *omega;
        b[(2 * block_idx + 1, 2 * block_idx + 1)] = -gamma;
    }
    let w = lg_ssm_mixing_matrix(dim);
    // σ_j = kappa_v^{j/(D-1)}: geometric spread from 1 to kappa_v.
    let sigma: Vec<f64> = (0..dim)
        .map(|j| params.kappa_v.powf(j as f64 / (dim as f64 - 1.0)))
        .collect();
    // F = diag(σ)·W·B_c·Wᵀ·diag(1/σ) — the §3.5 similarity A = V B V⁻¹ in
    // continuous time. W first rotates B_c into a dense normal matrix, then
    // the diagonal skew couples all directions with ratios up to kappa_v.
    let mut left = w.clone(); // diag(σ)·W: scale rows of W
    for k in 0..dim {
        left.row_mut(k).scale_mut(sigma[k]);
    }
    let mut right = w.transpose(); // Wᵀ·diag(1/σ): scale columns of Wᵀ
    for k in 0..dim {
        right.column_mut(k).scale_mut(1.0 / sigma[k]);
    }
    Ok(left * b * right)
}

/// Exactly discretize the LG-SSM SDE at sampling interval dt → (A, Q).
/// Mirrors lg_ssm_discretize_exact (serie/catalog.py).
///
/// Van Loan block exponential (§15.2, [@vanloan1978computing]):
///   M = [[F, L Q_c Lᵀ], [0, -Fᵀ]] · dt,  Φ = exp(M) = [[Φ₁₁, Φ₁₂], [0, Φ₂₂]]
/// then A = Φ₁₁ = exp(F·dt) and Q = Φ₁₂·Aᵀ because Φ₁₂ = Q_dt·Φ₂₂ with
/// Φ₂₂ = exp(-Fᵀ·dt) and Φ₂₂⁻¹ = exp(Fᵀ·dt) = Aᵀ — no matrix inversion.
fn lg_ssm_discretize_exact(params: &LgSsmParams, dt: f64) -> Result<(DMatrix<f64>, DMatrix<f64>), String> {
    if dt <= 0.0 {
        return Err(format!("lg_ssm sampling interval dt must be > 0, got {dt}"));
    }
    if params.q_c <= 0.0 {
        return Err(format!(
            "q_c must be > 0 for a nondegenerate process-noise covariance, got {}",
            params.q_c
        ));
    }
    let f = lg_ssm_drift_matrix(params)?;
    let dim = f.nrows();
    let mut van_loan_block = DMatrix::<f64>::zeros(2 * dim, 2 * dim);
    van_loan_block.view_mut((0, 0), (dim, dim)).copy_from(&f);
    for d in 0..dim {
        // L Q_c Lᵀ with L = I, Q_c = q_c·I.
        van_loan_block[(d, dim + d)] = params.q_c;
    }
    van_loan_block
        .view_mut((dim, dim), (dim, dim))
        .copy_from(&(-f.transpose()));
    let phi = (van_loan_block * dt).exp();
    let a = phi.view((0, 0), (dim, dim)).into_owned();
    let q = phi.view((0, dim), (dim, dim)) * a.transpose();
    Ok((a, q))
}

/// LG-SSM exact-discretization recurrence. Port of _run_lg_ssm
/// (serie/integrators.py):
///
///   x_{k+1} = A·x_k + chol(Q)·ε_k,   ε_k ~ N(0, I_D)
///
/// `unit_normals` has length steps·D, timestep-major: unit_normals[(t-1)·D..t·D]
/// is ε_t (Python draws rng.standard_normal(D) per step — same law).
/// A ParamChange event swaps the drift/diffusion params, so (A, Q, chol(Q))
/// are rebuilt exactly at the event step; the block count is structural and
/// must not change.
fn run_lg_ssm(
    systems: &[LgSsmParams],
    x0: &[f64],
    steps: usize,
    dt: f64,
    unit_normals: &[f64],
    shocks: &[ShockEvent],
) -> Result<Vec<f64>, String> {
    let dim = 2 * systems[0].gammas.len();
    let discretize = |p: &LgSsmParams| -> Result<(DMatrix<f64>, DMatrix<f64>), String> {
        if 2 * p.gammas.len() != dim {
            return Err(format!(
                "lg_ssm param shock must keep the block count: state dim is {dim}, shocked params imply {}",
                2 * p.gammas.len()
            ));
        }
        let (a, q) = lg_ssm_discretize_exact(p, dt)?;
        let chol =
            Cholesky::new(q).ok_or_else(|| "lg_ssm process-noise covariance Q is not positive definite".to_string())?;
        Ok((a, chol.unpack()))
    };
    let (mut a, mut chol_q) = discretize(&systems[0])?;

    let mut traj = vec![0.0f64; (steps + 1) * dim];
    let mut x = DVector::<f64>::from_column_slice(x0);
    traj[..dim].copy_from_slice(x.as_slice());

    let mut shock_idx: usize = 0;
    for t in 1..=steps {
        // Apply all events scheduled for this step.
        while shock_idx < shocks.len() {
            match &shocks[shock_idx] {
                ShockEvent::ParamChange { step, param_set_idx } if *step == t => {
                    let p = &systems[(*param_set_idx).min(systems.len() - 1)];
                    (a, chol_q) = discretize(p)?;
                    shock_idx += 1;
                }
                ShockEvent::StatePerturbation { step, state_idx, eps } if *step == t => {
                    x[*state_idx] += eps;
                    shock_idx += 1;
                }
                _ => break,
            }
        }

        let eps_t = DVector::<f64>::from_column_slice(&unit_normals[(t - 1) * dim..t * dim]);
        x = &a * &x + &chol_q * eps_t;
        traj[t * dim..(t + 1) * dim].copy_from_slice(x.as_slice());
    }
    Ok(traj)
}

fn run_lg_ssm_seeded(
    systems: &[LgSsmParams],
    x0: &[f64],
    steps: usize,
    dt: f64,
    seed: u64,
    shocks: &[ShockEvent],
) -> Result<Vec<f64>, String> {
    let dim = 2 * systems[0].gammas.len();
    let unit_normals = standard_normal_noise(seed, steps, dim);
    run_lg_ssm(systems, x0, steps, dt, &unit_normals, shocks)
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// §7  Trajectory Splicing (switch)
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

/// Generate a "switch" trajectory by splicing two independent integrations.
///
/// This is the Rust equivalent of the Python `_generate_switch`:
///   1. Integrate sys_a from x0_a for `steps` steps  → traj_a
///   2. Integrate sys_b from x0_b for `steps` steps  → traj_b
///   3. Splice:  result[:splice_step+1] = traj_a,  result[splice_step+1:] = traj_b
///
/// Both systems must be the same type S (same-system param switches).
/// Cross-system switches (e.g. Lorenz → Rössler) require Python-side
/// orchestration with two separate Rust calls + numpy splice.
///
/// The two integrations are independent and could be parallelized with
/// rayon in the future. For now, sequential is fine — integration dominates.
fn integrate_rk4_switch<S: OdeSystem>(
    sys_a: &S,
    x0_a: &[f64],
    sys_b: &S,
    x0_b: &[f64],
    steps: usize,
    dt: f64,
    splice_step: usize, // traj_a used for rows [0..=splice_step], traj_b for the rest
) -> Vec<f64> {
    let dim = sys_a.dim();
    // Both full-length integrations (no shocks, no burn_in — caller handles that)
    let traj_a = integrate_rk4(sys_a, x0_a, steps, dt);
    let traj_b = integrate_rk4(sys_b, x0_b, steps, dt);

    // Start from B, overwrite prefix with A
    let mut result = traj_b;
    let copy_end = (splice_step + 1) * dim; // exclusive end
    result[..copy_end].copy_from_slice(&traj_a[..copy_end]);
    result
}

/// Same pattern for discrete maps.
fn iterate_map_switch<S: MapSystem>(
    sys_a: &S,
    x0_a: &[f64],
    sys_b: &S,
    x0_b: &[f64],
    steps: usize,
    splice_step: usize,
) -> Vec<f64> {
    let dim = sys_a.dim();
    let traj_a = iterate_map(sys_a, x0_a, steps);
    let traj_b = iterate_map(sys_b, x0_b, steps);

    let mut result = traj_b;
    let copy_end = (splice_step + 1) * dim;
    result[..copy_end].copy_from_slice(&traj_a[..copy_end]);
    result
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// §8  Dispatch Macros
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
//
// Without macros, each system arm in generate_ode would be 8 identical
// lines differing only in the type name. The macro writes it once.
// macro_rules! is Rust's compile-time code template.
// It's not a function — it's a text substitution that happens before the compiler
// type-checks anything. When you write:
// dispatch_ode!(LorenzParams, &x0, steps, dt, param_matrix, n_params, shocks)
// the compiler literally pastes the body with $sys_type replaced by LorenzParams, $x0 replaced by &x0, etc.
// The result is fully typed, fully optimized code for that specific system. No runtime overhead.

macro_rules! dispatch_ode {
    // $sys_type:ty — captures a type (like LorenzParams).
    // The :ty tells the macro "this argument is a type, not a value."
    // $x0:expr — captures an expression (like &x0).
    // The :expr tells the macro "this is a value."
    ($sys_type:ty, $x0:expr, $steps:expr, $dt:expr,
     $param_matrix:expr, $n_params:expr, $shocks:expr) => {{
        // The double braces {{ }}: Outer braces are macro syntax.
        // Inner braces make a block expression so the whole thing evaluates to a value
        // (the Vec<f64> trajectory).
        let n_sets = $param_matrix.len() / $n_params;
        let systems: Vec<$sys_type> = (0..n_sets)
            //<$sys_type>::from_slice(...) — the angle brackets (<...>) are called "turbofish."
            // They explicitly tell Rust which type's from_slice to call.
            // Needed because $sys_type comes from the macro parameter.
            .map(|i| <$sys_type>::from_slice(&$param_matrix[i * $n_params..(i + 1) * $n_params]))
            /*
            Slice the flat param_matrix into rows of n_params each,
            call from_slice on each row to get typed structs, collect into a Vec.
            If no shocks, call integrate_rk4. If shocks, call integrate_rk4_shocks.
            */
            .collect();
        if $shocks.is_empty() {
            integrate_rk4(&systems[0], $x0, $steps, $dt)
        } else {
            integrate_rk4_shocks(&systems, $x0, $steps, $dt, $shocks)
        }
    }};
}

macro_rules! dispatch_map {
    ($sys_type:ty, $x0:expr, $steps:expr,
     $param_matrix:expr, $n_params:expr) => {{
        let sys = <$sys_type>::from_slice(&$param_matrix[0..$n_params]);
        iterate_map(&sys, $x0, $steps)
    }};
}

macro_rules! dispatch_sde_seeded {
    ($sys_type:ty, $x0:expr, $steps:expr, $dt:expr,
     $sigma_per_regime:expr, $seed:expr,
     $param_matrix:expr, $n_params:expr, $shocks:expr) => {{
        let n_sets = $param_matrix.len() / $n_params;
        let systems: Vec<$sys_type> = (0..n_sets)
            .map(|i| <$sys_type>::from_slice(&$param_matrix[i * $n_params..(i + 1) * $n_params]))
            .collect();
        if $shocks.is_empty() {
            integrate_em_seeded(&systems[0], $x0, $steps, $dt, $sigma_per_regime[0], $seed)
        } else {
            integrate_em_shocks_seeded(&systems, $x0, $steps, $dt, &$sigma_per_regime, $seed, $shocks)
        }
    }};
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// §9  Public Rust Entry Points
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
//
// These entry points keep slice-based param inputs but use native ShockEvent
// lists so stage wiring can stay fully Rust-native.

/// ODE integration: any system, any shocks, one entry point.
pub fn generate_ode(
    system: &str,
    x0: &[f64],
    steps: usize,
    dt: f64,
    shocks: &[ShockEvent],
    param_matrix: &[f64],
    n_params: usize,
) -> Result<Vec<f64>, String> {
    let sorted_shocks = sorted_shocks(shocks);

    let traj = match system {
        "lorenz" => dispatch_ode!(LorenzParams, x0, steps, dt, param_matrix, n_params, &sorted_shocks),
        "rossler" => dispatch_ode!(RosslerParams, x0, steps, dt, param_matrix, n_params, &sorted_shocks),
        "chua" => dispatch_ode!(ChuaParams, x0, steps, dt, param_matrix, n_params, &sorted_shocks),
        "vanderpol" => dispatch_ode!(VanDerPolParams, x0, steps, dt, param_matrix, n_params, &sorted_shocks),
        "duffing" => dispatch_ode!(DuffingParams, x0, steps, dt, param_matrix, n_params, &sorted_shocks),
        "hyper_rossler" => dispatch_ode!(
            HyperRosslerParams,
            x0,
            steps,
            dt,
            param_matrix,
            n_params,
            &sorted_shocks
        ),
        "lorenz96" => {
            let dim = x0.len();
            let n_sets = param_matrix.len() / n_params;
            let systems: Vec<Lorenz96Params> = (0..n_sets)
                .map(|i| Lorenz96Params::from_slice(&param_matrix[i * n_params..(i + 1) * n_params], dim))
                .collect();
            if sorted_shocks.is_empty() {
                integrate_rk4(&systems[0], x0, steps, dt)
            } else {
                integrate_rk4_shocks(&systems, x0, steps, dt, &sorted_shocks)
            }
        }
        _ => return Err(format!("Unknown ODE system: {}", system)),
    };
    Ok(traj)
}

/// Map iteration: any map system, one entry point.
pub fn generate_map(
    system: &str,
    x0: &[f64],
    steps: usize,
    param_matrix: &[f64],
    n_params: usize,
) -> Result<Vec<f64>, String> {
    let traj = match system {
        "henon" => dispatch_map!(HenonParams, x0, steps, param_matrix, n_params),
        "logistic" => dispatch_map!(LogisticParams, x0, steps, param_matrix, n_params),
        _ => return Err(format!("Unknown map system: {}", system)),
    };
    Ok(traj)
}

/// ODE switch (trajectory splicing): two param sets, two ICs, one splice point.
pub fn generate_ode_switch(
    system: &str,
    x0_a: &[f64],
    x0_b: &[f64],
    steps: usize,
    dt: f64,
    splice_step: usize,
    params_a: &[f64],
    params_b: &[f64],
) -> Result<Vec<f64>, String> {
    if splice_step > steps {
        return Err("splice_step must be <= steps".to_string());
    }

    macro_rules! sw {
        ($T:ty) => {{
            let a = <$T>::from_slice(params_a);
            let b = <$T>::from_slice(params_b);
            integrate_rk4_switch(&a, x0_a, &b, x0_b, steps, dt, splice_step)
        }};
    }

    let traj = match system {
        "lorenz" => sw!(LorenzParams),
        "rossler" => sw!(RosslerParams),
        "chua" => sw!(ChuaParams),
        "vanderpol" => sw!(VanDerPolParams),
        "duffing" => sw!(DuffingParams),
        "hyper_rossler" => sw!(HyperRosslerParams),
        "lorenz96" => {
            let dim = x0_a.len();
            let a = Lorenz96Params::from_slice(params_a, dim);
            let b = Lorenz96Params::from_slice(params_b, dim);
            integrate_rk4_switch(&a, x0_a, &b, x0_b, steps, dt, splice_step)
        }
        _ => return Err(format!("Unknown ODE system: {}", system)),
    };
    Ok(traj)
}

/// Map switch (trajectory splicing).
pub fn generate_map_switch(
    system: &str,
    x0_a: &[f64],
    x0_b: &[f64],
    steps: usize,
    splice_step: usize,
    params_a: &[f64],
    params_b: &[f64],
) -> Result<Vec<f64>, String> {
    if splice_step > steps {
        return Err("splice_step must be <= steps".to_string());
    }
    macro_rules! sw {
        ($T:ty) => {{
            let a = <$T>::from_slice(params_a);
            let b = <$T>::from_slice(params_b);
            iterate_map_switch(&a, x0_a, &b, x0_b, steps, splice_step)
        }};
    }
    let traj = match system {
        "henon" => sw!(HenonParams),
        "logistic" => sw!(LogisticParams),
        _ => return Err(format!("Unknown map system: {}", system)),
    };
    Ok(traj)
}

/// SDE Euler-Maruyama integration using Rust-owned RNG from a seed.
pub fn generate_sde_seeded(
    system: &str,
    x0: &[f64],
    steps: usize,
    dt: f64,
    sigma_per_regime: &[f64],
    seed: u64,
    shocks: &[ShockEvent],
    param_matrix: &[f64],
    n_params: usize,
) -> Result<Vec<f64>, String> {
    let sorted_shocks = sorted_shocks(shocks);

    let traj = match system {
        "ou" => dispatch_sde_seeded!(
            OuParams,
            x0,
            steps,
            dt,
            sigma_per_regime,
            seed,
            param_matrix,
            n_params,
            &sorted_shocks
        ),
        "doublewell" => dispatch_sde_seeded!(
            DoubleWellParams,
            x0,
            steps,
            dt,
            sigma_per_regime,
            seed,
            param_matrix,
            n_params,
            &sorted_shocks
        ),
        _ => return Err(format!("Unknown SDE system: {}", system)),
    };
    Ok(traj)
}

/// SDE switch (trajectory splicing) using Rust-owned RNG from a seed.
///
/// Mirrors the Python _generate_switch_python_pure (serie/integrators.py):
/// run A and B as independent full-length integrations — A with `seed`,
/// B with `seed + 1` — then splice: rows [0..=splice_step] from A, the
/// rest from B. B's params carry their own sigma_noise, so the diffusion
/// coefficient switches with the params (`sigma_b`).
pub fn generate_sde_switch_seeded(
    system: &str,
    x0_a: &[f64],
    x0_b: &[f64],
    steps: usize,
    dt: f64,
    sigma_a: f64,
    sigma_b: f64,
    splice_step: usize,
    seed: u64,
    params_a: &[f64],
    params_b: &[f64],
) -> Result<Vec<f64>, String> {
    if splice_step > steps {
        return Err("splice_step must be <= steps".to_string());
    }

    macro_rules! sw {
        ($T:ty) => {{
            let a = <$T>::from_slice(params_a);
            let b = <$T>::from_slice(params_b);
            let dim = a.dim();
            let traj_a = integrate_em_seeded(&a, x0_a, steps, dt, sigma_a, seed);
            // Python gives run B `seed + 1` (SwitchSpec: "B gets seed + 1").
            let traj_b = integrate_em_seeded(&b, x0_b, steps, dt, sigma_b, seed.wrapping_add(1));
            let mut result = traj_b;
            let copy_end = (splice_step + 1) * dim;
            result[..copy_end].copy_from_slice(&traj_a[..copy_end]);
            result
        }};
    }

    let traj = match system {
        "ou" => sw!(OuParams),
        "doublewell" => sw!(DoubleWellParams),
        _ => return Err(format!("Unknown SDE system: {}", system)),
    };
    Ok(traj)
}

/// Discrete statistical simulation (SLDS / Seasonal AR / GARCH / LG-SSM)
/// using Rust-owned RNG from a seed. Mirrors _generate_discrete_pure
/// (serie/integrators.py) string dispatch.
///
/// `dt` is the real sampling interval for lg_ssm (exact discretization); the
/// index-stepped simulators (slds/seasonal_ar/garch) ignore it — same
/// signature split as the Python _generate_discrete_pure.
pub fn generate_discrete_seeded(
    system: &str,
    x0: &[f64],
    steps: usize,
    dt: f64,
    seed: u64,
    shocks: &[ShockEvent],
    param_matrix: &[f64],
    n_params: usize,
) -> Result<Vec<f64>, String> {
    let sorted_shocks = sorted_shocks(shocks);

    macro_rules! param_sets {
        ($sys_type:ty) => {{
            let n_sets = param_matrix.len() / n_params;
            (0..n_sets)
                .map(|i| <$sys_type>::from_slice(&param_matrix[i * n_params..(i + 1) * n_params]))
                .collect::<Vec<$sys_type>>()
        }};
    }

    let traj = match system {
        "slds" => run_slds_seeded(&param_sets!(SldsParams), x0, steps, seed, &sorted_shocks),
        "seasonal_ar" => run_seasonal_ar_seeded(&param_sets!(SeasonalArParams), x0, steps, seed, &sorted_shocks),
        "garch" => run_garch_seeded(&param_sets!(GarchParams), x0, steps, seed, &sorted_shocks),
        "lg_ssm" => run_lg_ssm_seeded(&param_sets!(LgSsmParams), x0, steps, dt, seed, &sorted_shocks)?,
        _ => return Err(format!("Unknown discrete system: {}", system)),
    };
    Ok(traj)
}

/// Discrete switch (trajectory splicing): run A with `seed`, run B with
/// `seed + 1`, splice rows [0..=splice_step] from A onto B — the same
/// composition _generate_switch_python_pure applies to every system kind.
pub fn generate_discrete_switch_seeded(
    system: &str,
    x0_a: &[f64],
    x0_b: &[f64],
    steps: usize,
    dt: f64,
    splice_step: usize,
    seed: u64,
    params_a: &[f64],
    params_b: &[f64],
) -> Result<Vec<f64>, String> {
    if splice_step > steps {
        return Err("splice_step must be <= steps".to_string());
    }

    let traj_a = generate_discrete_seeded(system, x0_a, steps, dt, seed, &[], params_a, params_a.len())?;
    let traj_b = generate_discrete_seeded(
        system,
        x0_b,
        steps,
        dt,
        seed.wrapping_add(1),
        &[],
        params_b,
        params_b.len(),
    )?;

    let dim = traj_a.len() / (steps + 1);
    let mut result = traj_b;
    let copy_end = (splice_step + 1) * dim;
    result[..copy_end].copy_from_slice(&traj_a[..copy_end]);
    Ok(result)
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// §10  Tests
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

/// Tests for the ODE/SDE/map integrator and shock schedule.
///
/// Why these tests exist:
/// - Shape + finiteness tests are smoke tests: if the integrator diverges or
///   returns the wrong number of rows, everything downstream is wrong.
/// - Shock tests verify causal isolation: pre-shock trajectory must be
///   identical to the no-shock case (the shock must not leak backward in time).
/// - OU/DoubleWell exact-Euler tests verify the discretization formula against
///   hand-computed steps — if the Euler-Maruyama step is wrong, the generated
///   trajectories do not match the SDE they claim to represent.
/// - Seed determinism tests ensure reproducibility: same seed → identical output.
/// - Splice tests verify that trajectory_switch produces the correct composite
///   by comparing against independently integrated segments.
#[cfg(test)]
mod tests {
    use super::*;

    // ── §11a  ODE tests ─────────────────────────────────────────────

    #[test]
    fn test_lorenz_no_shock() {
        let sys = LorenzParams {
            sigma: 10.0,
            rho: 28.0,
            beta: 8.0 / 3.0,
        };
        let x0 = vec![1.0, 0.98, 1.1];
        let traj = integrate_rk4(&sys, &x0, 100, 0.01);
        assert_eq!(traj.len(), 101 * 3);
        assert!(traj.iter().all(|v| v.is_finite()));
        assert!((traj[0] - 1.0).abs() < 1e-15);
    }

    #[test]
    fn test_lorenz_param_shock() {
        let before = LorenzParams {
            sigma: 10.0,
            rho: 28.0,
            beta: 8.0 / 3.0,
        };
        let after = LorenzParams {
            sigma: 10.0,
            rho: 35.0,
            beta: 8.0 / 3.0,
        };
        let shocks = vec![ShockEvent::ParamChange {
            step: 50,
            param_set_idx: 1,
        }];
        let traj = integrate_rk4_shocks(&[before, after], &[1.0, 0.98, 1.1], 100, 0.01, &shocks);
        assert_eq!(traj.len(), 101 * 3);
        assert!(traj.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn test_lorenz_state_shock() {
        let sys = LorenzParams {
            sigma: 10.0,
            rho: 28.0,
            beta: 8.0 / 3.0,
        };
        let shocks = vec![ShockEvent::StatePerturbation {
            step: 50,
            state_idx: 0,
            eps: 5.0,
        }];
        let traj = integrate_rk4_shocks(&[sys], &[1.0, 0.98, 1.1], 100, 0.01, &shocks);
        assert_eq!(traj.len(), 101 * 3);
        assert!(traj.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn test_lorenz_switch() {
        let sys_a = LorenzParams {
            sigma: 10.0,
            rho: 28.0,
            beta: 8.0 / 3.0,
        };
        let sys_b = LorenzParams {
            sigma: 10.0,
            rho: 35.0,
            beta: 8.0 / 3.0,
        };
        let x0_a = vec![1.0, 0.98, 1.1];
        let x0_b = vec![0.5, 0.5, 0.5];
        let switched = integrate_rk4_switch(&sys_a, &x0_a, &sys_b, &x0_b, 1000, 0.01, 500);
        let traj_a = integrate_rk4(&sys_a, &x0_a, 1000, 0.01);
        let traj_b = integrate_rk4(&sys_b, &x0_b, 1000, 0.01);
        let cut = 501 * 3;
        assert_eq!(&switched[..cut], &traj_a[..cut]);
        assert_eq!(&switched[cut..], &traj_b[cut..]);
    }

    #[test]
    fn test_from_slice_roundtrip() {
        let p = LorenzParams {
            sigma: 10.0,
            rho: 28.0,
            beta: 2.667,
        };
        let flat = vec![10.0, 28.0, 2.667];
        let q = LorenzParams::from_slice(&flat);
        assert_eq!(p.sigma, q.sigma);
        assert_eq!(p.rho, q.rho);
        assert_eq!(p.beta, q.beta);
    }

    #[test]
    fn test_sorted_shocks() {
        let input = vec![
            ShockEvent::StatePerturbation {
                step: 200,
                state_idx: 1,
                eps: 5.0,
            },
            ShockEvent::ParamChange {
                step: 100,
                param_set_idx: 2,
            },
        ];
        let shocks = sorted_shocks(&input);
        assert_eq!(shocks.len(), 2);
        match &shocks[0] {
            ShockEvent::ParamChange { step, param_set_idx } => {
                assert_eq!(*step, 100);
                assert_eq!(*param_set_idx, 2);
            }
            _ => panic!("expected ParamChange"),
        }
        match &shocks[1] {
            ShockEvent::StatePerturbation { step, state_idx, eps } => {
                assert_eq!(*step, 200);
                assert_eq!(*state_idx, 1);
                assert_eq!(*eps, 5.0);
            }
            _ => panic!("expected StatePerturbation"),
        }
    }

    #[test]
    fn test_lorenz_param_shock_uses_explicit_param_set_idx() {
        let before = LorenzParams {
            sigma: 10.0,
            rho: 28.0,
            beta: 8.0 / 3.0,
        };
        let skipped = LorenzParams {
            sigma: 10.0,
            rho: 31.0,
            beta: 8.0 / 3.0,
        };
        let target = LorenzParams {
            sigma: 10.0,
            rho: 40.0,
            beta: 8.0 / 3.0,
        };
        let x0 = [1.0, 0.98, 1.1];
        let shock_step = 50;

        let shocks = vec![ShockEvent::ParamChange {
            step: shock_step,
            param_set_idx: 2,
        }];
        let traj = integrate_rk4_shocks(&[before.clone(), skipped, target.clone()], &x0, 100, 0.01, &shocks);

        let expected_shocks = vec![ShockEvent::ParamChange {
            step: shock_step,
            param_set_idx: 1,
        }];
        let expected = integrate_rk4_shocks(&[before, target], &x0, 100, 0.01, &expected_shocks);

        assert_eq!(traj, expected);
    }

    // ── §11b  Map tests ─────────────────────────────────────────────

    #[test]
    fn test_henon_basic() {
        let sys = HenonParams { a: 1.4, b: 0.3 };
        let traj = iterate_map(&sys, &[0.1, 0.1], 1000);
        assert_eq!(traj.len(), 1001 * 2);
        assert!(traj.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn test_logistic_basic() {
        let sys = LogisticParams { r: 3.9 };
        let traj = iterate_map(&sys, &[0.5], 100);
        assert_eq!(traj.len(), 101);
        for i in 0..101 {
            assert!(traj[i] >= 0.0 && traj[i] <= 1.0);
        }
    }

    // ── §11c  SDE tests (Euler-Maruyama) ────────────────────────────

    #[test]
    fn test_ou_zero_noise_decay() {
        // With zero noise, OU is purely deterministic mean-reversion:
        //   x_{n+1} = x_n + θ(μ - x_n)*dt
        // Starting at x=1.0 with μ=0.0, trajectory should decay toward 0.
        let steps = 100;
        let noise = vec![0.0; steps]; // zero noise → deterministic
        let sys = OuParams {
            theta: 0.2,
            mu: 0.0,
            sigma_noise: 0.3,
        };
        let traj = integrate_em(&sys, &[1.0], steps, 0.01, 0.3, &noise);

        assert_eq!(traj.len(), steps + 1);
        assert!((traj[0] - 1.0).abs() < 1e-15, "IC should be exactly 1.0");
        assert!(traj[steps].abs() < traj[0].abs(), "OU should decay toward mu=0");
        assert!(traj[steps].is_finite());
    }

    #[test]
    fn test_ou_zero_noise_exact_euler() {
        // Verify each EM step exactly matches manual Euler:
        //   x += theta * (mu - x) * dt
        let theta = 0.2;
        let mu = 5.0;
        let dt = 0.01;
        let steps = 50;
        let noise = vec![0.0; steps];
        let sys = OuParams {
            theta,
            mu,
            sigma_noise: 0.3,
        };
        let traj = integrate_em(&sys, &[1.0], steps, dt, 0.3, &noise);

        let mut x = 1.0;
        assert!((traj[0] - x).abs() < 1e-15);
        for i in 0..steps {
            x += theta * (mu - x) * dt;
            assert!(
                (traj[i + 1] - x).abs() < 1e-14,
                "OU step {}: expected {:.15e}, got {:.15e}",
                i,
                x,
                traj[i + 1]
            );
        }
    }

    #[test]
    fn test_ou_with_noise_stays_finite() {
        // Deterministic fake noise (sin-based) — OU should remain stable
        let steps = 10_000;
        let noise: Vec<f64> = (0..steps).map(|i| ((i as f64) * 0.1).sin()).collect();
        let sys = OuParams {
            theta: 0.2,
            mu: 0.0,
            sigma_noise: 0.3,
        };
        let traj = integrate_em(&sys, &[0.0], steps, 0.01, 0.3, &noise);

        assert_eq!(traj.len(), steps + 1);
        assert!(traj.iter().all(|v| v.is_finite()));
        let max_abs = traj.iter().map(|v| v.abs()).fold(0.0f64, f64::max);
        assert!(max_abs < 50.0, "OU wandered implausibly far: {}", max_abs);
    }

    #[test]
    fn test_doublewell_zero_noise_exact_euler() {
        // Verify: x += (-x³ + a*x) * dt
        let a = 1.0;
        let dt = 0.001;
        let steps = 20;
        let noise = vec![0.0; steps];
        let sys = DoubleWellParams { a, sigma_noise: 0.5 };
        let traj = integrate_em(&sys, &[0.5], steps, dt, 0.5, &noise);

        let mut x = 0.5;
        assert!((traj[0] - x).abs() < 1e-15);
        for i in 0..steps {
            x += (-(x * x * x) + a * x) * dt;
            assert!(
                (traj[i + 1] - x).abs() < 1e-13,
                "DW step {}: expected {:.15e}, got {:.15e}",
                i,
                x,
                traj[i + 1]
            );
        }
    }

    #[test]
    fn test_doublewell_zero_noise_toward_well() {
        // Start at x=0.5 with a=1.0, zero noise.
        // Drift = -x³ + x = -0.125 + 0.5 = 0.375 > 0 → moves toward well at +1.
        let steps = 1000;
        let noise = vec![0.0; steps];
        let sys = DoubleWellParams {
            a: 1.0,
            sigma_noise: 0.5,
        };
        let traj = integrate_em(&sys, &[0.5], steps, 0.01, 0.5, &noise);

        assert!(traj[steps] > 0.5, "Should drift toward well at +√a = +1");
        assert!(traj[steps] < 1.1, "Should settle near well, not diverge");
    }

    #[test]
    fn test_ou_param_shock() {
        // Shock at step 50: switch from theta=0.2 to theta=0.8.
        // Pre-shock must be bit-identical to no-shock baseline.
        let steps = 100;
        let noise: Vec<f64> = (0..steps).map(|i| ((i as f64) * 0.3).sin()).collect();
        let shock_step = 50;

        // No-shock baseline
        let sys_before = OuParams {
            theta: 0.2,
            mu: 0.0,
            sigma_noise: 0.3,
        };
        let traj_base = integrate_em(&sys_before, &[1.0], steps, 0.01, 0.3, &noise);

        // With param shock
        let sys_after = OuParams {
            theta: 0.8,
            mu: 0.0,
            sigma_noise: 0.6,
        };
        let shocks = vec![ShockEvent::ParamChange {
            step: shock_step,
            param_set_idx: 1,
        }];
        let traj_shock = integrate_em_shocks(
            &[sys_before.clone(), sys_after],
            &[1.0],
            steps,
            0.01,
            &[0.3, 0.6], // sigma_per_regime: before=0.3, after=0.6
            &noise,
            &shocks,
        );

        assert_eq!(traj_shock.len(), traj_base.len());

        // Pre-shock: bit-identical
        for i in 0..shock_step {
            assert!(
                (traj_shock[i] - traj_base[i]).abs() < 1e-15,
                "Pre-shock mismatch at index {}",
                i
            );
        }

        // Post-shock: must differ (different theta and sigma)
        let mut any_differ = false;
        for i in (shock_step + 2)..=steps {
            if (traj_shock[i] - traj_base[i]).abs() > 1e-14 {
                any_differ = true;
                break;
            }
        }
        assert!(any_differ, "Post-shock trajectory should differ from baseline");
    }

    #[test]
    fn test_ou_state_shock() {
        // State perturbation at step 50: x[0] += 5.0
        let steps = 100;
        let noise = vec![0.0; steps]; // zero noise for deterministic check

        let sys = OuParams {
            theta: 0.2,
            mu: 0.0,
            sigma_noise: 0.3,
        };
        let shocks = vec![ShockEvent::StatePerturbation {
            step: 50,
            state_idx: 0,
            eps: 5.0,
        }];
        let traj = integrate_em_shocks(&[sys], &[0.0], steps, 0.01, &[0.3], &noise, &shocks);

        // Before shock: x0=0, mu=0, drift=0, zero noise → stays at 0
        assert!(traj[49].abs() < 0.01, "Pre-shock should be near 0");

        // After shock: should jump by ~5.0, then decay back toward 0
        assert!(
            traj[50].abs() > 4.0,
            "State should jump at shock step, got {}",
            traj[50]
        );
    }

    #[test]
    fn test_em_noise_consumed_correctly() {
        // Verify that noise is applied at the right steps.
        // With zero drift (theta=0, mu=0), x_{n+1} = x_n + σ*√dt*ε_n
        // so traj[i] = x0 + σ*√dt * sum(ε_0..ε_{i-1})
        let steps = 5;
        let dt = 0.04; // √dt = 0.2
        let sigma = 0.5;
        let noise = vec![1.0, -1.0, 2.0, 0.0, -0.5]; // known sequence
        let sys = OuParams {
            theta: 0.0,
            mu: 0.0,
            sigma_noise: sigma,
        }; // zero drift
        let traj = integrate_em(&sys, &[0.0], steps, dt, sigma, &noise);

        let sqrt_dt = dt.sqrt(); // 0.2
        let _inc = sigma * sqrt_dt; // 0.1
        let expected = [0.0, 0.1, 0.0, 0.2, 0.2, 0.15]; // cumulative: 0, +0.1, -0.1, +0.2, 0, -0.05
        for i in 0..=steps {
            assert!(
                (traj[i] - expected[i]).abs() < 1e-14,
                "Step {}: expected {}, got {}",
                i,
                expected[i],
                traj[i]
            );
        }
    }

    // ── §11d  Discrete statistical tests (SLDS, Seasonal AR, GARCH) ──

    #[test]
    fn test_slds_hand_computed_steps() {
        // Hand-computed from the _run_slds recurrence, including the regime
        // timing convention (user-approved 2026-07-01): the transition draw
        // at step t updates z, and the dynamics at step t use the NEWLY
        // drawn regime's (A, Q) — a switch drawn at step t drives the
        // dynamics at step t itself.
        let sys = SldsParams {
            a1: 0.5,
            q1: 0.04, // sqrt = 0.2
            a2: 2.0,
            q2: 0.25, // sqrt = 0.5
            p11: 0.9,
            p22: 0.8,
        };
        // uniforms[0]=0.3 < 0.5 → initial z=1
        // t=1: z=1, u=0.95 ≥ p11 → z→2, dynamics use (A2,Q2):     x = 2.0·1.0 + 0.5·1.0  = 2.5
        // t=2: z=2, u=0.50 < p22 → z stays 2, dynamics (A2,Q2):   x = 2.0·2.5 + 0.5·(−1) = 4.5
        // t=3: z=2, u=0.85 ≥ p22 → z→1, dynamics use (A1,Q1):     x = 0.5·4.5 + 0.2·2.0  = 2.65
        let uniforms = vec![0.3, 0.95, 0.5, 0.85];
        let normals = vec![1.0, -1.0, 2.0];
        let traj = run_slds(&[sys], &[1.0], 3, &uniforms, &normals, &[]);

        let expected = [1.0, 2.5, 4.5, 2.65];
        for i in 0..=3 {
            assert!(
                (traj[i] - expected[i]).abs() < 1e-12,
                "SLDS step {}: expected {:.15e}, got {:.15e}",
                i,
                expected[i],
                traj[i]
            );
        }
    }

    #[test]
    fn test_slds_initial_regime_from_first_uniform() {
        // uniforms[0] ≥ 0.5 → initial z=2; at t=1 the transition draw
        // u=0.5 < p22 keeps z=2, so the step uses (A2, Q2).
        let sys = SldsParams {
            a1: 0.5,
            q1: 0.04,
            a2: 2.0,
            q2: 0.25,
            p11: 0.9,
            p22: 0.8,
        };
        let uniforms = vec![0.7, 0.5];
        let normals = vec![1.0];
        let traj = run_slds(&[sys], &[1.0], 1, &uniforms, &normals, &[]);
        // x = 2.0·1.0 + 0.5·1.0 = 2.5
        assert!((traj[1] - 2.5).abs() < 1e-12);
    }

    #[test]
    fn test_slds_switch_drawn_at_step_t_drives_step_t() {
        // Regime-timing pin for the user-approved 2026-07-01 convention:
        // force a 1→2 transition at exactly t=1 (u ≥ p11) and check the very
        // first step already uses (A2, Q2), not the pre-transition (A1, Q1).
        let sys = SldsParams {
            a1: 0.5,
            q1: 0.04, // sqrt = 0.2
            a2: 2.0,
            q2: 0.25, // sqrt = 0.5
            p11: 0.9,
            p22: 0.8,
        };
        let uniforms = vec![0.0, 0.95]; // z starts 1, transition fires at t=1
        let normals = vec![1.0];
        let traj = run_slds(&[sys], &[1.0], 1, &uniforms, &normals, &[]);
        // New convention: x = A2·1.0 + √Q2·1.0 = 2.5 (old convention gave 0.7).
        assert!((traj[1] - 2.5).abs() < 1e-12);
    }

    #[test]
    fn test_slds_param_shock_lands_at_exact_step() {
        // Uniforms all 0.0 → z=1 forever (0 < p11), so dynamics are the
        // deterministic AR(1) x ← A1·x + √Q1·ε with injected ε.
        let before = SldsParams {
            a1: 0.5,
            q1: 0.04,
            a2: 0.9,
            q2: 0.1,
            p11: 0.9,
            p22: 0.9,
        };
        let after = SldsParams {
            a1: 0.25,
            q1: 0.16, // sqrt = 0.4
            a2: 0.9,
            q2: 0.1,
            p11: 0.9,
            p22: 0.9,
        };
        let steps = 5;
        let uniforms = vec![0.0; steps + 1];
        let normals = vec![1.0; steps];
        let shock_step = 3;
        let shocks = vec![ShockEvent::ParamChange {
            step: shock_step,
            param_set_idx: 1,
        }];

        let base = run_slds(&[before.clone()], &[1.0], steps, &uniforms, &normals, &[]);
        let shocked = run_slds(&[before, after], &[1.0], steps, &uniforms, &normals, &shocks);

        // Rows before the shock step are identical.
        for i in 0..shock_step {
            assert_eq!(shocked[i], base[i], "pre-shock row {} must match baseline", i);
        }
        // The shock step itself already uses the new params:
        // x3 = 0.25·x2 + 0.4·1.0 with x2 from the baseline recurrence.
        let expected_row3 = 0.25 * base[2] + 0.4;
        assert!((shocked[shock_step] - expected_row3).abs() < 1e-12);
    }

    #[test]
    fn test_seasonal_ar_hand_computed_steps() {
        // Replicate the _run_seasonal_ar recurrence in scalar form:
        //   a(t) = a0 + amp_drift·t
        //   y ← a(t)·sin(2πt/S) + φ·y + σ·ε_t
        let sys = SeasonalArParams {
            s: 4.0,
            phi: 0.5,
            sigma: 0.1,
            a0: 2.0,
            amp_drift: 0.01,
        };
        let steps = 3;
        let normals = vec![1.0, -2.0, 0.5];
        let traj = run_seasonal_ar(&[sys.clone()], &[0.5], steps, &normals, &[]);

        let mut y = 0.5;
        assert_eq!(traj[0], y);
        for t in 1..=steps {
            let t_f = t as f64;
            let a_t = sys.a0 + sys.amp_drift * t_f;
            let seasonal = a_t * (2.0 * std::f64::consts::PI * t_f / sys.s).sin();
            y = seasonal + sys.phi * y + sys.sigma * normals[t - 1];
            assert!(
                (traj[t] - y).abs() < 1e-14,
                "SeasonalAR step {}: expected {:.15e}, got {:.15e}",
                t,
                y,
                traj[t]
            );
        }
    }

    #[test]
    fn test_seasonal_ar_noise_free_periodicity() {
        // With φ=0, σ=0, amp_drift=0 the recurrence is y_t = a0·sin(2πt/S),
        // so the output repeats with period S.
        let s = 24usize;
        let sys = SeasonalArParams {
            s: s as f64,
            phi: 0.0,
            sigma: 0.0,
            a0: 1.0,
            amp_drift: 0.0,
        };
        let steps = 3 * s;
        let normals = vec![0.0; steps];
        let traj = run_seasonal_ar(&[sys], &[0.0], steps, &normals, &[]);
        for t in 1..=(steps - s) {
            assert!(
                (traj[t] - traj[t + s]).abs() < 1e-9,
                "period-{} signature broken at t={}",
                s,
                t
            );
        }
    }

    #[test]
    fn test_seasonal_ar_state_kick_lands_at_exact_step() {
        let sys = SeasonalArParams {
            s: 4.0,
            phi: 0.5,
            sigma: 0.0,
            a0: 1.0,
            amp_drift: 0.0,
        };
        let steps = 6;
        let normals = vec![0.0; steps];
        let kick_step = 4;
        let shocks = vec![ShockEvent::StatePerturbation {
            step: kick_step,
            state_idx: 0,
            eps: 3.0,
        }];
        let base = run_seasonal_ar(&[sys.clone()], &[0.0], steps, &normals, &[]);
        let kicked = run_seasonal_ar(&[sys], &[0.0], steps, &normals, &shocks);

        for i in 0..kick_step {
            assert_eq!(kicked[i], base[i], "pre-kick row {} must match baseline", i);
        }
        // Kick is applied to y BEFORE the step-t update, so row 4 carries φ·(y3+3):
        // kicked[4] − base[4] = φ·3.0
        assert!((kicked[kick_step] - base[kick_step] - 0.5 * 3.0).abs() < 1e-12);
    }

    #[test]
    fn test_garch_hand_computed_steps() {
        // Hand-computed from the _run_garch recurrence:
        //   σ²_t = ω + α·r²_{t-1} + β·σ²_{t-1};  r_t = √σ²_t · ε_t
        let sys = GarchParams {
            omega: 0.2,
            alpha: 0.1,
            beta: 0.5,
        };
        let steps = 2;
        let normals = vec![2.0, -1.0];
        // x0 = [r_0, σ²_0]
        let traj = run_garch(&[sys], &[1.0, 0.5], steps, &normals, &[]);

        // t=1: σ² = 0.2 + 0.1·1² + 0.5·0.5 = 0.55;  r = √0.55·2
        let sigma2_1: f64 = 0.2 + 0.1 * 1.0 + 0.5 * 0.5;
        let r_1 = sigma2_1.sqrt() * 2.0;
        // t=2: σ² = 0.2 + 0.1·r_1² + 0.5·0.55;      r = √σ²·(−1)
        let sigma2_2: f64 = 0.2 + 0.1 * (r_1 * r_1) + 0.5 * sigma2_1;
        let r_2 = sigma2_2.sqrt() * -1.0;

        let expected = [1.0, 0.5, r_1, sigma2_1, r_2, sigma2_2];
        for i in 0..expected.len() {
            assert!(
                (traj[i] - expected[i]).abs() < 1e-14,
                "GARCH flat index {}: expected {:.15e}, got {:.15e}",
                i,
                expected[i],
                traj[i]
            );
        }
    }

    #[test]
    fn test_garch_param_shock_lands_at_exact_step() {
        // Zero normals → r_t = 0 for all t, so the σ² channel is the
        // deterministic recursion σ² ← ω + β·σ², making the regime change
        // exactly visible at the shock row.
        let before = GarchParams {
            omega: 0.2,
            alpha: 0.1,
            beta: 0.5,
        };
        let after = GarchParams {
            omega: 0.4,
            alpha: 0.2,
            beta: 0.25,
        };
        let steps = 5;
        let normals = vec![0.0; steps];
        let shock_step = 3;
        let shocks = vec![ShockEvent::ParamChange {
            step: shock_step,
            param_set_idx: 1,
        }];
        let base = run_garch(&[before.clone()], &[0.0, 1.0], steps, &normals, &[]);
        let shocked = run_garch(&[before, after], &[0.0, 1.0], steps, &normals, &shocks);

        // σ² channel identical before the shock step.
        for t in 0..shock_step {
            assert_eq!(shocked[t * 2 + 1], base[t * 2 + 1], "pre-shock σ² row {} must match", t);
        }
        // At the shock step: σ²_3 = ω' + β'·σ²_2 (baseline σ²_2, r_prev = 0).
        let expected_sigma2_3 = 0.4 + 0.25 * base[2 * 2 + 1];
        assert!((shocked[shock_step * 2 + 1] - expected_sigma2_3).abs() < 1e-14);
    }

    #[test]
    fn test_garch_state_perturbation_edits_stored_previous_return() {
        // Python _run_garch applies StatePerturbEvent to traj[t-1, 0]: the
        // stored previous return moves, shows up in output row t-1, and
        // feeds the σ² recursion.
        let sys = GarchParams {
            omega: 0.2,
            alpha: 0.1,
            beta: 0.5,
        };
        let steps = 3;
        let normals = vec![0.0; steps];
        let shocks = vec![ShockEvent::StatePerturbation {
            step: 2,
            state_idx: 0,
            eps: 4.0,
        }];
        let traj = run_garch(&[sys], &[0.0, 1.0], steps, &normals, &shocks);

        // Row 1 return was 0 (zero normals) then got +4.0 at step 2.
        assert_eq!(traj[2], 4.0); // flat index: row 1, channel 0

        // σ²_2 = 0.2 + 0.1·4² + 0.5·σ²_1 with σ²_1 = 0.2 + 0.5·1.0 = 0.7
        let expected_sigma2_2 = 0.2 + 0.1 * 16.0 + 0.5 * 0.7;
        assert!((traj[2 * 2 + 1] - expected_sigma2_2).abs() < 1e-14);
    }

    #[test]
    fn test_garch_unconditional_variance_matches_omega_over_one_minus_alpha_beta() {
        // Stationary GARCH(1,1): E[r²] = ω/(1−α−β). Default preset values
        // give 0.01/0.04 = 0.25. Long-run sample variance must approach it.
        let sys = GarchParams {
            omega: 0.01,
            alpha: 0.06,
            beta: 0.90,
        };
        let steps = 50_000;
        let traj =
            generate_discrete_seeded("garch", &[0.0, 0.25], steps, 1.0, 42, &[], &[0.01, 0.06, 0.90], 3).unwrap();
        let mean_r2: f64 = (1..=steps).map(|t| traj[t * 2] * traj[t * 2]).sum::<f64>() / steps as f64;
        let expected = sys.omega / (1.0 - sys.alpha - sys.beta);
        assert!(
            (mean_r2 - expected).abs() < 0.1 * expected,
            "GARCH sample E[r²] = {}, expected ≈ {}",
            mean_r2,
            expected
        );
    }

    #[test]
    fn test_discrete_seeded_determinism_and_seed_sensitivity() {
        let cases: [(&str, Vec<f64>, Vec<f64>); 3] = [
            ("slds", vec![0.0], vec![0.9, 0.05, 0.98, 0.35, 0.94, 0.95]),
            ("seasonal_ar", vec![0.0], vec![24.0, 0.5, 0.2, 1.0, 0.0]),
            ("garch", vec![0.0, 0.25], vec![0.01, 0.06, 0.90]),
        ];
        for (system, x0, params) in &cases {
            let a = generate_discrete_seeded(system, x0, 200, 1.0, 7, &[], params, params.len()).unwrap();
            let b = generate_discrete_seeded(system, x0, 200, 1.0, 7, &[], params, params.len()).unwrap();
            let c = generate_discrete_seeded(system, x0, 200, 1.0, 8, &[], params, params.len()).unwrap();
            assert_eq!(a, b, "{}: same seed must reproduce identical output", system);
            assert_ne!(a, c, "{}: different seed must change the output", system);
        }
    }

    // ── §11d (cont.)  LG-SSM tests (matrix OU, exact discretization) ──
    //
    // Mirrors TestLGSSM in serie/tests/test_gen.py: the same scientific
    // checks (a)–(c), determinism, and a cross-language structural parity
    // pin on (A, Q). RNG streams differ across engines by design, so the
    // parity target is the deterministic linear algebra, not the samples.

    /// lg_ssm_small preset numbers (serie/catalog.py lg_ssm_small_spec):
    /// periods 12/24 steps at dt = 1.0, γ = -ln(0.95), q_c = 2γ.
    fn lg_ssm_small_params(kappa_v: f64) -> LgSsmParams {
        lg_ssm_params_for(&[12.0, 24.0], 0.95, kappa_v)
    }

    fn lg_ssm_params_for(periods: &[f64], spectral_radius: f64, kappa_v: f64) -> LgSsmParams {
        let gamma = -spectral_radius.ln();
        LgSsmParams {
            gammas: vec![gamma; periods.len()],
            omegas: periods.iter().map(|p| 2.0 * std::f64::consts::PI / p).collect(),
            kappa_v,
            q_c: 2.0 * gamma,
        }
    }

    fn spectral_norm(m: &DMatrix<f64>) -> f64 {
        m.singular_values().iter().fold(0.0f64, |acc, v| acc.max(*v))
    }

    #[test]
    fn test_lg_ssm_discretize_matches_closed_form_normal_case() {
        // kappa_v = 1 admits closed forms that validate the Van Loan route
        // independently: A = W·e^{B_c·dt}·Wᵀ with e^{B_c·dt} block-diagonal
        // of e^{-γ·dt}·R(ω·dt) rotations, and Q = q_c·(1 - e^{-2γ·dt})/(2γ)·I
        // (each §15.3 block is normal with F + Fᵀ = -2γ·I).
        let p = lg_ssm_small_params(1.0);
        let dt = 1.0;
        let (a, q) = lg_ssm_discretize_exact(&p, dt).unwrap();
        let dim = 4;

        let mut e_b_dt = DMatrix::<f64>::zeros(dim, dim);
        for (i, (gamma, omega)) in p.gammas.iter().zip(p.omegas.iter()).enumerate() {
            let decay = (-gamma * dt).exp();
            let (s, c) = (omega * dt).sin_cos();
            e_b_dt[(2 * i, 2 * i)] = decay * c;
            e_b_dt[(2 * i, 2 * i + 1)] = -decay * s;
            e_b_dt[(2 * i + 1, 2 * i)] = decay * s;
            e_b_dt[(2 * i + 1, 2 * i + 1)] = decay * c;
        }
        let w = lg_ssm_mixing_matrix(dim);
        let a_closed = &w * e_b_dt * w.transpose();
        let gamma = p.gammas[0];
        let q_scalar = p.q_c * (1.0 - (-2.0 * gamma * dt).exp()) / (2.0 * gamma);

        assert!((&a - a_closed).abs().max() < 1e-12);
        let q_closed = DMatrix::<f64>::identity(dim, dim) * q_scalar;
        assert!((&q - q_closed).abs().max() < 1e-12);
        // §2.5 standardization pin: q_c = 2γ makes P = I solve P = A·P·Aᵀ + Q.
        let lyapunov_at_identity = &a * a.transpose() + &q;
        assert!((lyapunov_at_identity - DMatrix::<f64>::identity(dim, dim)).abs().max() < 1e-12);
    }

    #[test]
    fn test_lg_ssm_discretize_matches_python_scipy_pin() {
        // Cross-language structural parity: (A, Q) must match Python's
        // scipy.linalg.expm Van Loan result to 1e-12 from the same params.
        // Pins generated by (numpy print with floatmode='unique'):
        //   from serie.catalog import lg_ssm_small_spec, lg_ssm_discretize_exact
        //   A, Q = lg_ssm_discretize_exact(lg_ssm_small_spec(kappa_v=5.0).params, 1.0)
        let p = lg_ssm_small_params(5.0);
        let (a, q) = lg_ssm_discretize_exact(&p, 1.0).unwrap();

        #[rustfmt::skip]
        let a_pin = DMatrix::<f64>::from_row_slice(4, 4, &[
             0.8701768342849158,    0.05502619653632668,  0.039179295008764645,  0.07023232401673338,
            -0.31083024565459133,   0.8701768342849158,   0.18412124214127268,   0.03917929500876466,
            -0.3349782603636895,   -0.6004779237390907,   0.8701768342849158,    0.055026196536326656,
            -1.5742144766690607,   -0.33497826036368966, -0.3108302456545914,    0.8701768342849158,
        ]);
        #[rustfmt::skip]
        let q_pin = DMatrix::<f64>::from_row_slice(4, 4, &[
             9.2639441738159453e-02, -1.0489756490652015e-02, -1.6086070624827187e-02, -7.6800851424767447e-02,
            -1.0489756490652020e-02,  9.6661553202188860e-02, -1.6089280345853281e-02, -9.1925768589694938e-05,
            -1.6086070624827181e-02, -1.6089280345853274e-02,  1.0929262547380229e-01,  1.6154256219430337e-02,
            -7.6800851424767419e-02, -9.1925768589701701e-05,  1.6154256219430341e-02,  1.9170072976464295e-01,
        ]);

        let a_err = (&a - a_pin).abs().max();
        let q_err = (&q - q_pin).abs().max();
        assert!(a_err < 1e-12, "A parity vs scipy pin: max err {a_err:e}");
        assert!(q_err < 1e-12, "Q parity vs scipy pin: max err {q_err:e}");
    }

    /// P = Σ_{k≥0} Aᵏ·Q·(Aᵏ)ᵀ by doubling (P ← P + A_pow·P·A_powᵀ,
    /// A_pow ← A_pow²): the unique solution of P = A·P·Aᵀ + Q for ρ(A) < 1.
    fn solve_discrete_lyapunov_by_doubling(a: &DMatrix<f64>, q: &DMatrix<f64>) -> DMatrix<f64> {
        let mut p = q.clone();
        let mut a_pow = a.clone();
        for _ in 0..60 {
            p = &p + &a_pow * &p * a_pow.transpose();
            a_pow = &a_pow * &a_pow;
        }
        p
    }

    #[test]
    fn test_lg_ssm_stationary_covariance_matches_discrete_lyapunov() {
        // Check (a): for stable A, the empirical covariance of a long
        // trajectory matches P solving P = A·P·Aᵀ + Q within sampling
        // tolerance — the normal preset (P = I by §2.5 standardization) and
        // a non-normal kappa_v = 5 case (P far from identity).
        for kappa_v in [1.0, 5.0] {
            let params = lg_ssm_small_params(kappa_v);
            let dim = 4;
            let (a, q) = lg_ssm_discretize_exact(&params, 1.0).unwrap();
            let p = solve_discrete_lyapunov_by_doubling(&a, &q);
            if kappa_v == 1.0 {
                assert!((&p - DMatrix::<f64>::identity(dim, dim)).abs().max() < 1e-12);
            }

            let steps = 60_000usize;
            let traj = run_lg_ssm_seeded(&[params], &vec![0.0; dim], steps, 1.0, 42, &[]).unwrap();
            let n_rows = (steps + 1) as f64;
            let mut mean = vec![0.0f64; dim];
            for row in 0..=steps {
                for d in 0..dim {
                    mean[d] += traj[row * dim + d];
                }
            }
            for d in 0..dim {
                mean[d] /= n_rows;
            }
            let mut empirical = DMatrix::<f64>::zeros(dim, dim);
            for row in 0..=steps {
                for i in 0..dim {
                    for j in 0..dim {
                        empirical[(i, j)] += (traj[row * dim + i] - mean[i]) * (traj[row * dim + j] - mean[j]);
                    }
                }
            }
            empirical /= n_rows - 1.0;

            let relative_error = (&empirical - &p).abs().max() / p.abs().max();
            assert!(
                relative_error < 0.05,
                "kappa_v={kappa_v}: empirical vs Lyapunov relative error {relative_error}"
            );
        }
    }

    #[test]
    fn test_lg_ssm_van_loan_q_matches_brute_force_covariance_integral() {
        // Check (b): Q from the Van Loan block exponential equals composite-
        // Simpson quadrature of ∫₀^dt e^{Fs}·q_c I·e^{Fᵀs} ds (§15.2) for a
        // small non-normal case.
        let params = lg_ssm_small_params(5.0);
        let dt = 1.0;
        let (_, q) = lg_ssm_discretize_exact(&params, dt).unwrap();
        let f = lg_ssm_drift_matrix(&params).unwrap();
        let dim = f.nrows();
        let g = DMatrix::<f64>::identity(dim, dim) * params.q_c;

        let n_intervals = 2000usize; // even, for Simpson weights 1,4,2,…,4,1
        let h = dt / n_intervals as f64;
        let mut acc = DMatrix::<f64>::zeros(dim, dim);
        for i in 0..=n_intervals {
            let e = (&f * (h * i as f64)).exp();
            let weight = if i == 0 || i == n_intervals {
                1.0
            } else if i % 2 == 1 {
                4.0
            } else {
                2.0
            };
            acc += &e * &g * e.transpose() * weight;
        }
        let q_brute_force = acc * (h / 3.0);

        let err = (&q - q_brute_force).abs().max();
        assert!(err < 1e-12, "Van Loan vs brute-force integral: max err {err:e}");
    }

    #[test]
    fn test_lg_ssm_spectral_radius_fixed_across_kappa() {
        // The controlled non-normality design (§3.5): |eig(A)| = exp(-γ·dt)
        // is independent of kappa_v over the §3.6 menu.
        for kappa_v in [1.0, 5.0, 20.0, 100.0] {
            let params = lg_ssm_params_for(&[12.0, 24.0, 48.0, 96.0], 0.95, kappa_v);
            let (a, _) = lg_ssm_discretize_exact(&params, 1.0).unwrap();
            let rho = a
                .complex_eigenvalues()
                .iter()
                .fold(0.0f64, |acc, ev| acc.max(ev.norm()));
            assert!((rho - 0.95).abs() < 1e-12, "kappa_v={kappa_v}: rho={rho}");
        }
    }

    #[test]
    fn test_lg_ssm_nonnormal_transient_growth_exceeds_normal_at_fixed_radius() {
        // Check (c): §3.7 scenario 4 ("transient-risk system: ρ(A)=0.9 but
        // G_50(A) ≫ 1 via non-normal V"). At the same spectral radius the
        // ill-conditioned-V system amplifies transients (G_K(A) = max_k ‖Aᵏ‖₂,
        // §3.5) while the normal one contracts (‖Aᵏ‖₂ = ρᵏ); both decay
        // asymptotically.
        let periods = [12.0, 24.0, 48.0, 96.0];
        let (a_normal, _) = lg_ssm_discretize_exact(&lg_ssm_params_for(&periods, 0.9, 1.0), 1.0).unwrap();
        let (a_nonnormal, _) = lg_ssm_discretize_exact(&lg_ssm_params_for(&periods, 0.9, 100.0), 1.0).unwrap();

        let norms_of_powers = |a: &DMatrix<f64>, k_max: usize| -> Vec<f64> {
            let mut a_k = DMatrix::<f64>::identity(a.nrows(), a.ncols());
            let mut norms = Vec::with_capacity(k_max);
            for _ in 0..k_max {
                a_k = &a_k * a;
                norms.push(spectral_norm(&a_k));
            }
            norms
        };
        let norms_normal = norms_of_powers(&a_normal, 300);
        let norms_nonnormal = norms_of_powers(&a_nonnormal, 300);
        let g50_normal = norms_normal[..50].iter().fold(0.0f64, |acc, v| acc.max(*v));
        let g50_nonnormal = norms_nonnormal[..50].iter().fold(0.0f64, |acc, v| acc.max(*v));

        assert!(g50_normal <= 0.9 + 1e-9, "normal case: G_50 = rho, got {g50_normal}");
        assert!(
            g50_nonnormal > 5.0,
            "designed transient amplification, got {g50_nonnormal}"
        );
        assert!(g50_nonnormal > g50_normal);
        assert!(norms_normal[299] < 1e-2, "asymptotic decay (normal)");
        assert!(norms_nonnormal[299] < 1e-2, "asymptotic decay (non-normal)");
    }

    #[test]
    fn test_lg_ssm_seed_determinism_and_seed_sensitivity() {
        use crate::stage_params::SystemParams;

        let params = SystemParams::LgSsm(lg_ssm_small_params(1.0)).as_f64_vec();
        let x0 = vec![0.0; 4];
        let a = generate_discrete_seeded("lg_ssm", &x0, 200, 1.0, 7, &[], &params, params.len()).unwrap();
        let b = generate_discrete_seeded("lg_ssm", &x0, 200, 1.0, 7, &[], &params, params.len()).unwrap();
        let c = generate_discrete_seeded("lg_ssm", &x0, 200, 1.0, 8, &[], &params, params.len()).unwrap();
        assert_eq!(a.len(), 201 * 4);
        assert!(a.iter().all(|v| v.is_finite()));
        assert_eq!(a, b, "lg_ssm: same seed must reproduce identical output");
        assert_ne!(a, c, "lg_ssm: different seed must change the output");
    }

    #[test]
    fn test_lg_ssm_param_shock_lands_at_exact_step() {
        // With the SAME injected draws, an abrupt param change re-discretizes
        // (A, Q) at the event step: rows before the shock step are bitwise
        // identical, the shock-step row is the first to differ.
        let before = lg_ssm_small_params(1.0);
        let mut after = before.clone();
        for gamma in &mut after.gammas {
            *gamma *= 2.0; // faster damping, same block count
        }
        let steps = 10usize;
        let dim = 4usize;
        let unit_normals = standard_normal_noise(99, steps, dim);
        let shocks = [ShockEvent::ParamChange {
            step: 5,
            param_set_idx: 1,
        }];

        let base = run_lg_ssm(&[before.clone()], &vec![0.0; dim], steps, 1.0, &unit_normals, &[]).unwrap();
        let shocked = run_lg_ssm(&[before, after], &vec![0.0; dim], steps, 1.0, &unit_normals, &shocks).unwrap();

        assert_eq!(
            &base[..5 * dim],
            &shocked[..5 * dim],
            "pre-shock rows must be identical"
        );
        assert_ne!(
            &base[5 * dim..6 * dim],
            &shocked[5 * dim..6 * dim],
            "shock row must differ"
        );
    }

    #[test]
    fn test_lg_ssm_param_shock_rejects_block_count_change() {
        let before = lg_ssm_small_params(1.0);
        let after = lg_ssm_params_for(&[12.0], 0.95, 1.0); // one block instead of two
        let unit_normals = standard_normal_noise(1, 10, 4);
        let shocks = [ShockEvent::ParamChange {
            step: 5,
            param_set_idx: 1,
        }];
        let err = run_lg_ssm(&[before, after], &vec![0.0; 4], 10, 1.0, &unit_normals, &shocks).unwrap_err();
        assert!(err.contains("block count"), "got error: {err}");
    }

    // ── §11e  Seeded switch (splice) tests ────────────────────────────

    #[test]
    fn test_discrete_switch_splices_a_prefix_onto_b() {
        let params_a = [0.9, 0.05, 0.98, 0.35, 0.94, 0.95];
        let params_b = [0.87, 0.07, 0.99, 0.45, 0.90, 0.95];
        let steps = 100;
        let splice_step = 35;
        let seed = 11;

        let switched = generate_discrete_switch_seeded(
            "slds",
            &[0.0],
            &[0.0],
            steps,
            1.0,
            splice_step,
            seed,
            &params_a,
            &params_b,
        )
        .unwrap();
        // Run A carries `seed`, run B carries `seed + 1`.
        let traj_a = generate_discrete_seeded("slds", &[0.0], steps, 1.0, seed, &[], &params_a, 6).unwrap();
        let traj_b = generate_discrete_seeded("slds", &[0.0], steps, 1.0, seed + 1, &[], &params_b, 6).unwrap();

        let cut = splice_step + 1; // dim = 1
        assert_eq!(&switched[..cut], &traj_a[..cut]);
        assert_eq!(&switched[cut..], &traj_b[cut..]);
    }

    #[test]
    fn test_sde_switch_splices_a_prefix_onto_b() {
        // doublewell_switch shape: A = (a=1.5, σ=0.25), B = (a=1.0, σ=0.35).
        let params_a = [1.5, 0.25];
        let params_b = [1.0, 0.35];
        let steps = 100;
        let splice_step = 35;
        let seed = 11;

        let switched = generate_sde_switch_seeded(
            "doublewell",
            &[0.0],
            &[0.0],
            steps,
            0.5,
            0.25,
            0.35,
            splice_step,
            seed,
            &params_a,
            &params_b,
        )
        .unwrap();
        let traj_a = generate_sde_seeded("doublewell", &[0.0], steps, 0.5, &[0.25], seed, &[], &params_a, 2).unwrap();
        let traj_b =
            generate_sde_seeded("doublewell", &[0.0], steps, 0.5, &[0.35], seed + 1, &[], &params_b, 2).unwrap();

        let cut = splice_step + 1; // dim = 1
        assert_eq!(&switched[..cut], &traj_a[..cut]);
        assert_eq!(&switched[cut..], &traj_b[cut..]);
    }

    #[test]
    fn test_switch_rejects_splice_step_beyond_steps() {
        let params = [0.9, 0.05, 0.98, 0.35, 0.94, 0.95];
        let err =
            generate_discrete_switch_seeded("slds", &[0.0], &[0.0], 10, 1.0, 11, 0, &params, &params).unwrap_err();
        assert!(err.contains("splice_step"));
    }
}
