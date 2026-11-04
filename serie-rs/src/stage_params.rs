use std::collections::HashMap;

fn req(map: &HashMap<String, f64>, key: &str) -> Result<f64, String> {
    map.get(key)
        .copied()
        .ok_or_else(|| format!("missing parameter '{}'", key))
}

fn round_half_to_even(value: f64) -> i64 {
    let floor_value = value.floor();
    let fractional = value - floor_value;
    let floor_int = floor_value as i64;

    if fractional < 0.5 {
        floor_int
    } else if fractional > 0.5 {
        floor_int + 1
    } else if floor_int % 2 == 0 {
        floor_int
    } else {
        floor_int + 1
    }
}

fn round_half_to_even_usize(value: f64) -> usize {
    round_half_to_even(value) as usize
}

/// Lorenz system (1963).
///
/// The canonical chaotic attractor arising from simplified atmospheric convection.
///
///   dx/dt = σ(y - x)
///   dy/dt = x(ρ - z) - y
///   dz/dt = xy - βz
///
/// Standard chaotic regime: σ=10, ρ=28, β=8/3.
#[derive(Clone, Debug)]
pub struct LorenzParams {
    pub sigma: f64, // Prandtl number (ratio of momentum to thermal diffusivity)
    pub rho: f64,   // Rayleigh number (temperature gradient strength)
    pub beta: f64,  // Geometric factor related to cell aspect ratio
}

impl LorenzParams {
    pub fn try_from_map(map: &HashMap<String, f64>) -> Result<Self, String> {
        Ok(Self {
            sigma: req(map, "sigma")?,
            rho: req(map, "rho")?,
            beta: req(map, "beta")?,
        })
    }
}

/// Rössler system (1976).
///
/// A simpler chaotic system than Lorenz, with only one nonlinear term (xz).
///
///   dx/dt = -y - z
///   dy/dt = x + ay
///   dz/dt = b + z(x - c)
///
/// Standard chaotic regime: a=0.2, b=0.2, c=5.7.
#[derive(Clone, Debug)]
pub struct RosslerParams {
    pub a: f64, // Linear feedback in y
    pub b: f64, // Offset in z equation
    pub c: f64, // Nonlinear chaos parameter (controls attractor size)
}

impl RosslerParams {
    pub fn try_from_map(map: &HashMap<String, f64>) -> Result<Self, String> {
        Ok(Self {
            a: req(map, "a")?,
            b: req(map, "b")?,
            c: req(map, "c")?,
        })
    }
}

/// Chua circuit (1983).
///
/// The simplest electronic circuit exhibiting chaos. Uses piecewise-linear
/// nonlinearity h(x) = m0_outer_gt1·x + 0.5·(m1_inner_le1-m0_outer_gt1)·(|x+1| - |x-1|).
///
///   dx/dt = α(y - x - h(x))
///   dy/dt = x - y + z
///   dz/dt = -βy
///
/// Standard double-scroll: α=15.6, β=28, m1_inner_le1=-8/7, m0_outer_gt1=-5/7.
/// REQUIRE: dt ≤ 0.005 for numerical stability.
#[derive(Clone, Debug)]
pub struct ChuaParams {
    pub alpha: f64,        // Stiffness of x-dynamics
    pub beta: f64,         // z-dynamics coupling (chaos ~25-51)
    pub m1_inner_le1: f64, // Inner slope (|x| ≤ 1), standard: -8/7
    pub m0_outer_gt1: f64, // Outer slope (|x| > 1), standard: -5/7
}

impl ChuaParams {
    pub fn try_from_map(map: &HashMap<String, f64>) -> Result<Self, String> {
        let m1_inner_le1 = map
            .get("m1_inner_le1")
            .copied()
            .ok_or_else(|| "missing parameter 'm1_inner_le1'".to_string())?;
        let m0_outer_gt1 = map
            .get("m0_outer_gt1")
            .copied()
            .ok_or_else(|| "missing parameter 'm0_outer_gt1'".to_string())?;
        Ok(Self {
            alpha: req(map, "alpha")?,
            beta: req(map, "beta")?,
            m1_inner_le1,
            m0_outer_gt1,
        })
    }
}

/// Van der Pol oscillator (1920).
///
/// Self-sustained oscillator with nonlinear damping.
///
///   dx/dt = y
///   dy/dt = μ(1 - x²)y - x
///
/// μ > 0 controls nonlinearity strength. Large μ → relaxation oscillations.
#[derive(Clone, Debug)]
pub struct VanDerPolParams {
    pub mu: f64, // Nonlinearity parameter
}

impl VanDerPolParams {
    pub fn try_from_map(map: &HashMap<String, f64>) -> Result<Self, String> {
        Ok(Self { mu: req(map, "mu")? })
    }
}

/// Duffing oscillator (forced, damped).
///
/// Nonlinear oscillator with cubic stiffness and periodic forcing.
/// TIME-DEPENDENT (requires t in the vector field).
///
///   dx/dt = v
///   dv/dt = γcos(ωt) - δv - αx - βx³
///
/// Standard chaotic regime: α=1, β=-1, δ=0.2, γ=0.65, ω=1.2.
#[derive(Clone, Debug)]
pub struct DuffingParams {
    pub alpha: f64, // Linear stiffness
    pub beta: f64,  // Cubic stiffness (< 0 for softening spring)
    pub delta: f64, // Damping coefficient
    pub gamma: f64, // Forcing amplitude
    pub omega: f64, // Forcing frequency
}

impl DuffingParams {
    pub fn try_from_map(map: &HashMap<String, f64>) -> Result<Self, String> {
        Ok(Self {
            alpha: req(map, "alpha")?,
            beta: req(map, "beta")?,
            delta: req(map, "delta")?,
            gamma: req(map, "gamma")?,
            omega: req(map, "omega")?,
        })
    }
}

/// Hyper-Rössler system (4D hyperchaotic).
///
/// Extension of Rössler with an additional dimension w for hyperchaos
/// (two positive Lyapunov exponents).
///
///   dx/dt = -y - z
///   dy/dt = x + ay + w
///   dz/dt = b + xz
///   dw/dt = -cz + dw
///
/// Standard hyperchaotic regime: a=0.25, b=3.0, c=0.5, d=0.05.
#[derive(Clone, Debug)]
pub struct HyperRosslerParams {
    pub a: f64, // Linear feedback in y
    pub b: f64, // Offset in z
    pub c: f64, // Coupling z→w
    pub d: f64, // Growth term for w
}

impl HyperRosslerParams {
    pub fn try_from_map(map: &HashMap<String, f64>) -> Result<Self, String> {
        Ok(Self {
            a: req(map, "a")?,
            b: req(map, "b")?,
            c: req(map, "c")?,
            d: req(map, "d")?,
        })
    }
}

/// Lorenz-96 model (1996).
///
/// Simplified atmospheric model on a 1D periodic lattice of N variables.
/// Mimics energy transfer across spatial scales in the atmosphere.
///
///   dx_i/dt = (x_{i+1} - x_{i-2}) · x_{i-1} - x_i + F
///
/// Variable dimension (typically N=40). Chaos when F ≥ 8.
/// Periodic boundary: indices wrap modulo N.
#[derive(Clone, Debug)]
pub struct Lorenz96Params {
    pub forcing: f64, // External forcing F
    pub dim: usize,   // Number of lattice sites
}

impl Lorenz96Params {
    ///Lorenz96 is special: it takes an extra dim argument because dim comes from the initial
    /// condition length, not from the param array. Python only sends [forcing] as params.
    /// The dim shorthand on the right side — Self { forcing: p[0], dim } — is Rust's field
    /// init shorthand: when variable name matches field name, you can skip the colon.
    /// Same as Python's dict(x=x) but terser.
    pub fn try_from_map(map: &HashMap<String, f64>, dim: usize) -> Result<Self, String> {
        Ok(Self {
            forcing: req(map, "forcing")?,
            dim,
        })
    }
}

/// Hénon map (1976).
///
/// 2D discrete-time chaotic map. Area-contracting (|b| < 1) with a
/// strange attractor that is a fractal (Hausdorff dim ≈ 1.26).
///
///   x_{n+1} = 1 - a·x_n² + y_n
///   y_{n+1} = b·x_n
///
/// Standard chaotic regime: a=1.4, b=0.3.
#[derive(Clone, Debug)]
pub struct HenonParams {
    pub a: f64, // Nonlinear coefficient
    pub b: f64, // Linear coefficient (contraction rate)
}

impl HenonParams {
    pub fn try_from_map(map: &HashMap<String, f64>) -> Result<Self, String> {
        Ok(Self {
            a: req(map, "a")?,
            b: req(map, "b")?,
        })
    }
}

/// Logistic map.
///
/// 1D discrete-time map exhibiting the period-doubling route to chaos.
/// The prototypical example of how simple deterministic rules produce
/// complex, unpredictable behavior.
///
///   x_{n+1} = r·x_n·(1 - x_n)
///
/// Period-doubling cascade: r ∈ [3.0, 3.57).
/// Chaos: r ∈ [3.57, 4.0]. Full chaos at r ≈ 3.9.
/// x ∈ [0, 1] is invariant when r ∈ [0, 4].
#[derive(Clone, Debug)]
pub struct LogisticParams {
    pub r: f64, // Growth rate
}

impl LogisticParams {
    pub fn try_from_map(map: &HashMap<String, f64>) -> Result<Self, String> {
        Ok(Self { r: req(map, "r")? })
    }
}

/// Ornstein-Uhlenbeck process: dx = θ(μ - x)dt + σdW
///
/// Mean-reverting stochastic process. The drift is purely linear.
/// OUParams fields (Python): [theta, mu, sigma_noise]
/// We store only what the drift needs; sigma_noise is handled externally.
#[derive(Clone, Debug)]
pub struct OuParams {
    pub theta: f64,       // Mean reversion rate
    pub mu: f64,          // Long-term mean
    pub sigma_noise: f64, // Diffusion scale
}

impl OuParams {
    pub fn try_from_map(map: &HashMap<String, f64>) -> Result<Self, String> {
        Ok(Self {
            theta: req(map, "theta")?,
            mu: req(map, "mu")?,
            sigma_noise: req(map, "sigma_noise")?,
        })
    }
}

/// Double-well Langevin dynamics: dx = (-x³ + ax)dt + σdW
///
/// Particle in bistable potential U(x) = x⁴/4 - (a/2)x².
/// Minima at x = ±√a. Noise-induced transitions between wells.
/// DoubleWellParams fields (Python): [a, sigma_noise]
#[derive(Clone, Debug)]
pub struct DoubleWellParams {
    pub a: f64,           // Well separation (minima at ±√a)
    pub sigma_noise: f64, // Diffusion scale
}

impl DoubleWellParams {
    pub fn try_from_map(map: &HashMap<String, f64>) -> Result<Self, String> {
        Ok(Self {
            a: req(map, "a")?,
            sigma_noise: req(map, "sigma_noise")?,
        })
    }
}

/// LG-SSM: continuous-time linear Gaussian SDE sampled at a fixed interval dt
/// by EXACT discretization (no Euler stepping). Port of LGSSMParams
/// (serie/catalog.py).
///
/// Roadmap: docs/lab/directions/controlled_model_families_roadmap.md
///   §15.2 (matrix Ornstein-Uhlenbeck model),
///   §15.3 (damped oscillatory spectral blocks),
///   §3.5/§3.6 (controlled non-normality via eigenvector conditioning κ(V)).
///
/// Latent SDE (§15.2, with L = I_D and Q_c = q_c·I_D):
///   dx(t) = F x(t) dt + L dW(t)
/// Drift (§15.3 blocks assembled per §3.5):
///   B_c = blockdiag(F_1, …, F_m),  F_i = [[-γ_i, -ω_i], [ω_i, -γ_i]]
///   F   = V B_c V⁻¹,               V = diag(σ)·W (orthonormal DCT-II W,
///                                   σ_j = kappa_v^{j/(D-1)}, cond₂(V) = kappa_v)
/// Exact sampling at interval dt (§15.2, Van Loan):
///   x_{k+1} = A x_k + w_k,  A = exp(F·dt),  w_k ~ N(0, Q)
///
/// State dimension D = 2·gammas.len(). |eig(A)| = exp(-γ_i·dt) is independent
/// of kappa_v — spectral radius held fixed while V's conditioning varies.
///
/// REQUIRE: gammas.len() == omegas.len() >= 1; gammas > 0; kappa_v >= 1; q_c > 0.
#[derive(Clone, Debug)]
pub struct LgSsmParams {
    pub gammas: Vec<f64>, // Per-block damping rates γ_i (§15.3)
    pub omegas: Vec<f64>, // Per-block angular frequencies ω_i (§15.3)
    pub kappa_v: f64,     // Eigenvector conditioning κ(V) (§3.6 menu: 1, 5, 20, 100)
    pub q_c: f64,         // Process-noise spectral density: L = I, Q_c = q_c·I (§15.2)
}

impl LgSsmParams {
    /// Indexed-key map form, like DiagonalParams: per-block entries are
    /// "gamma0", "gamma1", …, "omega0", "omega1", …, plus the scalar knobs
    /// "kappa_v" and "q_c".
    pub fn try_from_map(map: &HashMap<String, f64>) -> Result<Self, String> {
        fn indexed(map: &HashMap<String, f64>, prefix: &str) -> Result<Vec<f64>, String> {
            let mut pairs: Vec<(usize, f64)> = map
                .iter()
                .filter_map(|(k, &v)| {
                    k.strip_prefix(prefix)
                        .and_then(|raw| raw.parse::<usize>().ok())
                        .map(|idx| (idx, v))
                })
                .collect();
            pairs.sort_by_key(|(idx, _)| *idx);
            if pairs.is_empty() {
                return Err(format!("missing lg_ssm keys {prefix}0,{prefix}1,..."));
            }
            let mut vals = Vec::with_capacity(pairs.len());
            for (i, (idx, val)) in pairs.into_iter().enumerate() {
                if idx != i {
                    return Err(format!("lg_ssm {prefix} keys must be contiguous: expected {prefix}{i}"));
                }
                vals.push(val);
            }
            Ok(vals)
        }
        let gammas = indexed(map, "gamma")?;
        let omegas = indexed(map, "omega")?;
        if gammas.len() != omegas.len() {
            return Err(format!(
                "lg_ssm requires one omega per gamma, got {} gammas and {} omegas",
                gammas.len(),
                omegas.len()
            ));
        }
        Ok(Self {
            gammas,
            omegas,
            kappa_v: req(map, "kappa_v")?,
            q_c: req(map, "q_c")?,
        })
    }
}

#[derive(Clone, Debug)]
pub struct SldsParams {
    pub a1: f64,
    pub q1: f64,
    pub a2: f64,
    pub q2: f64,
    pub p11: f64,
    pub p22: f64,
}

impl SldsParams {
    pub fn try_from_map(map: &HashMap<String, f64>) -> Result<Self, String> {
        Ok(Self {
            a1: req(map, "A1")?,
            q1: req(map, "Q1")?,
            a2: req(map, "A2")?,
            q2: req(map, "Q2")?,
            p11: req(map, "p11")?,
            p22: req(map, "p22")?,
        })
    }
}

#[derive(Clone, Debug)]
pub struct SeasonalArParams {
    pub s: f64,
    pub phi: f64,
    pub sigma: f64,
    pub a0: f64,
    pub amp_drift: f64,
}

impl SeasonalArParams {
    pub fn try_from_map(map: &HashMap<String, f64>) -> Result<Self, String> {
        Ok(Self {
            s: req(map, "S")?,
            phi: req(map, "phi")?,
            sigma: req(map, "sigma")?,
            a0: req(map, "a0")?,
            amp_drift: req(map, "amp_drift")?,
        })
    }
}

#[derive(Clone, Debug)]
pub struct GarchParams {
    pub omega: f64,
    pub alpha: f64,
    pub beta: f64,
}

impl GarchParams {
    pub fn try_from_map(map: &HashMap<String, f64>) -> Result<Self, String> {
        Ok(Self {
            omega: req(map, "omega")?,
            alpha: req(map, "alpha")?,
            beta: req(map, "beta")?,
        })
    }
}

#[derive(Clone, Debug)]
pub struct KsParams {
    pub nu: f64,
    pub lx: f64,
    pub nx: usize,
    pub dealias: bool,
}

impl KsParams {
    pub fn try_from_map(map: &HashMap<String, f64>) -> Result<Self, String> {
        Ok(Self {
            nu: req(map, "nu")?,
            lx: req(map, "Lx")?,
            nx: req(map, "nx")? as usize,
            dealias: req(map, "dealias")? >= 0.5,
        })
    }
}

#[derive(Clone, Debug)]
pub struct DiagonalParams {
    pub eigenvalues: Vec<f64>,
}

impl DiagonalParams {
    pub fn try_from_map(map: &HashMap<String, f64>) -> Result<Self, String> {
        let mut pairs: Vec<(usize, f64)> = map
            .iter()
            .filter_map(|(k, &v)| {
                if let Some(raw) = k.strip_prefix('e') {
                    raw.parse::<usize>().ok().map(|idx| (idx, v))
                } else {
                    None
                }
            })
            .collect();
        pairs.sort_by_key(|(idx, _)| *idx);
        if pairs.is_empty() {
            return Err("missing diagonal eigenvalues keys e0,e1,...".to_string());
        }
        let mut vals = Vec::with_capacity(pairs.len());
        for (i, (idx, val)) in pairs.into_iter().enumerate() {
            if idx != i {
                return Err(format!("diagonal eigenvalue keys must be contiguous: expected e{}", i));
            }
            vals.push(val);
        }
        Ok(Self { eigenvalues: vals })
    }
}

#[derive(Clone, Debug)]
pub enum SystemParams {
    Lorenz(LorenzParams),
    Rossler(RosslerParams),
    HyperRossler(HyperRosslerParams),
    Chua(ChuaParams),
    Duffing(DuffingParams),
    VanDerPol(VanDerPolParams),
    Lorenz96(Lorenz96Params),
    Henon(HenonParams),
    Logistic(LogisticParams),
    OU(OuParams),
    DoubleWell(DoubleWellParams),
    LgSsm(LgSsmParams),
    SLDS(SldsParams),
    SeasonalAR(SeasonalArParams),
    GARCH(GarchParams),
    KS(KsParams),
    Diagonal(DiagonalParams),
}

impl SystemParams {
    pub(crate) fn as_f64_vec(&self) -> Vec<f64> {
        match self {
            Self::Lorenz(params) => vec![params.sigma, params.rho, params.beta],
            Self::Rossler(params) => vec![params.a, params.b, params.c],
            Self::HyperRossler(params) => vec![params.a, params.b, params.c, params.d],
            Self::Chua(params) => vec![params.alpha, params.beta, params.m1_inner_le1, params.m0_outer_gt1],
            Self::Duffing(params) => vec![params.alpha, params.beta, params.delta, params.gamma, params.omega],
            Self::VanDerPol(params) => vec![params.mu],
            // Lorenz96 dim is structural shape, not a driftable runtime parameter.
            // Keep it fixed from the current instance and expose only forcing.
            Self::Lorenz96(params) => vec![params.forcing],
            Self::Henon(params) => vec![params.a, params.b],
            Self::Logistic(params) => vec![params.r],
            Self::OU(params) => vec![params.theta, params.mu, params.sigma_noise],
            Self::DoubleWell(params) => vec![params.a, params.sigma_noise],
            // Flattened as [gammas…, omegas…, kappa_v, q_c] — the same order
            // Python _transition_field_values uses for LGSSMParams.
            Self::LgSsm(params) => {
                let mut values = params.gammas.clone();
                values.extend_from_slice(&params.omegas);
                values.push(params.kappa_v);
                values.push(params.q_c);
                values
            }
            Self::SLDS(params) => vec![params.a1, params.q1, params.a2, params.q2, params.p11, params.p22],
            Self::SeasonalAR(params) => {
                vec![params.s, params.phi, params.sigma, params.a0, params.amp_drift]
            }
            Self::GARCH(params) => vec![params.omega, params.alpha, params.beta],
            Self::KS(params) => vec![
                params.nu,
                params.lx,
                params.nx as f64,
                if params.dealias { 1.0 } else { 0.0 },
            ],
            Self::Diagonal(params) => params.eigenvalues.clone(),
        }
    }

    fn from_f64_slice(&self, values: &[f64]) -> Self {
        match self {
            Self::Lorenz(_) => Self::Lorenz(LorenzParams {
                sigma: values[0],
                rho: values[1],
                beta: values[2],
            }),
            Self::Rossler(_) => Self::Rossler(RosslerParams {
                a: values[0],
                b: values[1],
                c: values[2],
            }),
            Self::HyperRossler(_) => Self::HyperRossler(HyperRosslerParams {
                a: values[0],
                b: values[1],
                c: values[2],
                d: values[3],
            }),
            Self::Chua(_) => Self::Chua(ChuaParams {
                alpha: values[0],
                beta: values[1],
                m1_inner_le1: values[2],
                m0_outer_gt1: values[3],
            }),
            Self::Duffing(_) => Self::Duffing(DuffingParams {
                alpha: values[0],
                beta: values[1],
                delta: values[2],
                gamma: values[3],
                omega: values[4],
            }),
            Self::VanDerPol(_) => Self::VanDerPol(VanDerPolParams { mu: values[0] }),
            // Preserve the existing lattice size; mid-run shape changes are not allowed.
            Self::Lorenz96(current) => Self::Lorenz96(Lorenz96Params {
                forcing: values[0],
                dim: current.dim,
            }),
            Self::Henon(_) => Self::Henon(HenonParams {
                a: values[0],
                b: values[1],
            }),
            Self::Logistic(_) => Self::Logistic(LogisticParams { r: values[0] }),
            Self::OU(_) => Self::OU(OuParams {
                theta: values[0],
                mu: values[1],
                sigma_noise: values[2],
            }),
            Self::DoubleWell(_) => Self::DoubleWell(DoubleWellParams {
                a: values[0],
                sigma_noise: values[1],
            }),
            // Block count is structural shape (like Lorenz96 dim): preserved
            // from the current instance, never interpolated.
            Self::LgSsm(current) => {
                let n_blocks = current.gammas.len();
                Self::LgSsm(LgSsmParams {
                    gammas: values[..n_blocks].to_vec(),
                    omegas: values[n_blocks..2 * n_blocks].to_vec(),
                    kappa_v: values[2 * n_blocks],
                    q_c: values[2 * n_blocks + 1],
                })
            }
            Self::SLDS(_) => Self::SLDS(SldsParams {
                a1: values[0],
                q1: values[1],
                a2: values[2],
                q2: values[3],
                p11: values[4],
                p22: values[5],
            }),
            Self::SeasonalAR(_) => Self::SeasonalAR(SeasonalArParams {
                s: values[0],
                phi: values[1],
                sigma: values[2],
                a0: values[3],
                amp_drift: values[4],
            }),
            Self::GARCH(_) => Self::GARCH(GarchParams {
                omega: values[0],
                alpha: values[1],
                beta: values[2],
            }),
            Self::KS(_) => Self::KS(KsParams {
                nu: values[0],
                lx: values[1],
                nx: round_half_to_even_usize(values[2]),
                dealias: values[3] >= 0.5,
            }),
            Self::Diagonal(_) => Self::Diagonal(DiagonalParams {
                eigenvalues: values.to_vec(),
            }),
        }
    }

    pub(crate) fn interpolate_toward(&self, after: &Self, transition_progress: f64) -> Result<Self, String> {
        if std::mem::discriminant(self) != std::mem::discriminant(after) {
            return Err(format!(
                "param interpolation requires matching variants, got {:?} vs {:?}",
                self, after
            ));
        }

        let before_values = self.as_f64_vec();
        let after_values = after.as_f64_vec();
        if before_values.len() != after_values.len() {
            return Err(format!(
                "param interpolation requires matching field counts, got {} vs {}",
                before_values.len(),
                after_values.len()
            ));
        }

        // CHG (user-approved 2026-07-01): two-product lerp, matching Python
        // _interpolate_scalar textually: before·(1−α) + after·α. At α = 1.0
        // this is before·0.0 + after·1.0 = after exactly, so the final
        // scheduled update of a transition lands bitwise on the target params.
        // (The previous before + (after−before)·α form did not guarantee that.)
        let mut interpolated_values = before_values
            .iter()
            .zip(after_values.iter())
            .map(|(before, after)| before * (1.0 - transition_progress) + after * transition_progress)
            .collect::<Vec<_>>();

        // Structural/integer params never interpolate: they define the
        // system's shape or topology, not a driftable quantity. Seasonal-AR S
        // (the period) mirrors Python _transition_field_names, which excludes
        // S from transitions — it stays at the before-value for every
        // scheduled update. (Lorenz96 dim is likewise structural; it is
        // preserved by from_f64_slice taking it from the current instance.)
        if let Self::SeasonalAR(before_params) = self {
            interpolated_values[0] = before_params.s;
        }
        Ok(self.from_f64_slice(&interpolated_values))
    }
}

/// Tests for parameter interpolation.
///
/// Why these tests exist:
/// - The lerp form before·(1−α) + after·α must land bitwise on `after` at
///   α = 1.0 — the transition compiler relies on it so a completed transition
///   is exactly the requested target regime, matching Python.
/// - Structural params (Seasonal-AR S) must never move during a windowed
///   transition, mirroring Python _transition_field_names.
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn interpolate_at_progress_one_lands_exactly_on_after() {
        let before = SystemParams::Lorenz(LorenzParams {
            sigma: 10.0,
            rho: 28.0,
            beta: 8.0 / 3.0,
        });
        let after = SystemParams::Lorenz(LorenzParams {
            sigma: 10.1,
            rho: 28.1,
            beta: 8.1 / 3.0,
        });
        match before.interpolate_toward(&after, 1.0).unwrap() {
            SystemParams::Lorenz(landed) => {
                assert_eq!(landed.sigma, 10.1);
                assert_eq!(landed.rho, 28.1);
                assert_eq!(landed.beta, 8.1 / 3.0);
            }
            other => panic!("expected Lorenz params, got {:?}", other),
        }
    }

    #[test]
    fn interpolate_matches_python_two_product_lerp_form() {
        let before = SystemParams::OU(OuParams {
            theta: 0.2,
            mu: 0.0,
            sigma_noise: 0.3,
        });
        let after = SystemParams::OU(OuParams {
            theta: 0.8,
            mu: 0.5,
            sigma_noise: 0.6,
        });
        let alpha = 0.25;
        match before.interpolate_toward(&after, alpha).unwrap() {
            SystemParams::OU(mid) => {
                // Python _interpolate_scalar: before * (1.0 - alpha) + after * alpha
                assert_eq!(mid.theta, 0.2 * (1.0 - alpha) + 0.8 * alpha);
                assert_eq!(mid.mu, 0.0 * (1.0 - alpha) + 0.5 * alpha);
                assert_eq!(mid.sigma_noise, 0.3 * (1.0 - alpha) + 0.6 * alpha);
            }
            other => panic!("expected OU params, got {:?}", other),
        }
    }

    #[test]
    fn seasonal_ar_period_stays_fixed_during_interpolation() {
        let before = SystemParams::SeasonalAR(SeasonalArParams {
            s: 24.0,
            phi: 0.5,
            sigma: 0.2,
            a0: 1.0,
            amp_drift: 0.0,
        });
        let after = SystemParams::SeasonalAR(SeasonalArParams {
            s: 48.0, // structural: must NOT move
            phi: 0.8,
            sigma: 0.35,
            a0: 1.4,
            amp_drift: 0.0,
        });
        for alpha in [0.5, 1.0] {
            match before.interpolate_toward(&after, alpha).unwrap() {
                SystemParams::SeasonalAR(mid) => {
                    assert_eq!(mid.s, 24.0, "S must stay at the before-value at alpha={}", alpha);
                    assert_eq!(mid.phi, 0.5 * (1.0 - alpha) + 0.8 * alpha);
                }
                other => panic!("expected SeasonalAR params, got {:?}", other),
            }
        }
    }
}
