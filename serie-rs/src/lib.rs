/// Exact discrete-time controlled linear-Gaussian oracle and action-bank
/// coverage accounting for Fern Program II's CPU-only Level-0 fixture.
pub mod controlled_linear;
/// Smooth controlled Lorenz-63, Rössler, Duffing, Van der Pol, and Lorenz-96
/// fields advanced by zero-order-hold RK4 with exact discrete-step and
/// accumulated Jacobians.
// CHG 2026-08-28: the declaration and its doc comment above were dropped by the
// 084dbbea move and restored from 084dbbea^; controlled_ode.rs never left.
pub mod controlled_ode;
/// Partially observed nonlinear Pendulum with known sinusoidal forcing-state
/// truth and exact Jacobians of its declared discrete transition.
pub mod controlled_pendulum;
pub mod dgp_bank;
pub mod draw_protocol;
pub mod numeric_frame;
#[cfg(feature = "known-bank")]
pub mod stage_clean;
#[cfg(feature = "diagnostics")]
pub mod stage_diag_cli;
#[cfg(feature = "diagnostics")]
pub mod stage_diag_text;
pub mod stage_generate;
#[cfg(feature = "known-data")]
pub mod stage_load;
#[cfg(feature = "pipeline")]
pub mod stage_noise;
pub mod stage_params;
#[cfg(feature = "pipeline")]
pub mod stage_partition;
#[cfg(feature = "pipeline")]
pub mod stage_ready;
pub mod stage_synthetic;
