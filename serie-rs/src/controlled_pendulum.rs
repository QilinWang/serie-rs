//! Partially observed controlled Pendulum for forced-delay reconstruction.
//!
//! The physical state is `(angle, angular_velocity)`, while the public
//! observation is angle only. A separate sinusoidal torque driver exposes its
//! exact circular forcing coordinates `(sin(phase), cos(phase))`. The fixture
//! therefore gives state-delay and action-delay encoders different, known
//! reconstruction targets without treating the action history as base noise.
//!
//! The plant uses one declared symplectic-Euler transition:
//!
//! ```text
//! angular_velocity[t + 1] = angular_velocity[t]
//!     + dt * (-damping * angular_velocity[t]
//!             - gravity_over_length * sin(angle[t])
//!             + torque_gain * action[t])
//! angle[t + 1] = angle[t] + dt * angular_velocity[t + 1]
//! ```
//!
//! Its local state and action Jacobians are analytic. Direct-horizon action
//! sensitivities are accumulated by the chain rule and are lower triangular
//! under the transition indexing above.
//!
//! ```
//! use serie_rs::controlled_pendulum::{
//!     ControlledPendulum, PendulumState, SinusoidalTorqueDriver,
//!     sinusoidally_forced_trajectory,
//! };
//!
//! let plant = ControlledPendulum::try_new(0.05, 0.15, 9.81, 0.8)?;
//! let driver = SinusoidalTorqueDriver::try_new(0.6, 0.2)?;
//! let trajectory = sinusoidally_forced_trajectory(
//!     &plant,
//!     &driver,
//!     PendulumState { angle: 0.1, angular_velocity: -0.3 },
//!     0.7,
//!     16,
//! )?;
//! assert_eq!(trajectory.observed_angles.len(), 17);
//! # Ok::<(), String>(())
//! ```

use nalgebra::{Matrix2, Vector2};
use ndarray::Array4;

const PHASE_RECONSTRUCTION_TOLERANCE: f64 = 1e-10;

/// Complete physical Pendulum state. Training fixtures may expose only
/// `angle`, while evaluation retains both coordinates as oracle truth.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PendulumState {
    pub angle: f64,
    pub angular_velocity: f64,
}

/// One exact transition and its local derivatives.
#[derive(Clone, Debug, PartialEq)]
pub struct PendulumStep {
    pub next_state: PendulumState,
    pub local_state_jacobian: Matrix2<f64>,
    pub local_action_jacobian: Vector2<f64>,
}

/// Nonlinear controlled Pendulum with an explicit integration contract.
#[derive(Clone, Debug, PartialEq)]
pub struct ControlledPendulum {
    timestep: f64,
    damping: f64,
    gravity_over_length: f64,
    torque_gain: f64,
}

impl ControlledPendulum {
    /// Construct a controlled plant only from finite physical parameters.
    pub fn try_new(timestep: f64, damping: f64, gravity_over_length: f64, torque_gain: f64) -> Result<Self, String> {
        for (name, value) in [
            ("timestep", timestep),
            ("damping", damping),
            ("gravity_over_length", gravity_over_length),
            ("torque_gain", torque_gain),
        ] {
            if !value.is_finite() {
                return Err(format!("{name} must be finite, got {value}"));
            }
        }
        if timestep <= 0.0 {
            return Err(format!("timestep must be > 0, got {timestep}"));
        }
        if damping < 0.0 {
            return Err(format!("damping must be >= 0, got {damping}"));
        }
        if gravity_over_length <= 0.0 {
            return Err(format!("gravity_over_length must be > 0, got {gravity_over_length}"));
        }
        if torque_gain == 0.0 {
            return Err("torque_gain must be nonzero for a controlled fixture".to_string());
        }
        Ok(Self {
            timestep,
            damping,
            gravity_over_length,
            torque_gain,
        })
    }

    /// Advance one action interval and return the analytic local `A_t` and
    /// `B_t` derivatives of the declared discrete transition.
    pub fn step(&self, state: PendulumState, action: f64) -> Result<PendulumStep, String> {
        validate_state(state)?;
        if !action.is_finite() {
            return Err(format!("action must be finite, got {action}"));
        }

        let angular_acceleration = -self.damping * state.angular_velocity
            - self.gravity_over_length * state.angle.sin()
            + self.torque_gain * action;
        let next_angular_velocity = state.angular_velocity + self.timestep * angular_acceleration;
        let next_angle = state.angle + self.timestep * next_angular_velocity;
        if !next_angle.is_finite() || !next_angular_velocity.is_finite() {
            return Err("Pendulum transition produced a non-finite state".to_string());
        }

        let angular_velocity_from_angle = -self.timestep * self.gravity_over_length * state.angle.cos();
        let angular_velocity_from_angular_velocity = 1.0 - self.timestep * self.damping;
        let local_state_jacobian = Matrix2::new(
            1.0 + self.timestep * angular_velocity_from_angle,
            self.timestep * angular_velocity_from_angular_velocity,
            angular_velocity_from_angle,
            angular_velocity_from_angular_velocity,
        );
        let local_action_jacobian = Vector2::new(
            self.timestep * self.timestep * self.torque_gain,
            self.timestep * self.torque_gain,
        );

        Ok(PendulumStep {
            next_state: PendulumState {
                angle: next_angle,
                angular_velocity: next_angular_velocity,
            },
            local_state_jacobian,
            local_action_jacobian,
        })
    }

    /// Recover `angular_velocity[t]` exactly from the two angle observations
    /// used by this symplectic-Euler contract:
    /// `angle[t] = angle[t-1] + timestep * angular_velocity[t]`.
    pub fn angular_velocity_from_two_angles(&self, previous_angle: f64, current_angle: f64) -> Result<f64, String> {
        for (name, angle) in [("previous_angle", previous_angle), ("current_angle", current_angle)] {
            if !angle.is_finite() {
                return Err(format!("{name} must be finite, got {angle}"));
            }
        }
        Ok((current_angle - previous_angle) / self.timestep)
    }

    /// Apply one complete planned action path.
    ///
    /// `action_jacobian` has shape `[horizon, 2, horizon, 1]`. Its axes name
    /// future timestep, state coordinate, action timestep, and action channel.
    pub fn direct_horizon(
        &self,
        initial_state: PendulumState,
        future_actions: &[f64],
    ) -> Result<ControlledPendulumHorizon, String> {
        validate_state(initial_state)?;
        if let Some(action) = future_actions.iter().copied().find(|action| !action.is_finite()) {
            return Err(format!(
                "future_actions must contain only finite values, found {action}"
            ));
        }

        let horizon = future_actions.len();
        let mut state = initial_state;
        let mut states = Vec::with_capacity(horizon);
        let mut local_state_jacobians = Vec::with_capacity(horizon);
        let mut local_action_jacobians = Vec::with_capacity(horizon);
        let mut initial_state_jacobians = Vec::with_capacity(horizon);
        let mut initial_state_sensitivity = Matrix2::identity();
        let mut action_sensitivities = vec![Vector2::zeros(); horizon];
        let mut action_jacobian = Array4::zeros((horizon, 2, horizon, 1));

        for (timestep, action) in future_actions.iter().copied().enumerate() {
            let step = self.step(state, action)?;
            initial_state_sensitivity = step.local_state_jacobian * initial_state_sensitivity;
            for sensitivity in &mut action_sensitivities[..timestep] {
                *sensitivity = step.local_state_jacobian * *sensitivity;
            }
            action_sensitivities[timestep] = step.local_action_jacobian;
            for action_timestep in 0..=timestep {
                action_jacobian[(timestep, 0, action_timestep, 0)] = action_sensitivities[action_timestep][0];
                action_jacobian[(timestep, 1, action_timestep, 0)] = action_sensitivities[action_timestep][1];
            }

            state = step.next_state;
            states.push(state);
            local_state_jacobians.push(step.local_state_jacobian);
            local_action_jacobians.push(step.local_action_jacobian);
            initial_state_jacobians.push(initial_state_sensitivity);
        }

        Ok(ControlledPendulumHorizon {
            states,
            local_state_jacobians,
            local_action_jacobians,
            initial_state_jacobians,
            action_jacobian,
        })
    }
}

/// Exact state path and derivatives for one planned action sequence.
#[derive(Clone, Debug, PartialEq)]
pub struct ControlledPendulumHorizon {
    pub states: Vec<PendulumState>,
    pub local_state_jacobians: Vec<Matrix2<f64>>,
    pub local_action_jacobians: Vec<Vector2<f64>>,
    pub initial_state_jacobians: Vec<Matrix2<f64>>,
    pub action_jacobian: Array4<f64>,
}

/// Ground-truth circular coordinates for the forcing driver.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ForcingPhaseCoordinates {
    pub sine: f64,
    pub cosine: f64,
}

/// Known low-dimensional torque driver `action = amplitude * sin(phase)`.
#[derive(Clone, Debug, PartialEq)]
pub struct SinusoidalTorqueDriver {
    amplitude: f64,
    phase_increment: f64,
}

impl SinusoidalTorqueDriver {
    /// Construct a driver whose phase is identifiable from two consecutive
    /// actions. Multiples of pi are refused because their two-action map loses
    /// the cosine coordinate.
    pub fn try_new(amplitude: f64, phase_increment: f64) -> Result<Self, String> {
        if !amplitude.is_finite() || amplitude <= 0.0 {
            return Err(format!("amplitude must be finite and > 0, got {amplitude}"));
        }
        if !phase_increment.is_finite() {
            return Err(format!("phase_increment must be finite, got {phase_increment}"));
        }
        if phase_increment.sin().abs() <= PHASE_RECONSTRUCTION_TOLERANCE {
            return Err(format!(
                "phase_increment must not be a multiple of pi; two action delays cannot identify phase when sin(increment) is zero, got {phase_increment}"
            ));
        }
        Ok(Self {
            amplitude,
            phase_increment,
        })
    }

    pub fn action(&self, phase: f64) -> Result<f64, String> {
        validate_phase(phase)?;
        Ok(self.amplitude * phase.sin())
    }

    pub fn phase_coordinates(&self, phase: f64) -> Result<ForcingPhaseCoordinates, String> {
        validate_phase(phase)?;
        Ok(ForcingPhaseCoordinates {
            sine: phase.sin(),
            cosine: phase.cos(),
        })
    }

    pub fn next_phase(&self, phase: f64) -> Result<f64, String> {
        validate_phase(phase)?;
        Ok((phase + self.phase_increment).rem_euclid(std::f64::consts::TAU))
    }

    /// Recover the current circular forcing coordinates from
    /// `(action[t - 1], action[t])`.
    pub fn phase_coordinates_from_two_actions(
        &self,
        previous_action: f64,
        current_action: f64,
    ) -> Result<ForcingPhaseCoordinates, String> {
        for (name, action) in [("previous_action", previous_action), ("current_action", current_action)] {
            if !action.is_finite() {
                return Err(format!("{name} must be finite, got {action}"));
            }
        }
        let sine = current_action / self.amplitude;
        let previous_sine = previous_action / self.amplitude;
        let cosine = (sine * self.phase_increment.cos() - previous_sine) / self.phase_increment.sin();
        let unit_circle_residual = (sine * sine + cosine * cosine - 1.0).abs();
        if unit_circle_residual > PHASE_RECONSTRUCTION_TOLERANCE {
            return Err(format!(
                "the two actions are inconsistent with this sinusoidal driver; unit-circle residual is {unit_circle_residual:e}"
            ));
        }
        Ok(ForcingPhaseCoordinates { sine, cosine })
    }
}

/// Partially observed trajectory with separate physical-state and forcing-state
/// truth. `observed_angles`, `latent_states`, and `forcing_coordinates` include
/// the initial time; `actions` has one value per transition.
#[derive(Clone, Debug, PartialEq)]
pub struct SinusoidallyForcedPendulumTrajectory {
    pub observed_angles: Vec<f64>,
    pub latent_states: Vec<PendulumState>,
    pub forcing_coordinates: Vec<ForcingPhaseCoordinates>,
    pub actions: Vec<f64>,
    pub response: ControlledPendulumHorizon,
}

/// Generate one known-forcing trajectory without hiding the driver or
/// integration identity inside a preset.
pub fn sinusoidally_forced_trajectory(
    plant: &ControlledPendulum,
    driver: &SinusoidalTorqueDriver,
    initial_state: PendulumState,
    initial_phase: f64,
    steps: usize,
) -> Result<SinusoidallyForcedPendulumTrajectory, String> {
    if steps == 0 {
        return Err("steps must be > 0 for a forced Pendulum trajectory".to_string());
    }
    validate_state(initial_state)?;
    validate_phase(initial_phase)?;

    let mut phase = initial_phase;
    let mut actions = Vec::with_capacity(steps);
    let mut forcing_coordinates = Vec::with_capacity(steps + 1);
    for _ in 0..steps {
        forcing_coordinates.push(driver.phase_coordinates(phase)?);
        actions.push(driver.action(phase)?);
        phase = driver.next_phase(phase)?;
    }
    forcing_coordinates.push(driver.phase_coordinates(phase)?);

    let response = plant.direct_horizon(initial_state, &actions)?;
    let mut latent_states = Vec::with_capacity(steps + 1);
    latent_states.push(initial_state);
    latent_states.extend(response.states.iter().copied());
    let observed_angles = latent_states.iter().map(|state| state.angle).collect();

    Ok(SinusoidallyForcedPendulumTrajectory {
        observed_angles,
        latent_states,
        forcing_coordinates,
        actions,
        response,
    })
}

fn validate_state(state: PendulumState) -> Result<(), String> {
    for (name, value) in [("angle", state.angle), ("angular_velocity", state.angular_velocity)] {
        if !value.is_finite() {
            return Err(format!("Pendulum {name} must be finite, got {value}"));
        }
    }
    Ok(())
}

fn validate_phase(phase: f64) -> Result<(), String> {
    if !phase.is_finite() {
        return Err(format!("forcing phase must be finite, got {phase}"));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    const FINITE_DIFFERENCE_STEP: f64 = 1e-6;
    const DERIVATIVE_TOLERANCE: f64 = 2e-7;

    fn plant() -> ControlledPendulum {
        ControlledPendulum::try_new(0.05, 0.15, 9.81, 0.8).unwrap()
    }

    fn finite_difference_state(
        plant: &ControlledPendulum,
        state: PendulumState,
        action: f64,
        coordinate: usize,
    ) -> Vector2<f64> {
        let mut plus = state;
        let mut minus = state;
        if coordinate == 0 {
            plus.angle += FINITE_DIFFERENCE_STEP;
            minus.angle -= FINITE_DIFFERENCE_STEP;
        } else {
            plus.angular_velocity += FINITE_DIFFERENCE_STEP;
            minus.angular_velocity -= FINITE_DIFFERENCE_STEP;
        }
        let plus = plant.step(plus, action).unwrap().next_state;
        let minus = plant.step(minus, action).unwrap().next_state;
        Vector2::new(
            (plus.angle - minus.angle) / (2.0 * FINITE_DIFFERENCE_STEP),
            (plus.angular_velocity - minus.angular_velocity) / (2.0 * FINITE_DIFFERENCE_STEP),
        )
    }

    #[test]
    fn local_jacobians_match_independent_finite_differences() {
        let plant = plant();
        let state = PendulumState {
            angle: 0.37,
            angular_velocity: -0.42,
        };
        let action = 0.63;
        let step = plant.step(state, action).unwrap();

        assert!((step.next_state.angle - 0.3415488815310593).abs() < 1e-15);
        assert!((step.next_state.angular_velocity - -0.5690223693788139).abs() < 1e-15);
        let cross_language_state_jacobian = Matrix2::new(0.977134671849012, 0.049625, -0.45730656301975997, 0.9925);
        let cross_language_action_jacobian = Vector2::new(0.0020000000000000005, 0.04000000000000001);
        assert!((step.local_state_jacobian - cross_language_state_jacobian).norm() < 1e-15);
        assert!((step.local_action_jacobian - cross_language_action_jacobian).norm() < 1e-15);

        for coordinate in 0..2 {
            let finite_difference = finite_difference_state(&plant, state, action, coordinate);
            let analytic = step.local_state_jacobian.column(coordinate);
            assert!((analytic[0] - finite_difference[0]).abs() < DERIVATIVE_TOLERANCE);
            assert!((analytic[1] - finite_difference[1]).abs() < DERIVATIVE_TOLERANCE);
        }

        let plus = plant.step(state, action + FINITE_DIFFERENCE_STEP).unwrap().next_state;
        let minus = plant.step(state, action - FINITE_DIFFERENCE_STEP).unwrap().next_state;
        let finite_difference = Vector2::new(
            (plus.angle - minus.angle) / (2.0 * FINITE_DIFFERENCE_STEP),
            (plus.angular_velocity - minus.angular_velocity) / (2.0 * FINITE_DIFFERENCE_STEP),
        );
        assert!((step.local_action_jacobian - finite_difference).norm() < DERIVATIVE_TOLERANCE);
    }

    #[test]
    fn direct_horizon_action_jacobian_is_causal_and_exact() {
        let plant = plant();
        let initial_state = PendulumState {
            angle: -0.2,
            angular_velocity: 0.4,
        };
        let actions = vec![-0.7, 0.2, 0.9, -0.1];
        let oracle = plant.direct_horizon(initial_state, &actions).unwrap();

        for action_timestep in 0..actions.len() {
            for future_timestep in 0..action_timestep {
                assert_eq!(oracle.action_jacobian[(future_timestep, 0, action_timestep, 0)], 0.0);
                assert_eq!(oracle.action_jacobian[(future_timestep, 1, action_timestep, 0)], 0.0);
            }

            let mut plus_actions = actions.clone();
            let mut minus_actions = actions.clone();
            plus_actions[action_timestep] += FINITE_DIFFERENCE_STEP;
            minus_actions[action_timestep] -= FINITE_DIFFERENCE_STEP;
            let plus = plant.direct_horizon(initial_state, &plus_actions).unwrap();
            let minus = plant.direct_horizon(initial_state, &minus_actions).unwrap();
            for future_timestep in action_timestep..actions.len() {
                let finite_angle = (plus.states[future_timestep].angle - minus.states[future_timestep].angle)
                    / (2.0 * FINITE_DIFFERENCE_STEP);
                let finite_velocity = (plus.states[future_timestep].angular_velocity
                    - minus.states[future_timestep].angular_velocity)
                    / (2.0 * FINITE_DIFFERENCE_STEP);
                assert!(
                    (oracle.action_jacobian[(future_timestep, 0, action_timestep, 0)] - finite_angle).abs()
                        < DERIVATIVE_TOLERANCE
                );
                assert!(
                    (oracle.action_jacobian[(future_timestep, 1, action_timestep, 0)] - finite_velocity).abs()
                        < DERIVATIVE_TOLERANCE
                );
            }
        }
    }

    #[test]
    fn two_action_delays_reconstruct_the_forcing_phase() {
        let driver = SinusoidalTorqueDriver::try_new(1.7, 0.31).unwrap();
        for phase in [0.2, 1.0, 2.4, 5.8] {
            let previous_action = driver.action(phase - 0.31).unwrap();
            let current_action = driver.action(phase).unwrap();
            let reconstructed = driver
                .phase_coordinates_from_two_actions(previous_action, current_action)
                .unwrap();
            let truth = driver.phase_coordinates(phase).unwrap();
            assert!((reconstructed.sine - truth.sine).abs() < 1e-12);
            assert!((reconstructed.cosine - truth.cosine).abs() < 1e-12);
        }
    }

    #[test]
    fn two_angle_delays_reconstruct_the_current_angular_velocity() {
        let plant = plant();
        let driver = SinusoidalTorqueDriver::try_new(0.6, 0.2).unwrap();
        let trajectory = sinusoidally_forced_trajectory(
            &plant,
            &driver,
            PendulumState {
                angle: 0.1,
                angular_velocity: -0.3,
            },
            0.7,
            5,
        )
        .unwrap();

        for timestep in 1..trajectory.latent_states.len() {
            let reconstructed = plant
                .angular_velocity_from_two_angles(
                    trajectory.observed_angles[timestep - 1],
                    trajectory.observed_angles[timestep],
                )
                .unwrap();
            assert!((reconstructed - trajectory.latent_states[timestep].angular_velocity).abs() < 1e-12);
        }
    }

    #[test]
    fn forced_trajectory_separates_observation_state_and_driver_truth() {
        let plant = plant();
        let driver = SinusoidalTorqueDriver::try_new(0.6, 0.2).unwrap();
        let initial_state = PendulumState {
            angle: 0.1,
            angular_velocity: -0.3,
        };
        let trajectory = sinusoidally_forced_trajectory(&plant, &driver, initial_state, 0.7, 5).unwrap();

        assert_eq!(trajectory.actions.len(), 5);
        assert_eq!(trajectory.observed_angles.len(), 6);
        assert_eq!(trajectory.latent_states.len(), 6);
        assert_eq!(trajectory.forcing_coordinates.len(), 6);
        assert_eq!(trajectory.observed_angles[0], initial_state.angle);
        assert_eq!(trajectory.latent_states[0].angular_velocity, -0.3);
        assert_eq!(trajectory.response.states.len(), 5);
        assert_eq!(trajectory.response.action_jacobian.shape(), &[5, 2, 5, 1]);
    }

    #[test]
    fn invalid_plant_and_unidentifiable_driver_are_refused() {
        assert!(
            ControlledPendulum::try_new(0.0, 0.1, 9.81, 1.0)
                .unwrap_err()
                .contains("timestep must be > 0")
        );
        assert!(
            ControlledPendulum::try_new(0.1, 0.1, 9.81, 0.0)
                .unwrap_err()
                .contains("torque_gain must be nonzero")
        );
        assert!(
            SinusoidalTorqueDriver::try_new(1.0, std::f64::consts::PI)
                .unwrap_err()
                .contains("multiple of pi")
        );
    }
}
