//! Exact discrete-time controlled linear-Gaussian oracle.
//!
//! This is deliberately a small mathematical boundary, not a second data
//! generator or scheduler. A caller supplies the scientific world explicitly:
//!
//! ```text
//! x[t + 1] = A x[t] + B u[t] + w[t],     w[t] ~ N(0, Q)
//! ```
//!
//! The module returns the conditional direct-horizon marginals and the action
//! Jacobians implied by that world. Fern's future worker decides how to turn
//! those values into a bank, artifact, or scored model result.

use nalgebra::{DMatrix, DVector, SymmetricEigen};
use ndarray::Array4;

/// Numerical acceptance for a covariance supplied as decimal floating-point
/// input. This guards matrix structure, not a scientific comparison tolerance.
const MATRIX_STRUCTURE_TOLERANCE: f64 = 1e-12;

/// A fully specified discrete-time controlled linear-Gaussian world.
#[derive(Clone, Debug, PartialEq)]
pub struct ControlledLinearGaussian {
    state_transition: DMatrix<f64>,
    action_response: DMatrix<f64>,
    process_covariance: DMatrix<f64>,
}

impl ControlledLinearGaussian {
    /// Construct a world only when all three matrices describe the same state
    /// space and `process_covariance` is symmetric positive semidefinite.
    pub fn try_new(
        state_transition: DMatrix<f64>,
        action_response: DMatrix<f64>,
        process_covariance: DMatrix<f64>,
    ) -> Result<Self, String> {
        let state_dim = state_transition.nrows();
        if state_dim == 0 || state_transition.ncols() != state_dim {
            return Err(format!(
                "state_transition must be a nonempty square matrix, got {}x{}",
                state_transition.nrows(),
                state_transition.ncols()
            ));
        }
        if action_response.nrows() != state_dim {
            return Err(format!(
                "action_response must have {state_dim} state rows, got {}",
                action_response.nrows()
            ));
        }
        if process_covariance.nrows() != state_dim || process_covariance.ncols() != state_dim {
            return Err(format!(
                "process_covariance must be {state_dim}x{state_dim}, got {}x{}",
                process_covariance.nrows(),
                process_covariance.ncols()
            ));
        }
        for (name, matrix) in [
            ("state_transition", &state_transition),
            ("action_response", &action_response),
            ("process_covariance", &process_covariance),
        ] {
            if let Some(value) = matrix.iter().copied().find(|value| !value.is_finite()) {
                return Err(format!("{name} must contain only finite values, found {value}"));
            }
        }

        let covariance_scale = process_covariance
            .iter()
            .fold(1.0_f64, |scale, value| scale.max(value.abs()));
        let asymmetry = matrix_asymmetry(&process_covariance);
        if asymmetry > MATRIX_STRUCTURE_TOLERANCE * covariance_scale {
            return Err(format!(
                "process_covariance must be symmetric within {MATRIX_STRUCTURE_TOLERANCE:e} relative tolerance; maximum asymmetry is {asymmetry:e}"
            ));
        }
        let smallest_eigenvalue = SymmetricEigen::new(process_covariance.clone())
            .eigenvalues
            .iter()
            .copied()
            .fold(f64::INFINITY, f64::min);
        if smallest_eigenvalue < -MATRIX_STRUCTURE_TOLERANCE * covariance_scale {
            return Err(format!(
                "process_covariance must be positive semidefinite; smallest eigenvalue is {smallest_eigenvalue:e}"
            ));
        }

        Ok(Self {
            state_transition,
            action_response,
            process_covariance,
        })
    }

    pub fn state_dimension(&self) -> usize {
        self.state_transition.nrows()
    }

    pub fn action_dimension(&self) -> usize {
        self.action_response.ncols()
    }

    /// Return direct-horizon conditional marginals given a fixed state and the
    /// complete future action sequence. `actions` is timestep-major:
    /// `[horizon, action_dimension]`.
    pub fn direct_horizon(
        &self,
        initial_state: &DVector<f64>,
        actions: &DMatrix<f64>,
    ) -> Result<DirectHorizonOracle, String> {
        if initial_state.len() != self.state_dimension() {
            return Err(format!(
                "initial_state must have {} values, got {}",
                self.state_dimension(),
                initial_state.len()
            ));
        }
        if let Some(value) = initial_state.iter().copied().find(|value| !value.is_finite()) {
            return Err(format!("initial_state must contain only finite values, found {value}"));
        }
        if actions.ncols() != self.action_dimension() {
            return Err(format!(
                "actions must have {} action columns, got {}",
                self.action_dimension(),
                actions.ncols()
            ));
        }
        if let Some(value) = actions.iter().copied().find(|value| !value.is_finite()) {
            return Err(format!("actions must contain only finite values, found {value}"));
        }

        let mut mean = initial_state.clone();
        // The initial state is conditioned on, so its conditional covariance is zero.
        let mut covariance = DMatrix::zeros(self.state_dimension(), self.state_dimension());
        let mut means = Vec::with_capacity(actions.nrows());
        let mut covariances = Vec::with_capacity(actions.nrows());
        let mut initial_state_jacobians = Vec::with_capacity(actions.nrows());
        let mut initial_state_sensitivity = DMatrix::identity(self.state_dimension(), self.state_dimension());
        let mut action_sensitivities =
            vec![DMatrix::zeros(self.state_dimension(), self.action_dimension()); actions.nrows()];
        let mut action_jacobian = Array4::zeros((
            actions.nrows(),
            self.state_dimension(),
            actions.nrows(),
            self.action_dimension(),
        ));
        for timestep in 0..actions.nrows() {
            let action = actions.row(timestep).transpose();
            mean = &self.state_transition * mean + &self.action_response * action;
            covariance =
                &self.state_transition * covariance * self.state_transition.transpose() + &self.process_covariance;
            initial_state_sensitivity = &self.state_transition * initial_state_sensitivity;
            for sensitivity in &mut action_sensitivities[..timestep] {
                *sensitivity = &self.state_transition * &*sensitivity;
            }
            action_sensitivities[timestep] = self.action_response.clone();
            for action_timestep in 0..=timestep {
                for state_coordinate in 0..self.state_dimension() {
                    for action_coordinate in 0..self.action_dimension() {
                        action_jacobian[(timestep, state_coordinate, action_timestep, action_coordinate)] =
                            action_sensitivities[action_timestep][(state_coordinate, action_coordinate)];
                    }
                }
            }
            means.push(mean.clone());
            covariances.push(covariance.clone());
            initial_state_jacobians.push(initial_state_sensitivity.clone());
        }
        Ok(DirectHorizonOracle {
            means,
            covariances,
            local_state_jacobians: vec![self.state_transition.clone(); actions.nrows()],
            local_action_jacobians: vec![self.action_response.clone(); actions.nrows()],
            initial_state_jacobians,
            action_jacobian,
        })
    }

    /// Exact derivative `d E[x_horizon] / d u_action_timestep`.
    ///
    /// Both indices are zero-based future steps. For example, at horizon 2 the
    /// action at step 0 has Jacobian `A * B`, while the action at step 1 has
    /// Jacobian `B`.
    pub fn action_jacobian(&self, horizon: usize, action_timestep: usize) -> Result<DMatrix<f64>, String> {
        if horizon == 0 {
            return Err("horizon must be >= 1 for an action Jacobian".to_string());
        }
        if action_timestep >= horizon {
            return Err(format!(
                "action_timestep={action_timestep} is outside direct horizon {horizon}"
            ));
        }
        let mut transition_power = DMatrix::identity(self.state_dimension(), self.state_dimension());
        for _ in 0..(horizon - action_timestep - 1) {
            transition_power = &self.state_transition * transition_power;
        }
        Ok(transition_power * &self.action_response)
    }
}

/// Exact conditional direct-horizon marginals, one entry per future action.
#[derive(Clone, Debug, PartialEq)]
pub struct DirectHorizonOracle {
    pub means: Vec<DVector<f64>>,
    pub covariances: Vec<DMatrix<f64>>,
    pub local_state_jacobians: Vec<DMatrix<f64>>,
    pub local_action_jacobians: Vec<DMatrix<f64>>,
    pub initial_state_jacobians: Vec<DMatrix<f64>>,
    pub action_jacobian: Array4<f64>,
}

impl DirectHorizonOracle {
    fn endpoint_action_response_matrix(&self, endpoint_timestep: usize) -> Result<DMatrix<f64>, String> {
        let shape = self.action_jacobian.shape();
        let horizon = shape[0];
        let state_dimension = shape[1];
        let action_dimension = shape[3];
        if endpoint_timestep >= horizon {
            return Err(format!(
                "endpoint_timestep={endpoint_timestep} is outside horizon {horizon}"
            ));
        }
        let action_steps = endpoint_timestep + 1;
        let mut action_response = DMatrix::zeros(state_dimension, action_steps * action_dimension);
        for state_coordinate in 0..state_dimension {
            for action_timestep in 0..action_steps {
                for action_coordinate in 0..action_dimension {
                    action_response[(state_coordinate, action_timestep * action_dimension + action_coordinate)] =
                        self.action_jacobian[(endpoint_timestep, state_coordinate, action_timestep, action_coordinate)];
                }
            }
        }
        Ok(action_response)
    }

    /// Return `Cov([x_1; …; x_H])` with shape `[H N, H N]`.
    ///
    /// Diagonal blocks are the endpoint-state covariances. Off-diagonal block
    /// `(g, h)` for `g > h` is the time-ordered transition from step `h` to
    /// step `g` applied to the earlier endpoint, because later process noise
    /// is independent of earlier state.
    pub fn joint_path_covariance(&self) -> Result<DMatrix<f64>, String> {
        let horizon = self.covariances.len();
        if horizon == 0 {
            return Err("joint path covariance requires a nonempty horizon".to_string());
        }
        let state_dimension = self.covariances[0].nrows();
        let width = horizon * state_dimension;
        let mut joint = DMatrix::zeros(width, width);
        for earlier in 0..horizon {
            let earlier_covariance = &self.covariances[earlier];
            let earlier_start = earlier * state_dimension;
            joint
                .view_mut((earlier_start, earlier_start), (state_dimension, state_dimension))
                .copy_from(earlier_covariance);
            let mut transition = DMatrix::identity(state_dimension, state_dimension);
            for later in (earlier + 1)..horizon {
                transition = &self.local_state_jacobians[later - 1] * transition;
                let block = &transition * earlier_covariance;
                let later_start = later * state_dimension;
                joint
                    .view_mut((later_start, earlier_start), (state_dimension, state_dimension))
                    .copy_from(&block);
                joint
                    .view_mut((earlier_start, later_start), (state_dimension, state_dimension))
                    .copy_from(&block.transpose());
            }
        }
        Ok(joint)
    }

    /// Return the finite-horizon controllability Gramian `W_c = G G^T` at
    /// one declared future endpoint.
    pub fn controllability_gramian(&self, endpoint_timestep: usize) -> Result<DMatrix<f64>, String> {
        let action_response = self.endpoint_action_response_matrix(endpoint_timestep)?;
        Ok(&action_response * action_response.transpose())
    }

    /// Solve `||G delta_u - desired||² + ridge ||delta_u||²` through the
    /// exact direct-horizon action map. The result is timestep-major.
    pub fn ridge_endpoint_action_intervention(
        &self,
        endpoint_timestep: usize,
        desired_displacement: &DVector<f64>,
        ridge: f64,
    ) -> Result<DMatrix<f64>, String> {
        if !ridge.is_finite() || ridge <= 0.0 {
            return Err(format!("ridge must be finite and > 0, got {ridge}"));
        }
        let action_response = self.endpoint_action_response_matrix(endpoint_timestep)?;
        let state_dimension = action_response.nrows();
        if desired_displacement.len() != state_dimension {
            return Err(format!(
                "desired_displacement must have {state_dimension} values, got {}",
                desired_displacement.len()
            ));
        }
        if let Some(value) = desired_displacement.iter().copied().find(|value| !value.is_finite()) {
            return Err(format!(
                "desired_displacement must contain only finite values, found {value}"
            ));
        }
        let regularized_gramian = &action_response * action_response.transpose()
            + DMatrix::identity(state_dimension, state_dimension) * ridge;
        let Some(dual_solution) = regularized_gramian.lu().solve(desired_displacement) else {
            return Err("regularized controllability solve failed".to_string());
        };
        let flattened_intervention = action_response.transpose() * dual_solution;
        Ok(DMatrix::from_row_slice(
            endpoint_timestep + 1,
            self.action_jacobian.shape()[3],
            flattened_intervention.as_slice(),
        ))
    }
}

/// A finite sequence of exact local `A_t/B_t/Q_t` operators.
///
/// Unlike [`ControlledLinearGaussian`], this object is explicitly
/// time-varying. Its sequence length is the only valid direct horizon, which
/// prevents a caller from silently extending the final local operator.
#[derive(Clone, Debug, PartialEq)]
pub struct ControlledLinearSequence {
    state_transitions: Vec<DMatrix<f64>>,
    action_responses: Vec<DMatrix<f64>>,
    process_covariances: Vec<DMatrix<f64>>,
    state_dimension: usize,
    action_dimension: usize,
}

impl ControlledLinearSequence {
    pub fn try_new(
        state_transitions: Vec<DMatrix<f64>>,
        action_responses: Vec<DMatrix<f64>>,
        process_covariances: Vec<DMatrix<f64>>,
    ) -> Result<Self, String> {
        if state_transitions.is_empty() {
            return Err("state_transitions must contain at least one future step".to_string());
        }
        let horizon = state_transitions.len();
        if action_responses.len() != horizon || process_covariances.len() != horizon {
            return Err(format!(
                "state_transitions, action_responses, and process_covariances must share one horizon; got {horizon}, {}, and {}",
                action_responses.len(),
                process_covariances.len()
            ));
        }

        let mut state_dimension = 0;
        let mut action_dimension = 0;
        for timestep in 0..horizon {
            let validated = ControlledLinearGaussian::try_new(
                state_transitions[timestep].clone(),
                action_responses[timestep].clone(),
                process_covariances[timestep].clone(),
            )
            .map_err(|error| format!("invalid controlled linear step {timestep}: {error}"))?;
            if timestep == 0 {
                state_dimension = validated.state_dimension();
                action_dimension = validated.action_dimension();
            } else if validated.state_dimension() != state_dimension || validated.action_dimension() != action_dimension
            {
                return Err(format!(
                    "controlled linear step {timestep} has state/action dimensions {}/{}, expected {state_dimension}/{action_dimension}",
                    validated.state_dimension(),
                    validated.action_dimension()
                ));
            }
        }

        Ok(Self {
            state_transitions,
            action_responses,
            process_covariances,
            state_dimension,
            action_dimension,
        })
    }

    pub fn horizon(&self) -> usize {
        self.state_transitions.len()
    }

    /// Evaluate exactly the declared LTV sequence and accumulate `Phi/G`.
    pub fn direct_horizon(
        &self,
        initial_state: &DVector<f64>,
        actions: &DMatrix<f64>,
    ) -> Result<DirectHorizonOracle, String> {
        if initial_state.len() != self.state_dimension {
            return Err(format!(
                "initial_state must have {} values, got {}",
                self.state_dimension,
                initial_state.len()
            ));
        }
        if actions.nrows() != self.horizon() || actions.ncols() != self.action_dimension {
            return Err(format!(
                "actions must have shape {}x{}, got {}x{}",
                self.horizon(),
                self.action_dimension,
                actions.nrows(),
                actions.ncols()
            ));
        }
        if let Some(value) = initial_state
            .iter()
            .chain(actions.iter())
            .copied()
            .find(|value| !value.is_finite())
        {
            return Err(format!(
                "initial state and actions must contain only finite values, found {value}"
            ));
        }

        let mut mean = initial_state.clone();
        let mut covariance = DMatrix::zeros(self.state_dimension, self.state_dimension);
        let mut means = Vec::with_capacity(self.horizon());
        let mut covariances = Vec::with_capacity(self.horizon());
        let mut initial_state_jacobians = Vec::with_capacity(self.horizon());
        let mut initial_state_sensitivity = DMatrix::identity(self.state_dimension, self.state_dimension);
        let mut action_sensitivities =
            vec![DMatrix::zeros(self.state_dimension, self.action_dimension); self.horizon()];
        let mut action_jacobian = Array4::zeros((
            self.horizon(),
            self.state_dimension,
            self.horizon(),
            self.action_dimension,
        ));

        for timestep in 0..self.horizon() {
            let state_transition = &self.state_transitions[timestep];
            let action_response = &self.action_responses[timestep];
            let action = actions.row(timestep).transpose();
            mean = state_transition * mean + action_response * action;
            covariance =
                state_transition * covariance * state_transition.transpose() + &self.process_covariances[timestep];
            initial_state_sensitivity = state_transition * initial_state_sensitivity;
            for sensitivity in &mut action_sensitivities[..timestep] {
                *sensitivity = state_transition * &*sensitivity;
            }
            action_sensitivities[timestep] = action_response.clone();
            for action_timestep in 0..=timestep {
                for state_coordinate in 0..self.state_dimension {
                    for action_coordinate in 0..self.action_dimension {
                        action_jacobian[(timestep, state_coordinate, action_timestep, action_coordinate)] =
                            action_sensitivities[action_timestep][(state_coordinate, action_coordinate)];
                    }
                }
            }
            means.push(mean.clone());
            covariances.push(covariance.clone());
            initial_state_jacobians.push(initial_state_sensitivity.clone());
        }

        Ok(DirectHorizonOracle {
            means,
            covariances,
            local_state_jacobians: self.state_transitions.clone(),
            local_action_jacobians: self.action_responses.clone(),
            initial_state_jacobians,
            action_jacobian,
        })
    }
}

/// Scalar controlled world with one unobserved actuator-memory state.
///
/// The transition order is explicit:
///
/// ```text
/// x[t + 1] = state_gain * x[t] + actuator_gain * r[t]
/// r[t + 1] = actuator_retention * r[t] + action_gain * u[t]
/// ```
///
/// Therefore `u[t]` cannot affect `x[t + 1]`; its first possible observed
/// effect is `x[t + 2]`. Past actions reconstruct `r[t]`, while future actions
/// define the causal intervention path.
#[derive(Clone, Debug, PartialEq)]
pub struct ScalarHiddenActuatorOracle {
    state_gain: f64,
    actuator_gain: f64,
    actuator_retention: f64,
    action_gain: f64,
}

impl ScalarHiddenActuatorOracle {
    /// Construct a stable nonnegative actuator-memory world.
    pub fn try_new(
        state_gain: f64,
        actuator_gain: f64,
        actuator_retention: f64,
        action_gain: f64,
    ) -> Result<Self, String> {
        for (name, value) in [
            ("state_gain", state_gain),
            ("actuator_gain", actuator_gain),
            ("actuator_retention", actuator_retention),
            ("action_gain", action_gain),
        ] {
            if !value.is_finite() {
                return Err(format!("{name} must be finite, got {value}"));
            }
        }
        if !(0.0..1.0).contains(&actuator_retention) {
            return Err(format!(
                "actuator_retention must satisfy 0 <= retention < 1, got {actuator_retention}"
            ));
        }
        Ok(Self {
            state_gain,
            actuator_gain,
            actuator_retention,
            action_gain,
        })
    }

    /// Reconstruct the current actuator state by applying past actions in
    /// chronological order from a declared earlier actuator state.
    pub fn actuator_state_after(&self, initial_actuator_state: f64, past_actions: &[f64]) -> Result<f64, String> {
        validate_scalar_and_actions(initial_actuator_state, "initial_actuator_state", past_actions)?;
        let mut actuator_state = initial_actuator_state;
        for action in past_actions {
            actuator_state = self.actuator_retention * actuator_state + self.action_gain * action;
        }
        Ok(actuator_state)
    }

    /// Return exact observed and hidden trajectories plus
    /// `d x[horizon] / d u[action_timestep]` for one future action path.
    pub fn direct_horizon(
        &self,
        initial_observed_state: f64,
        initial_actuator_state: f64,
        future_actions: &[f64],
    ) -> Result<ScalarHiddenActuatorHorizon, String> {
        validate_scalar_and_actions(initial_observed_state, "initial_observed_state", future_actions)?;
        if !initial_actuator_state.is_finite() {
            return Err(format!(
                "initial_actuator_state must be finite, got {initial_actuator_state}"
            ));
        }

        let horizon = future_actions.len();
        let mut observed_state = initial_observed_state;
        let mut actuator_state = initial_actuator_state;
        let mut observed_states = Vec::with_capacity(horizon);
        let mut actuator_states = Vec::with_capacity(horizon);
        let mut observed_action_jacobian = DMatrix::zeros(horizon, horizon);
        let mut observed_sensitivity = vec![0.0; horizon];
        let mut actuator_sensitivity = vec![0.0; horizon];

        for (timestep, action) in future_actions.iter().copied().enumerate() {
            let next_observed_state = self.state_gain * observed_state + self.actuator_gain * actuator_state;
            let next_actuator_state = self.actuator_retention * actuator_state + self.action_gain * action;

            for action_timestep in 0..timestep {
                observed_sensitivity[action_timestep] = self.state_gain * observed_sensitivity[action_timestep]
                    + self.actuator_gain * actuator_sensitivity[action_timestep];
                actuator_sensitivity[action_timestep] *= self.actuator_retention;
            }
            actuator_sensitivity[timestep] = self.action_gain;
            for action_timestep in 0..=timestep {
                observed_action_jacobian[(timestep, action_timestep)] = observed_sensitivity[action_timestep];
            }

            if !next_observed_state.is_finite() || !next_actuator_state.is_finite() {
                return Err(format!(
                    "controlled trajectory became non-finite at future timestep {timestep}"
                ));
            }
            observed_state = next_observed_state;
            actuator_state = next_actuator_state;
            observed_states.push(observed_state);
            actuator_states.push(actuator_state);
        }

        Ok(ScalarHiddenActuatorHorizon {
            observed_states,
            actuator_states,
            observed_action_jacobian,
        })
    }
}

/// Exact output of the scalar hidden-actuator world.
#[derive(Clone, Debug, PartialEq)]
pub struct ScalarHiddenActuatorHorizon {
    pub observed_states: Vec<f64>,
    pub actuator_states: Vec<f64>,
    pub observed_action_jacobian: DMatrix<f64>,
}

/// Scalar deterministic world whose action gain changes with current state.
///
/// ```text
/// x[t + 1] = state_gain * x[t]
///            + (action_offset + state_action_gain * x[t]) * u[t]
/// ```
#[derive(Clone, Debug, PartialEq)]
pub struct ScalarStateDependentControlOracle {
    state_gain: f64,
    action_offset: f64,
    state_action_gain: f64,
}

impl ScalarStateDependentControlOracle {
    pub fn try_new(state_gain: f64, action_offset: f64, state_action_gain: f64) -> Result<Self, String> {
        for (name, value) in [
            ("state_gain", state_gain),
            ("action_offset", action_offset),
            ("state_action_gain", state_action_gain),
        ] {
            if !value.is_finite() {
                return Err(format!("{name} must be finite, got {value}"));
            }
        }
        Ok(Self {
            state_gain,
            action_offset,
            state_action_gain,
        })
    }

    /// Return the exact trajectory and lower-triangular direct-horizon action
    /// Jacobian for one future action path.
    pub fn direct_horizon(
        &self,
        initial_state: f64,
        future_actions: &[f64],
    ) -> Result<ScalarStateDependentControlHorizon, String> {
        validate_scalar_and_actions(initial_state, "initial_state", future_actions)?;
        let horizon = future_actions.len();
        let mut state = initial_state;
        let mut states = Vec::with_capacity(horizon);
        let mut action_jacobian = DMatrix::zeros(horizon, horizon);
        let mut action_sensitivity = vec![0.0; horizon];

        for (timestep, action) in future_actions.iter().copied().enumerate() {
            let local_state_jacobian = self.state_gain + self.state_action_gain * action;
            let local_action_jacobian = self.action_offset + self.state_action_gain * state;
            for action_timestep in 0..timestep {
                action_sensitivity[action_timestep] *= local_state_jacobian;
            }
            action_sensitivity[timestep] = local_action_jacobian;

            let next_state = self.state_gain * state + local_action_jacobian * action;
            if !next_state.is_finite() {
                return Err(format!(
                    "controlled trajectory became non-finite at future timestep {timestep}"
                ));
            }
            state = next_state;
            states.push(state);
            for action_timestep in 0..=timestep {
                action_jacobian[(timestep, action_timestep)] = action_sensitivity[action_timestep];
            }
        }

        Ok(ScalarStateDependentControlHorizon {
            states,
            action_jacobian,
        })
    }
}

/// Exact output of the scalar state-dependent-control world.
#[derive(Clone, Debug, PartialEq)]
pub struct ScalarStateDependentControlHorizon {
    pub states: Vec<f64>,
    pub action_jacobian: DMatrix<f64>,
}

fn validate_scalar_and_actions(initial_value: f64, initial_name: &str, actions: &[f64]) -> Result<(), String> {
    if !initial_value.is_finite() {
        return Err(format!("{initial_name} must be finite, got {initial_value}"));
    }
    if let Some(action) = actions.iter().copied().find(|action| !action.is_finite()) {
        return Err(format!("actions must contain only finite values, found {action}"));
    }
    Ok(())
}

/// A frozen collection of complete action sequences with one shared horizon
/// and action dimension. It deliberately contains no state/context policy:
/// those identities belong to the calling experimental bank.
#[derive(Clone, Debug, PartialEq)]
pub struct ActionSequenceBank {
    sequences: Vec<DMatrix<f64>>,
    horizon: usize,
    action_dimension: usize,
}

impl ActionSequenceBank {
    /// Each sequence is timestep-major `[horizon, action_dimension]`.
    pub fn try_new(sequences: Vec<DMatrix<f64>>) -> Result<Self, String> {
        let Some(first) = sequences.first() else {
            return Err("action bank must contain at least one complete action sequence".to_string());
        };
        let horizon = first.nrows();
        let action_dimension = first.ncols();
        if horizon == 0 || action_dimension == 0 {
            return Err(format!(
                "action sequences must have nonzero horizon and action dimension, got {}x{}",
                horizon, action_dimension
            ));
        }
        for (sequence_index, sequence) in sequences.iter().enumerate() {
            if sequence.nrows() != horizon || sequence.ncols() != action_dimension {
                return Err(format!(
                    "action sequence {sequence_index} has shape {}x{}, expected {}x{}",
                    sequence.nrows(),
                    sequence.ncols(),
                    horizon,
                    action_dimension
                ));
            }
            if let Some(value) = sequence.iter().copied().find(|value| !value.is_finite()) {
                return Err(format!(
                    "action sequence {sequence_index} must contain only finite values, found {value}"
                ));
            }
        }
        Ok(Self {
            sequences,
            horizon,
            action_dimension,
        })
    }

    pub fn horizon(&self) -> usize {
        self.horizon
    }

    pub fn action_dimension(&self) -> usize {
        self.action_dimension
    }

    pub fn sequence_count(&self) -> usize {
        self.sequences.len()
    }

    pub fn sequences(&self) -> &[DMatrix<f64>] {
        &self.sequences
    }

    /// Measure the centered empirical action support. The caller pins the
    /// rank tolerance because this is an experimental identification choice,
    /// not a hidden numerical default.
    pub fn coverage(&self, rank_tolerance: f64) -> Result<ActionBankCoverage, String> {
        if !rank_tolerance.is_finite() || rank_tolerance <= 0.0 {
            return Err(format!("rank_tolerance must be finite and > 0, got {rank_tolerance}"));
        }
        let sample_count = self.sequence_count() * self.horizon;
        let mut values = DMatrix::zeros(sample_count, self.action_dimension);
        let mut row = 0;
        for sequence in &self.sequences {
            for timestep in 0..self.horizon {
                values.row_mut(row).copy_from(&sequence.row(timestep));
                row += 1;
            }
        }
        let mean = DVector::from_iterator(
            self.action_dimension,
            (0..self.action_dimension).map(|column| values.column(column).iter().sum::<f64>() / sample_count as f64),
        );
        for mut row in values.row_iter_mut() {
            row -= mean.transpose();
        }
        let covariance = if sample_count > 1 {
            values.transpose() * &values / (sample_count - 1) as f64
        } else {
            DMatrix::zeros(self.action_dimension, self.action_dimension)
        };
        let singular_values = values.svd(false, false).singular_values.as_slice().to_vec();
        let rank = singular_values.iter().filter(|value| **value > rank_tolerance).count();
        Ok(ActionBankCoverage {
            sample_count,
            action_dimension: self.action_dimension,
            mean,
            covariance,
            singular_values,
            rank,
            rank_tolerance,
        })
    }
}

/// Empirical coverage that determines which action-Jacobian directions the
/// bank identifies. `rank < action_dimension` means its complement is
/// scientifically `unidentified`, not an oracle or model error.
#[derive(Clone, Debug, PartialEq)]
pub struct ActionBankCoverage {
    pub sample_count: usize,
    pub action_dimension: usize,
    pub mean: DVector<f64>,
    pub covariance: DMatrix<f64>,
    pub singular_values: Vec<f64>,
    pub rank: usize,
    pub rank_tolerance: f64,
}

impl ActionBankCoverage {
    pub fn is_full_rank(&self) -> bool {
        self.rank == self.action_dimension
    }
}

/// Which functionals a report may claim. The oracle tier, not the caller’s
/// vocabulary, decides whether a requested report is authorized.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OracleReportKind {
    ConditionalMean,
    ConditionalCovariance,
    ActionJacobian,
    GaussianLogScoreProjection,
    RadialLaw,
    AngularAdequacy,
    Support,
    Multimodality,
    TrajectoryLaw,
    ReverseKl,
}

impl OracleReportKind {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::ConditionalMean => "conditional_mean",
            Self::ConditionalCovariance => "conditional_covariance",
            Self::ActionJacobian => "action_jacobian",
            Self::GaussianLogScoreProjection => "gaussian_log_score_projection",
            Self::RadialLaw => "radial_law",
            Self::AngularAdequacy => "angular_adequacy",
            Self::Support => "support",
            Self::Multimodality => "multimodality",
            Self::TrajectoryLaw => "trajectory_law",
            Self::ReverseKl => "reverse_kl",
        }
    }

    fn requires_full_law(self) -> bool {
        matches!(
            self,
            Self::RadialLaw
                | Self::AngularAdequacy
                | Self::Support
                | Self::Multimodality
                | Self::TrajectoryLaw
                | Self::ReverseKl
        )
    }
}

/// Declared mathematical object behind the oracle numbers.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OracleTier {
    FullLaw,
    ConditionalMoment,
    CertifiedTransform,
}

impl OracleTier {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::FullLaw => "full_law",
            Self::ConditionalMoment => "conditional_moment",
            Self::CertifiedTransform => "certified_transform",
        }
    }

    fn authorizes(self, report: OracleReportKind) -> bool {
        match self {
            Self::FullLaw => true,
            Self::ConditionalMoment => !report.requires_full_law(),
            Self::CertifiedTransform => !report.requires_full_law(),
        }
    }
}

/// Direct multi-horizon forecast versus a separately justified local kernel.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ForecastScope {
    DirectHorizon,
    LocalKernelRollout,
}

impl ForecastScope {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::DirectHorizon => "direct_horizon",
            Self::LocalKernelRollout => "local_kernel_rollout",
        }
    }
}

/// First Program II experiments observe the state itself.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ObservationProcess {
    DirectlyObservedState,
}

impl ObservationProcess {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::DirectlyObservedState => "directly_observed_state",
        }
    }
}

/// How process-noise scales are shared across a future path.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ProcessNoiseDependence {
    IndependentStep,
    SharedPathScale,
    Persistent,
}

impl ProcessNoiseDependence {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::IndependentStep => "independent_step",
            Self::SharedPathScale => "shared_path_scale",
            Self::Persistent => "persistent",
        }
    }
}

/// Whether the action bank is a frozen open-loop catalog or a policy path.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ActionProvenance {
    OpenLoop,
    PolicyConditioned,
}

impl ActionProvenance {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::OpenLoop => "open_loop",
            Self::PolicyConditioned => "policy_conditioned",
        }
    }
}

/// Unconditional action support versus a state-paired bank that does not exist yet.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ActionCoverageKind {
    UnconditionalAction,
    StateConditional,
}

impl ActionCoverageKind {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::UnconditionalAction => "unconditional_action",
            Self::StateConditional => "state_conditional",
        }
    }
}

/// `R = P` is the full-capacity control. `R < P` is a capacity-limited family.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GeometryCapacityClass {
    FullCapacity,
    CapacityLimited,
}

impl GeometryCapacityClass {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::FullCapacity => "full_capacity",
            Self::CapacityLimited => "capacity_limited",
        }
    }
}

/// Whether a Jacobian claim covers the whole `B` map or only excited directions.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ActionJacobianClaim {
    FullMap,
    ExcitedSubspaceOnly,
}

impl ActionJacobianClaim {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::FullMap => "full_map",
            Self::ExcitedSubspaceOnly => "excited_subspace_only",
        }
    }
}

/// Nested geometry identity. The oracle stays full-dimensional; `P` and `R`
/// name later projections, and `block_size` is only a scheduling knob.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GeometryIdentity {
    full_observable_dimension: usize,
    channel_groups: Vec<Vec<usize>>,
    block_dimension: usize,
    reflection_count: usize,
    block_size: usize,
    householder_backend: String,
    capacity_class: GeometryCapacityClass,
}

impl GeometryIdentity {
    pub fn try_new(
        full_observable_dimension: usize,
        channel_groups: Vec<Vec<usize>>,
        block_dimension: usize,
        reflection_count: usize,
        block_size: usize,
        householder_backend: impl Into<String>,
        capacity_class: GeometryCapacityClass,
    ) -> Result<Self, String> {
        if full_observable_dimension == 0 {
            return Err("full_observable_dimension must be > 0".to_string());
        }
        if block_dimension == 0 {
            return Err("block_dimension P must be > 0".to_string());
        }
        if block_size == 0 {
            return Err("block_size is a scheduling knob and must be > 0".to_string());
        }
        if reflection_count == 0 {
            return Err("reflection_count R must be > 0".to_string());
        }
        if reflection_count > block_dimension {
            return Err(format!(
                "reflection_count R={reflection_count} cannot exceed block_dimension P={block_dimension}"
            ));
        }
        match capacity_class {
            GeometryCapacityClass::FullCapacity if reflection_count != block_dimension => {
                return Err(format!(
                    "full_capacity requires R = P, got R={reflection_count} and P={block_dimension}"
                ));
            }
            GeometryCapacityClass::CapacityLimited if reflection_count >= block_dimension => {
                return Err(format!(
                    "capacity_limited requires R < P, got R={reflection_count} and P={block_dimension}"
                ));
            }
            _ => {}
        }
        let householder_backend = householder_backend.into();
        if householder_backend.trim().is_empty() {
            return Err("householder_backend identity is required".to_string());
        }
        if channel_groups.is_empty() {
            return Err("channel_groups must partition the full observable coordinates".to_string());
        }
        let mut seen = vec![false; full_observable_dimension];
        for (group_index, group) in channel_groups.iter().enumerate() {
            if group.len() != block_dimension {
                return Err(format!(
                    "channel group {group_index} has length {}, expected block_dimension P={block_dimension}",
                    group.len()
                ));
            }
            for &channel in group {
                if channel >= full_observable_dimension {
                    return Err(format!(
                        "channel {channel} is outside full_observable_dimension {full_observable_dimension}"
                    ));
                }
                if seen[channel] {
                    return Err(format!("channel {channel} appears in more than one group"));
                }
                seen[channel] = true;
            }
        }
        if seen.iter().any(|present| !present) {
            return Err(
                "channel_groups must be a partition of 0..full_observable_dimension; some coordinates are missing"
                    .to_string(),
            );
        }
        Ok(Self {
            full_observable_dimension,
            channel_groups,
            block_dimension,
            reflection_count,
            block_size,
            householder_backend,
            capacity_class,
        })
    }

    pub fn full_observable_dimension(&self) -> usize {
        self.full_observable_dimension
    }

    pub fn channel_groups(&self) -> &[Vec<usize>] {
        &self.channel_groups
    }

    pub fn block_dimension(&self) -> usize {
        self.block_dimension
    }

    pub fn reflection_count(&self) -> usize {
        self.reflection_count
    }

    pub fn block_size(&self) -> usize {
        self.block_size
    }

    pub fn householder_backend(&self) -> &str {
        &self.householder_backend
    }

    pub fn capacity_class(&self) -> GeometryCapacityClass {
        self.capacity_class
    }
}

/// Process-noise identity. A radial law needs replicates or an exact expectation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ProcessNoiseIdentity {
    dependence: ProcessNoiseDependence,
    replicate_count: usize,
    exact_analytic_expectation: bool,
}

impl ProcessNoiseIdentity {
    pub fn try_new(
        dependence: ProcessNoiseDependence,
        replicate_count: usize,
        exact_analytic_expectation: bool,
    ) -> Result<Self, String> {
        if !exact_analytic_expectation && replicate_count == 0 {
            return Err(
                "process-noise provenance requires replicate_count > 0 or exact_analytic_expectation".to_string(),
            );
        }
        Ok(Self {
            dependence,
            replicate_count,
            exact_analytic_expectation,
        })
    }

    pub fn dependence(&self) -> ProcessNoiseDependence {
        self.dependence
    }

    pub fn replicate_count(&self) -> usize {
        self.replicate_count
    }

    pub fn exact_analytic_expectation(&self) -> bool {
        self.exact_analytic_expectation
    }

    fn supports_radial_law(self) -> bool {
        self.exact_analytic_expectation || self.replicate_count >= 2
    }
}

/// Optional certified-transform numerical bound. Both values must be pinned together.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct OracleErrorBound {
    inversion_error_bound: f64,
    smallest_claimed_contrast: f64,
}

impl OracleErrorBound {
    pub fn try_new(inversion_error_bound: f64, smallest_claimed_contrast: f64) -> Result<Self, String> {
        if !inversion_error_bound.is_finite() || inversion_error_bound <= 0.0 {
            return Err(format!(
                "inversion_error_bound must be finite and > 0, got {inversion_error_bound}"
            ));
        }
        if !smallest_claimed_contrast.is_finite() || smallest_claimed_contrast <= 0.0 {
            return Err(format!(
                "smallest_claimed_contrast must be finite and > 0, got {smallest_claimed_contrast}"
            ));
        }
        if inversion_error_bound >= smallest_claimed_contrast {
            return Err(format!(
                "oracle numerical error {inversion_error_bound} is not smaller than the smallest claimed contrast {smallest_claimed_contrast}"
            ));
        }
        Ok(Self {
            inversion_error_bound,
            smallest_claimed_contrast,
        })
    }

    pub fn inversion_error_bound(&self) -> f64 {
        self.inversion_error_bound
    }

    pub fn smallest_claimed_contrast(&self) -> f64 {
        self.smallest_claimed_contrast
    }
}

/// Complete Program II case identity. Construction refuses missing or contradictory fields.
#[derive(Clone, Debug, PartialEq)]
pub struct ControlledCaseIdentity {
    physics_context_id: String,
    oracle_tier: OracleTier,
    forecast_scope: ForecastScope,
    observation_process: ObservationProcess,
    action_provenance: ActionProvenance,
    action_coverage_kind: ActionCoverageKind,
    action_jacobian_claim: ActionJacobianClaim,
    process_noise: ProcessNoiseIdentity,
    geometry: GeometryIdentity,
    requested_reports: Vec<OracleReportKind>,
    oracle_error_bound: Option<OracleErrorBound>,
}

impl ControlledCaseIdentity {
    pub fn try_new(
        physics_context_id: impl Into<String>,
        oracle_tier: OracleTier,
        forecast_scope: ForecastScope,
        observation_process: ObservationProcess,
        action_provenance: ActionProvenance,
        action_coverage_kind: ActionCoverageKind,
        action_jacobian_claim: ActionJacobianClaim,
        process_noise: ProcessNoiseIdentity,
        geometry: GeometryIdentity,
        requested_reports: Vec<OracleReportKind>,
        oracle_error_bound: Option<OracleErrorBound>,
    ) -> Result<Self, String> {
        let physics_context_id = physics_context_id.into();
        if physics_context_id.trim().is_empty() {
            return Err("physics_context identity is required".to_string());
        }
        if requested_reports.is_empty() {
            return Err("requested_reports must name at least one authorized metric".to_string());
        }
        if action_coverage_kind == ActionCoverageKind::StateConditional {
            return Err(
                "state-conditional action coverage requires a state-paired bank; the current ActionSequenceBank is unconditional"
                    .to_string(),
            );
        }
        if oracle_tier == OracleTier::CertifiedTransform && oracle_error_bound.is_none() {
            return Err("certified_transform requires a pinned inversion error bound and contrast".to_string());
        }
        for report in &requested_reports {
            if !oracle_tier.authorizes(*report) {
                return Err(format!(
                    "oracle tier {} does not authorize report {}",
                    oracle_tier.as_str(),
                    report.as_str()
                ));
            }
            if *report == OracleReportKind::TrajectoryLaw && forecast_scope == ForecastScope::DirectHorizon {
                return Err(
                    "direct_horizon scope cannot authorize a trajectory_law report; that is scope confusion"
                        .to_string(),
                );
            }
            if *report == OracleReportKind::RadialLaw && !process_noise.supports_radial_law() {
                return Err(
                    "radial_law requires process-noise replicate_count >= 2 or an exact analytic expectation"
                        .to_string(),
                );
            }
        }
        Ok(Self {
            physics_context_id,
            oracle_tier,
            forecast_scope,
            observation_process,
            action_provenance,
            action_coverage_kind,
            action_jacobian_claim,
            process_noise,
            geometry,
            requested_reports,
            oracle_error_bound,
        })
    }

    pub fn physics_context_id(&self) -> &str {
        &self.physics_context_id
    }

    pub fn oracle_tier(&self) -> OracleTier {
        self.oracle_tier
    }

    pub fn forecast_scope(&self) -> ForecastScope {
        self.forecast_scope
    }

    pub fn observation_process(&self) -> ObservationProcess {
        self.observation_process
    }

    pub fn action_provenance(&self) -> ActionProvenance {
        self.action_provenance
    }

    pub fn action_coverage_kind(&self) -> ActionCoverageKind {
        self.action_coverage_kind
    }

    pub fn action_jacobian_claim(&self) -> ActionJacobianClaim {
        self.action_jacobian_claim
    }

    pub fn process_noise(&self) -> ProcessNoiseIdentity {
        self.process_noise
    }

    pub fn geometry(&self) -> &GeometryIdentity {
        &self.geometry
    }

    pub fn requested_reports(&self) -> &[OracleReportKind] {
        &self.requested_reports
    }

    pub fn oracle_error_bound(&self) -> Option<OracleErrorBound> {
        self.oracle_error_bound
    }

    /// Bind a frozen action bank to this identity. Missing excitation cannot
    /// be scored as a model failure when the claim is a full action map.
    pub fn admit(&self, coverage: ActionBankCoverage) -> Result<AdmittedControlledCase, String> {
        if self.action_jacobian_claim == ActionJacobianClaim::FullMap && !coverage.is_full_rank() {
            return Err(format!(
                "full action-Jacobian claim is unidentified: action-bank rank {} < action_dimension {}",
                coverage.rank, coverage.action_dimension
            ));
        }
        let action_jacobian_support = if coverage.is_full_rank() {
            ActionJacobianSupport::FullMap
        } else {
            ActionJacobianSupport::ExcitedSubspace {
                rank: coverage.rank,
                unidentified_dimension: coverage.action_dimension - coverage.rank,
            }
        };
        Ok(AdmittedControlledCase {
            identity: self.clone(),
            coverage,
            action_jacobian_support,
        })
    }
}

/// Identified action-Jacobian support after the bank coverage is bound.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ActionJacobianSupport {
    FullMap,
    ExcitedSubspace { rank: usize, unidentified_dimension: usize },
}

/// A case that passed the identity and excitation gates. It still does not
/// select Level-0 matrices or authorize a training job.
#[derive(Clone, Debug, PartialEq)]
pub struct AdmittedControlledCase {
    pub identity: ControlledCaseIdentity,
    pub coverage: ActionBankCoverage,
    pub action_jacobian_support: ActionJacobianSupport,
}

/// The Design-pinned Level-0A canary. The numbers are the claim.
///
/// `A, B, Q, x0, H=8` and the 33-sequence balanced impulse bank are taken
/// from `docs/programs/designs/2026-08-15-controlled-dynamics-oracle-contract.md`.
/// The fixture dies if the iterative oracle misses the independent power-sum
/// under the mixed tolerance, if instantaneous rank is not 2, if the
/// sequence-design rank is not 16, or if the four declared defects still
/// reproduce the pinned closed form.
pub struct Level0ACanary;

/// Measured values from [`Level0ACanary::evaluate`].
#[derive(Clone, Debug, PartialEq)]
pub struct Level0ACanaryReport {
    pub instantaneous_rank: usize,
    pub sequence_design_rank: usize,
    pub closed_form_mixed_residual: f64,
    pub block_seam_nats: f64,
    pub orientation_capacity_nats: f64,
    pub detected_defects: Vec<&'static str>,
}

impl Level0ACanary {
    pub const PHYSICS_CONTEXT_ID: &'static str = "level0a-controlled-lgssm";
    pub const HORIZON: usize = 8;
    pub const RANK_TOLERANCE: f64 = 1e-8;
    pub const MIXED_ABS: f64 = 1e-12;
    pub const MIXED_REL: f64 = 1e-10;
    pub const IMPULSE_AMPLITUDE: f64 = 0.5;

    pub fn state_transition() -> DMatrix<f64> {
        DMatrix::from_row_slice(2, 2, &[0.75, 0.20, -0.10, 0.65])
    }

    pub fn action_response() -> DMatrix<f64> {
        DMatrix::from_row_slice(2, 2, &[0.50, -0.10, 0.20, 0.40])
    }

    pub fn process_covariance() -> DMatrix<f64> {
        DMatrix::from_row_slice(2, 2, &[0.040, 0.012, 0.012, 0.025])
    }

    pub fn initial_state() -> DVector<f64> {
        DVector::from_vec(vec![0.40, -0.30])
    }

    pub fn world() -> Result<ControlledLinearGaussian, String> {
        ControlledLinearGaussian::try_new(
            Self::state_transition(),
            Self::action_response(),
            Self::process_covariance(),
        )
    }

    pub fn action_bank() -> Result<ActionSequenceBank, String> {
        ActionSequenceBank::try_new(impulse_action_sequences(Self::HORIZON, 2, Self::IMPULSE_AMPLITUDE))
    }

    pub fn identity() -> Result<ControlledCaseIdentity, String> {
        ControlledCaseIdentity::try_new(
            Self::PHYSICS_CONTEXT_ID,
            OracleTier::FullLaw,
            ForecastScope::DirectHorizon,
            ObservationProcess::DirectlyObservedState,
            ActionProvenance::OpenLoop,
            ActionCoverageKind::UnconditionalAction,
            ActionJacobianClaim::FullMap,
            ProcessNoiseIdentity::try_new(ProcessNoiseDependence::IndependentStep, 0, true)?,
            GeometryIdentity::try_new(
                2,
                vec![vec![0, 1]],
                2,
                2,
                8,
                "cpu-reference",
                GeometryCapacityClass::FullCapacity,
            )?,
            vec![
                OracleReportKind::ConditionalMean,
                OracleReportKind::ConditionalCovariance,
                OracleReportKind::ActionJacobian,
                OracleReportKind::GaussianLogScoreProjection,
            ],
            None,
        )
    }

    /// Run the canary. `Err` is a failed falsification of this fixture.
    pub fn evaluate() -> Result<Level0ACanaryReport, String> {
        let world = Self::world()?;
        let bank = Self::action_bank()?;
        let coverage = bank.coverage(Self::RANK_TOLERANCE)?;
        if coverage.rank != 2 {
            return Err(format!("instantaneous action rank must be 2, got {}", coverage.rank));
        }
        let sequence_design_rank = sequence_design_rank(&bank, Self::RANK_TOLERANCE)?;
        if sequence_design_rank != 16 {
            return Err(format!("sequence-design rank must be 16, got {sequence_design_rank}"));
        }
        Self::identity()?.admit(coverage)?;

        let residual = fixture_closed_form_residual(&world, &bank, &Self::initial_state())?;
        if residual > 0.0 {
            return Err(format!(
                "iterative oracle missed the pinned closed form; mixed residual {residual:e}"
            ));
        }

        let detected_defects = detect_level0a_defects(&world, &bank)?;
        let expected = [
            "wrong_B01",
            "diagonal_Q",
            "shifted_impulse",
            "dropped_action_axis",
            "split_block_seam",
        ];
        for name in expected {
            if !detected_defects.iter().any(|detected| *detected == name) {
                return Err(format!("declared defect {name} was not detected"));
            }
        }

        let geometry = geometry_gaps(&Self::process_covariance(), &[vec![0, 1]])?;
        if geometry.block_seam_nats > 1e-10 {
            return Err(format!(
                "R=P full-block seam must be numerical zero, got {} nats",
                geometry.block_seam_nats
            ));
        }
        if geometry.orientation_capacity_nats > 1e-10 {
            return Err(format!(
                "R=P orientation capacity must be numerical zero, got {} nats",
                geometry.orientation_capacity_nats
            ));
        }

        Ok(Level0ACanaryReport {
            instantaneous_rank: 2,
            sequence_design_rank,
            closed_form_mixed_residual: residual,
            block_seam_nats: geometry.block_seam_nats,
            orientation_capacity_nats: geometry.orientation_capacity_nats,
            detected_defects,
        })
    }
}

/// Action-conditioned covariance on the Level-0A mean mechanics.
///
/// `Q(a) = s(a1)^2 R(θ(a2)) Q0 R(θ(a2))^T` with `s = 1 + 0.5 a1` and
/// `θ = 0.4 a2`. Mean and Jacobians stay at the 0A law. The fixture dies if
/// a constant-`Q0` world, a swapped scale/rotation assignment, or a
/// `B[0,1] = 0` world still matches the pinned 0B closed form.
pub struct Level0BCanary;

/// Measured values from [`Level0BCanary::evaluate`].
#[derive(Clone, Debug, PartialEq)]
pub struct Level0BCanaryReport {
    pub mean_matches_level0a: bool,
    pub closed_form_mixed_residual: f64,
    pub block_seam_nats: f64,
    pub orientation_capacity_nats: f64,
    pub detected_defects: Vec<&'static str>,
}

impl Level0BCanary {
    pub const PHYSICS_CONTEXT_ID: &'static str = "level0b-action-conditioned-covariance";
    pub const SCALE_GAIN: f64 = 0.5;
    pub const ANGLE_GAIN: f64 = 0.4;

    pub fn identity() -> Result<ControlledCaseIdentity, String> {
        ControlledCaseIdentity::try_new(
            Self::PHYSICS_CONTEXT_ID,
            OracleTier::FullLaw,
            ForecastScope::DirectHorizon,
            ObservationProcess::DirectlyObservedState,
            ActionProvenance::OpenLoop,
            ActionCoverageKind::UnconditionalAction,
            ActionJacobianClaim::FullMap,
            ProcessNoiseIdentity::try_new(ProcessNoiseDependence::IndependentStep, 0, true)?,
            GeometryIdentity::try_new(
                2,
                vec![vec![0, 1]],
                2,
                2,
                8,
                "cpu-reference",
                GeometryCapacityClass::FullCapacity,
            )?,
            vec![
                OracleReportKind::ConditionalMean,
                OracleReportKind::ConditionalCovariance,
                OracleReportKind::ActionJacobian,
                OracleReportKind::GaussianLogScoreProjection,
            ],
            None,
        )
    }

    pub fn evaluate() -> Result<Level0BCanaryReport, String> {
        let bank = Level0ACanary::action_bank()?;
        let coverage = bank.coverage(Level0ACanary::RANK_TOLERANCE)?;
        Self::identity()?.admit(coverage)?;
        let initial_state = Level0ACanary::initial_state();
        let residual = level0b_closed_form_residual(&bank, &initial_state, false)?;
        if residual > 0.0 {
            return Err(format!(
                "iterative 0B oracle missed the action-conditioned closed form; mixed residual {residual:e}"
            ));
        }
        if !level0b_mean_matches_level0a(&bank, &initial_state)? {
            return Err("Level 0B mean must stay on the Level 0A action response".to_string());
        }
        if !level0b_scale_preserves_orientation()? {
            return Err("scale-only action must keep Q0 principal directions".to_string());
        }
        if !level0b_rotation_preserves_spectrum()? {
            return Err("rotation-only action must keep Q0 eigenvalues".to_string());
        }
        let detected_defects = detect_level0b_defects(&bank, &initial_state)?;
        for name in ["constant_Q0", "swapped_scale_rotation", "wrong_B01", "covariance_lag"] {
            if !detected_defects.iter().any(|detected| *detected == name) {
                return Err(format!("declared 0B defect {name} was not detected"));
            }
        }
        let rotated = geometry_gaps(&level0b_covariance(0.0, 0.5), &[vec![0, 1]])?;
        if rotated.orientation_capacity_nats > 1e-10 {
            return Err(format!(
                "R=P must close the rotated 0B covariance, got {} nats",
                rotated.orientation_capacity_nats
            ));
        }

        Ok(Level0BCanaryReport {
            mean_matches_level0a: true,
            closed_form_mixed_residual: residual,
            block_seam_nats: rotated.block_seam_nats,
            orientation_capacity_nats: rotated.orientation_capacity_nats,
            detected_defects,
        })
    }
}

/// Action-conditioned radial mixing on a spherical world.
///
/// `A = 0.7 I`, `Q = 0.0225 I`, and five constant action paths select
/// `W=1`, `Gamma(8,8)`, or `Gamma(2,2)`. Shared-path `E[R^4]` is `8`, `9`,
/// `12`. Independent-step `E[R^4]` matches at horizon 1 and drops by
/// horizon 8. Action must change the radial law.
pub struct Level0CCanary;

/// Measured values from [`Level0CCanary::evaluate`].
#[derive(Clone, Debug, PartialEq)]
pub struct Level0CCanaryReport {
    pub shared_fourth_moments: [f64; 3],
    pub independent_horizon8_fourth_moments: [f64; 3],
    pub second_moment: f64,
    pub characteristic_function_residual: f64,
    pub sign_matched: bool,
    pub detected_defects: Vec<&'static str>,
}

impl Level0CCanary {
    pub const PHYSICS_CONTEXT_ID: &'static str = "level0c-action-conditioned-radial";
    pub const DRIFT: f64 = 0.7;
    pub const SCALE: f64 = 0.15;
    pub const SHARED_FOURTH: [f64; 3] = [8.0, 9.0, 12.0];
    pub const INDEPENDENT_H8_FOURTH: [f64; 3] = [8.0, 8.344564470901103, 9.37825788360441];
    pub const SECOND_MOMENT: f64 = 2.0;
    pub const CF_RADII: [f64; 4] = [0.25, 0.5, 1.0, 2.0];

    pub fn state_transition() -> DMatrix<f64> {
        DMatrix::from_diagonal(&DVector::from_vec(vec![Self::DRIFT, Self::DRIFT]))
    }

    pub fn process_covariance() -> DMatrix<f64> {
        let variance = Self::SCALE * Self::SCALE;
        DMatrix::from_diagonal(&DVector::from_vec(vec![variance, variance]))
    }

    pub fn world() -> Result<ControlledLinearGaussian, String> {
        ControlledLinearGaussian::try_new(
            Self::state_transition(),
            Level0ACanary::action_response(),
            Self::process_covariance(),
        )
    }

    pub fn action_bank() -> Result<ActionSequenceBank, String> {
        let horizon = Level0ACanary::HORIZON;
        ActionSequenceBank::try_new(
            [(0.0, 0.0), (0.5, 0.0), (-0.5, 0.0), (0.0, 0.5), (0.0, -0.5)]
                .into_iter()
                .map(|(action1, action2)| {
                    let mut sequence = DMatrix::zeros(horizon, 2);
                    for timestep in 0..horizon {
                        sequence[(timestep, 0)] = action1;
                        sequence[(timestep, 1)] = action2;
                    }
                    sequence
                })
                .collect(),
        )
    }

    pub fn identity(dependence: ProcessNoiseDependence) -> Result<ControlledCaseIdentity, String> {
        ControlledCaseIdentity::try_new(
            Self::PHYSICS_CONTEXT_ID,
            OracleTier::FullLaw,
            ForecastScope::DirectHorizon,
            ObservationProcess::DirectlyObservedState,
            ActionProvenance::OpenLoop,
            ActionCoverageKind::UnconditionalAction,
            ActionJacobianClaim::FullMap,
            ProcessNoiseIdentity::try_new(dependence, 0, true)?,
            GeometryIdentity::try_new(
                2,
                vec![vec![0, 1]],
                2,
                2,
                8,
                "cpu-reference",
                GeometryCapacityClass::FullCapacity,
            )?,
            vec![
                OracleReportKind::ConditionalMean,
                OracleReportKind::ConditionalCovariance,
                OracleReportKind::ActionJacobian,
                OracleReportKind::RadialLaw,
            ],
            None,
        )
    }

    pub fn evaluate() -> Result<Level0CCanaryReport, String> {
        let bank = Self::action_bank()?;
        let coverage = bank.coverage(Level0ACanary::RANK_TOLERANCE)?;
        if coverage.rank != 2 {
            return Err(format!(
                "Level 0C bank must excite both action axes, rank {}",
                coverage.rank
            ));
        }
        Self::identity(ProcessNoiseDependence::IndependentStep)?.admit(coverage.clone())?;
        Self::identity(ProcessNoiseDependence::SharedPathScale)?.admit(coverage)?;

        let world = Self::world()?;
        let initial_state = Level0ACanary::initial_state();
        let moment_residual = fixture_closed_form_residual(&world, &bank, &initial_state)?;
        if moment_residual > 0.0 {
            return Err(format!(
                "Level 0C moment contract missed the spherical closed form; mixed residual {moment_residual:e}"
            ));
        }

        let mut shared_fourth = [0.0; 3];
        let mut independent_h8 = [0.0; 3];
        let mut cf_residual = 0.0_f64;
        for (sequence, arm) in bank.sequences().iter().zip([0, 1, 1, 2, 2]) {
            let alpha = level0c_alpha(sequence);
            let (shared, independent, independent_h1) = level0c_radius_moments(alpha);
            shared_fourth[arm] = shared;
            independent_h8[arm] = independent;
            if mixed_residual(shared, Self::SHARED_FOURTH[arm]) > 0.0 {
                return Err(format!(
                    "shared E[R^4] for arm {arm} is {shared}, expected {}",
                    Self::SHARED_FOURTH[arm]
                ));
            }
            if mixed_residual(independent, Self::INDEPENDENT_H8_FOURTH[arm]) > 0.0 {
                return Err(format!(
                    "independent horizon-8 E[R^4] for arm {arm} is {independent}, expected {}",
                    Self::INDEPENDENT_H8_FOURTH[arm]
                ));
            }
            if mixed_residual(independent_h1, shared) > 0.0 {
                return Err(format!("horizon-1 independent E[R^4] must match shared for arm {arm}"));
            }
            cf_residual = cf_residual.max(level0c_cf_residual(alpha));
        }
        if cf_residual > 0.0 {
            return Err(format!(
                "radial characteristic function missed the pinned formula; mixed residual {cf_residual:e}"
            ));
        }
        if !level0c_sign_pairs_match(&bank)? {
            return Err("sign-matched Level 0C paths must share the centered radial law".to_string());
        }

        let detected_defects = detect_level0c_defects(&bank)?;
        for name in [
            "gaussian_W",
            "swapped_alpha",
            "rate_not_shape",
            "independent_labeled_shared",
        ] {
            if !detected_defects.iter().any(|detected| *detected == name) {
                return Err(format!("declared 0C defect {name} was not detected"));
            }
        }
        Ok(Level0CCanaryReport {
            shared_fourth_moments: shared_fourth,
            independent_horizon8_fourth_moments: independent_h8,
            second_moment: Self::SECOND_MOMENT,
            characteristic_function_residual: cf_residual,
            sign_matched: true,
            detected_defects,
        })
    }
}

/// Action-free positive slice of the Level 0C radial mechanism.
///
/// The state law is `x[t+1] = 0.7 x[t] + 0.15 sqrt(W[t]) z[t]`, with
/// `W[t] ~ Gamma(shape=4, rate=4)` independently at every step and
/// `z[t] ~ N(0, I2)`. Since `E[W]=1`, its process covariance is exactly the
/// same `0.0225 I2` as the Gaussian control. Only the innovation magnitude law
/// changes: whitened `E[R^4]` is 10 at one step and attenuates toward the
/// Gaussian value 8 as independent innovations accumulate.
pub struct ActionIndependentGammaAttenuationCanary;

/// Exact moment contract for [`ActionIndependentGammaAttenuationCanary`].
#[derive(Clone, Debug, PartialEq)]
pub struct ActionIndependentGammaAttenuationReport {
    pub one_step_second_moment: f64,
    pub one_step_fourth_moment: f64,
    pub horizon8_fourth_moment: f64,
    pub horizon24_fourth_moment: f64,
    pub process_covariance: DMatrix<f64>,
}

impl ActionIndependentGammaAttenuationCanary {
    pub const SOURCE_NAME: &'static str = "radial_attenuation_gamma4";
    pub const DRIFT: f64 = Level0CCanary::DRIFT;
    pub const SCALE: f64 = Level0CCanary::SCALE;
    pub const GAMMA_SHAPE: f64 = 4.0;
    pub const GAMMA_RATE: f64 = 4.0;
    pub const HORIZON: usize = Level0ACanary::HORIZON;
    pub const ONE_STEP_SECOND_MOMENT: f64 = 2.0;
    pub const GAUSSIAN_FOURTH_MOMENT: f64 = 8.0;
    pub const ONE_STEP_FOURTH_MOMENT: f64 = 10.0;
    pub const HORIZON8_FOURTH_MOMENT: f64 = 8.689128941802204;
    /// Smallest one-block horizon the production Fern patch selector accepts.
    pub const HORIZON24_FOURTH_MOMENT: f64 = 8.684563808640855;

    pub fn process_covariance() -> DMatrix<f64> {
        Level0CCanary::process_covariance()
    }

    pub fn evaluate() -> Result<ActionIndependentGammaAttenuationReport, String> {
        let (shared, independent_horizon8, independent_horizon1) = level0c_radius_moments(Some(Self::GAMMA_SHAPE));
        if mixed_residual(independent_horizon1, Self::ONE_STEP_FOURTH_MOMENT) > 0.0 {
            return Err(format!(
                "Gamma(4,4) one-step E[R^4] is {independent_horizon1}, expected {}",
                Self::ONE_STEP_FOURTH_MOMENT
            ));
        }
        if mixed_residual(independent_horizon8, Self::HORIZON8_FOURTH_MOMENT) > 0.0 {
            return Err(format!(
                "Gamma(4,4) horizon-8 E[R^4] is {independent_horizon8}, expected {}",
                Self::HORIZON8_FOURTH_MOMENT
            ));
        }
        if independent_horizon8 <= Self::GAUSSIAN_FOURTH_MOMENT || independent_horizon8 >= shared {
            return Err(
                "independent Gamma attenuation must lie strictly between Gaussian and shared-path tails".to_string(),
            );
        }
        let independent_horizon24 = independent_radius_fourth_moment(Self::GAMMA_SHAPE, 24)?;
        if mixed_residual(independent_horizon24, Self::HORIZON24_FOURTH_MOMENT) > 0.0 {
            return Err(format!(
                "Gamma(4,4) horizon-24 E[R^4] is {independent_horizon24}, expected {}",
                Self::HORIZON24_FOURTH_MOMENT
            ));
        }
        if independent_horizon24 <= Self::GAUSSIAN_FOURTH_MOMENT || independent_horizon24 >= independent_horizon8 {
            return Err("horizon-24 Gamma tails must remain above Gaussian and below horizon 8".to_string());
        }

        Ok(ActionIndependentGammaAttenuationReport {
            one_step_second_moment: Self::ONE_STEP_SECOND_MOMENT,
            one_step_fourth_moment: independent_horizon1,
            horizon8_fourth_moment: independent_horizon8,
            horizon24_fourth_moment: independent_horizon24,
            process_covariance: Self::process_covariance(),
        })
    }
}

/// Gaussian control for [`ActionIndependentGammaAttenuationCanary`].
///
/// This is deliberately a named fixture rather than an implicit `W=1`
/// default. It keeps `A`, `Q`, horizon, trajectory length, and the absence of
/// actions fixed while changing only the radial innovation law. The paired
/// Fern ablation can therefore distinguish a generic extra-parameter gain
/// from a gain specific to non-Gaussian magnitude.
pub struct ActionIndependentGaussianAttenuationCanary;

/// Exact moment contract for [`ActionIndependentGaussianAttenuationCanary`].
#[derive(Clone, Debug, PartialEq)]
pub struct ActionIndependentGaussianAttenuationReport {
    pub one_step_second_moment: f64,
    pub one_step_fourth_moment: f64,
    pub horizon8_fourth_moment: f64,
    pub process_covariance: DMatrix<f64>,
}

impl ActionIndependentGaussianAttenuationCanary {
    pub const SOURCE_NAME: &'static str = "radial_attenuation_gaussian";
    pub const DRIFT: f64 = ActionIndependentGammaAttenuationCanary::DRIFT;
    pub const SCALE: f64 = ActionIndependentGammaAttenuationCanary::SCALE;
    pub const HORIZON: usize = ActionIndependentGammaAttenuationCanary::HORIZON;
    pub const ONE_STEP_SECOND_MOMENT: f64 = 2.0;
    pub const ONE_STEP_FOURTH_MOMENT: f64 = 8.0;
    pub const HORIZON8_FOURTH_MOMENT: f64 = 8.0;

    pub fn process_covariance() -> DMatrix<f64> {
        ActionIndependentGammaAttenuationCanary::process_covariance()
    }

    pub fn evaluate() -> Result<ActionIndependentGaussianAttenuationReport, String> {
        let (_, horizon8, horizon1) = level0c_radius_moments(None);
        if mixed_residual(horizon1, Self::ONE_STEP_FOURTH_MOMENT) > 0.0 {
            return Err(format!(
                "Gaussian one-step E[R^4] is {horizon1}, expected {}",
                Self::ONE_STEP_FOURTH_MOMENT
            ));
        }
        if mixed_residual(horizon8, Self::HORIZON8_FOURTH_MOMENT) > 0.0 {
            return Err(format!(
                "Gaussian horizon-8 E[R^4] is {horizon8}, expected {}",
                Self::HORIZON8_FOURTH_MOMENT
            ));
        }
        Ok(ActionIndependentGaussianAttenuationReport {
            one_step_second_moment: Self::ONE_STEP_SECOND_MOMENT,
            one_step_fourth_moment: horizon1,
            horizon8_fourth_moment: horizon8,
            process_covariance: Self::process_covariance(),
        })
    }
}

fn impulse_action_sequences(horizon: usize, action_dimension: usize, amplitude: f64) -> Vec<DMatrix<f64>> {
    let mut sequences = vec![DMatrix::zeros(horizon, action_dimension)];
    for timestep in 0..horizon {
        for action in 0..action_dimension {
            for sign in [amplitude, -amplitude] {
                let mut sequence = DMatrix::zeros(horizon, action_dimension);
                sequence[(timestep, action)] = sign;
                sequences.push(sequence);
            }
        }
    }
    sequences
}

fn sequence_design_rank(bank: &ActionSequenceBank, rank_tolerance: f64) -> Result<usize, String> {
    let width = bank.horizon() * bank.action_dimension();
    let mut design = DMatrix::zeros(bank.sequence_count(), width);
    for (row, sequence) in bank.sequences().iter().enumerate() {
        for timestep in 0..bank.horizon() {
            for action in 0..bank.action_dimension() {
                design[(row, timestep * bank.action_dimension() + action)] = sequence[(timestep, action)];
            }
        }
    }
    let singular_values = design.svd(false, false).singular_values;
    Ok(singular_values.iter().filter(|value| **value > rank_tolerance).count())
}

fn mixed_residual(actual: f64, truth: f64) -> f64 {
    let allowed = Level0ACanary::MIXED_ABS + Level0ACanary::MIXED_REL * truth.abs();
    ((actual - truth).abs() - allowed).max(0.0)
}

fn fixture_closed_form_residual(
    world: &ControlledLinearGaussian,
    bank: &ActionSequenceBank,
    initial_state: &DVector<f64>,
) -> Result<f64, String> {
    let mut residual = 0.0_f64;
    for sequence in bank.sequences() {
        let oracle = world.direct_horizon(initial_state, sequence)?;
        residual = residual.max(closed_form_mixed_residual(world, initial_state, sequence, &oracle));
    }
    Ok(residual)
}

fn detect_level0a_defects(
    world: &ControlledLinearGaussian,
    bank: &ActionSequenceBank,
) -> Result<Vec<&'static str>, String> {
    let initial_state = Level0ACanary::initial_state();
    let mut detected = Vec::new();

    let mut wrong_b = Level0ACanary::action_response();
    wrong_b[(0, 1)] = 0.0;
    let wrong_b_world = ControlledLinearGaussian::try_new(
        Level0ACanary::state_transition(),
        wrong_b,
        Level0ACanary::process_covariance(),
    )?;
    if mutant_misses_pinned_fixture(&wrong_b_world, bank, world, &initial_state)? {
        detected.push("wrong_B01");
    }

    let mut diagonal_q = Level0ACanary::process_covariance();
    diagonal_q[(0, 1)] = 0.0;
    diagonal_q[(1, 0)] = 0.0;
    let diagonal_q_world = ControlledLinearGaussian::try_new(
        Level0ACanary::state_transition(),
        Level0ACanary::action_response(),
        diagonal_q,
    )?;
    if mutant_misses_pinned_fixture(&diagonal_q_world, bank, world, &initial_state)? {
        detected.push("diagonal_Q");
    }

    let mut shifted = bank.sequences().to_vec();
    let Some(first_impulse) = shifted
        .iter_mut()
        .find(|sequence| sequence[(0, 0)] == Level0ACanary::IMPULSE_AMPLITUDE)
    else {
        return Err("expected a +0.5 impulse at timestep 0, action 0".to_string());
    };
    first_impulse[(0, 0)] = 0.0;
    first_impulse[(1, 0)] = Level0ACanary::IMPULSE_AMPLITUDE;
    let shifted_bank = ActionSequenceBank::try_new(shifted)?;
    if mutant_bank_misses_pinned_fixture(world, &shifted_bank, bank, &initial_state)? {
        detected.push("shifted_impulse");
    }

    let dropped: Vec<DMatrix<f64>> = bank
        .sequences()
        .iter()
        .map(|sequence| {
            let mut dropped_sequence = sequence.clone();
            for timestep in 0..sequence.nrows() {
                dropped_sequence[(timestep, 1)] = 0.0;
            }
            dropped_sequence
        })
        .collect();
    let dropped_bank = ActionSequenceBank::try_new(dropped)?;
    let dropped_rank = dropped_bank.coverage(Level0ACanary::RANK_TOLERANCE)?.rank;
    if dropped_rank < 2 || mutant_bank_misses_pinned_fixture(world, &dropped_bank, bank, &initial_state)? {
        detected.push("dropped_action_axis");
    }

    let split = geometry_gaps(&Level0ACanary::process_covariance(), &[vec![0], vec![1]])?;
    if split.block_seam_nats > 1e-8 {
        detected.push("split_block_seam");
    }

    Ok(detected)
}

struct GeometryGaps {
    block_seam_nats: f64,
    orientation_capacity_nats: f64,
}

fn logdet_spd(matrix: &DMatrix<f64>) -> Result<f64, String> {
    let eigenvalues = SymmetricEigen::new(matrix.clone()).eigenvalues;
    if eigenvalues.iter().any(|value| *value <= 0.0) {
        return Err("covariance must be positive definite for a Gaussian log-score seam".to_string());
    }
    Ok(eigenvalues.iter().map(|value| value.ln()).sum())
}

fn block_extracted_covariance(matrix: &DMatrix<f64>, groups: &[Vec<usize>]) -> DMatrix<f64> {
    let dimension = matrix.nrows();
    let mut blocked = DMatrix::zeros(dimension, dimension);
    for group in groups {
        for &row in group {
            for &column in group {
                blocked[(row, column)] = matrix[(row, column)];
            }
        }
    }
    blocked
}

fn exact_householder_reflections(target: &DMatrix<f64>) -> Vec<DVector<f64>> {
    let width = target.nrows();
    let mut working = SymmetricEigen::new(target.clone()).eigenvectors;
    let mut vectors = Vec::new();
    for column in 0..width.saturating_sub(1) {
        let tail_len = width - column;
        let mut tail = DVector::zeros(tail_len);
        for index in 0..tail_len {
            tail[index] = working[(column + index, column)];
        }
        let norm = tail.norm();
        if norm <= 1e-15 {
            continue;
        }
        let alpha = if tail[0] != 0.0 {
            -tail[0].signum() * norm
        } else {
            -norm
        };
        tail[0] -= alpha;
        if tail.norm() <= 1e-13 {
            continue;
        }
        tail /= tail.norm();
        let mut padded = DVector::zeros(width);
        for index in 0..tail_len {
            padded[column + index] = tail[index];
        }
        let projection = padded.transpose() * &working;
        working -= 2.0 * &padded * projection;
        vectors.push(padded);
    }
    vectors.reverse();
    vectors
}

fn householder_product(vectors: &[DVector<f64>], width: usize) -> DMatrix<f64> {
    let mut rotation = DMatrix::identity(width, width);
    for vector in vectors {
        let unit = vector / vector.norm();
        let reflector = DMatrix::identity(width, width) - 2.0 * &unit * unit.transpose();
        rotation = reflector * rotation;
    }
    rotation
}

fn orientation_capacity_nats(target: &DMatrix<f64>) -> Result<f64, String> {
    let width = target.nrows();
    let rotation = householder_product(&exact_householder_reflections(target), width);
    let rotated = rotation.transpose() * target * &rotation;
    let mut log_diagonal = 0.0_f64;
    for index in 0..width {
        let diagonal = rotated[(index, index)];
        if diagonal <= 0.0 {
            return Err("rotated covariance diagonal is not positive".to_string());
        }
        log_diagonal += diagonal.ln();
    }
    Ok(0.5 * (log_diagonal - logdet_spd(target)?))
}

fn geometry_gaps(target: &DMatrix<f64>, groups: &[Vec<usize>]) -> Result<GeometryGaps, String> {
    let blocked = block_extracted_covariance(target, groups);
    Ok(GeometryGaps {
        block_seam_nats: 0.5 * (logdet_spd(&blocked)? - logdet_spd(target)?),
        orientation_capacity_nats: orientation_capacity_nats(&blocked)?,
    })
}

fn mutant_misses_pinned_fixture(
    mutant_world: &ControlledLinearGaussian,
    bank: &ActionSequenceBank,
    pinned_world: &ControlledLinearGaussian,
    initial_state: &DVector<f64>,
) -> Result<bool, String> {
    for sequence in bank.sequences() {
        let mutant = mutant_world.direct_horizon(initial_state, sequence)?;
        if closed_form_mixed_residual(pinned_world, initial_state, sequence, &mutant) > 0.0 {
            return Ok(true);
        }
    }
    Ok(false)
}

fn mutant_bank_misses_pinned_fixture(
    world: &ControlledLinearGaussian,
    mutant_bank: &ActionSequenceBank,
    pinned_bank: &ActionSequenceBank,
    initial_state: &DVector<f64>,
) -> Result<bool, String> {
    for (mutant_sequence, pinned_sequence) in mutant_bank.sequences().iter().zip(pinned_bank.sequences()) {
        let mutant = world.direct_horizon(initial_state, mutant_sequence)?;
        if closed_form_mixed_residual(world, initial_state, pinned_sequence, &mutant) > 0.0 {
            return Ok(true);
        }
    }
    Ok(false)
}

fn matrix_power(matrix: &DMatrix<f64>, exponent: usize) -> DMatrix<f64> {
    let mut result = DMatrix::identity(matrix.nrows(), matrix.ncols());
    for _ in 0..exponent {
        result = matrix * result;
    }
    result
}

fn closed_form_mixed_residual(
    world: &ControlledLinearGaussian,
    initial_state: &DVector<f64>,
    actions: &DMatrix<f64>,
    oracle: &DirectHorizonOracle,
) -> f64 {
    let horizon = actions.nrows();
    let mut residual = 0.0_f64;
    for future_timestep in 0..horizon {
        let mut mean = matrix_power(&world.state_transition, future_timestep + 1) * initial_state;
        let mut covariance = DMatrix::zeros(world.state_dimension(), world.state_dimension());
        let initial_state_jacobian = matrix_power(&world.state_transition, future_timestep + 1);
        for action_timestep in 0..=future_timestep {
            let transition = matrix_power(&world.state_transition, future_timestep - action_timestep);
            mean += &transition * &world.action_response * actions.row(action_timestep).transpose();
            let noise_transition = matrix_power(&world.state_transition, action_timestep);
            covariance += &noise_transition * &world.process_covariance * noise_transition.transpose();
            let jacobian = &transition * &world.action_response;
            for state_coordinate in 0..world.state_dimension() {
                for action_coordinate in 0..world.action_dimension() {
                    residual = residual.max(mixed_residual(
                        oracle.action_jacobian[(future_timestep, state_coordinate, action_timestep, action_coordinate)],
                        jacobian[(state_coordinate, action_coordinate)],
                    ));
                }
            }
        }
        for state_coordinate in 0..world.state_dimension() {
            residual = residual.max(mixed_residual(
                oracle.means[future_timestep][state_coordinate],
                mean[state_coordinate],
            ));
            for other in 0..world.state_dimension() {
                residual = residual.max(mixed_residual(
                    oracle.covariances[future_timestep][(state_coordinate, other)],
                    covariance[(state_coordinate, other)],
                ));
                residual = residual.max(mixed_residual(
                    oracle.initial_state_jacobians[future_timestep][(state_coordinate, other)],
                    initial_state_jacobian[(state_coordinate, other)],
                ));
            }
        }
    }
    residual
}

fn level0b_covariance(action1: f64, action2: f64) -> DMatrix<f64> {
    let scale = 1.0 + Level0BCanary::SCALE_GAIN * action1;
    let theta = Level0BCanary::ANGLE_GAIN * action2;
    let cosine = theta.cos();
    let sine = theta.sin();
    let rotation = DMatrix::from_row_slice(2, 2, &[cosine, -sine, sine, cosine]);
    let base = Level0ACanary::process_covariance();
    (&rotation * base * rotation.transpose()) * (scale * scale)
}

fn level0b_sequence(actions: &DMatrix<f64>, swapped: bool) -> Result<ControlledLinearSequence, String> {
    let horizon = actions.nrows();
    let mut transitions = Vec::with_capacity(horizon);
    let mut responses = Vec::with_capacity(horizon);
    let mut covariances = Vec::with_capacity(horizon);
    for timestep in 0..horizon {
        let action1 = actions[(timestep, 0)];
        let action2 = actions[(timestep, 1)];
        let covariance = if swapped {
            level0b_covariance(action2, action1)
        } else {
            level0b_covariance(action1, action2)
        };
        transitions.push(Level0ACanary::state_transition());
        responses.push(Level0ACanary::action_response());
        covariances.push(covariance);
    }
    ControlledLinearSequence::try_new(transitions, responses, covariances)
}

fn level0b_closed_form_residual(
    bank: &ActionSequenceBank,
    initial_state: &DVector<f64>,
    swapped: bool,
) -> Result<f64, String> {
    let mut residual = 0.0_f64;
    let transition = Level0ACanary::state_transition();
    let response = Level0ACanary::action_response();
    for sequence in bank.sequences() {
        let world = level0b_sequence(sequence, swapped)?;
        let oracle = world.direct_horizon(initial_state, sequence)?;
        for future_timestep in 0..sequence.nrows() {
            let mut mean = matrix_power(&transition, future_timestep + 1) * initial_state;
            let mut covariance = DMatrix::zeros(2, 2);
            let initial_state_jacobian = matrix_power(&transition, future_timestep + 1);
            for action_timestep in 0..=future_timestep {
                let power = matrix_power(&transition, future_timestep - action_timestep);
                mean += &power * &response * sequence.row(action_timestep).transpose();
                let action1 = sequence[(action_timestep, 0)];
                let action2 = sequence[(action_timestep, 1)];
                let process = if swapped {
                    level0b_covariance(action2, action1)
                } else {
                    level0b_covariance(action1, action2)
                };
                covariance += &power * process * power.transpose();
                let jacobian = &power * &response;
                for state_coordinate in 0..2 {
                    for action_coordinate in 0..2 {
                        residual = residual.max(mixed_residual(
                            oracle.action_jacobian
                                [(future_timestep, state_coordinate, action_timestep, action_coordinate)],
                            jacobian[(state_coordinate, action_coordinate)],
                        ));
                    }
                }
            }
            for state_coordinate in 0..2 {
                residual = residual.max(mixed_residual(
                    oracle.means[future_timestep][state_coordinate],
                    mean[state_coordinate],
                ));
                for other in 0..2 {
                    residual = residual.max(mixed_residual(
                        oracle.covariances[future_timestep][(state_coordinate, other)],
                        covariance[(state_coordinate, other)],
                    ));
                    residual = residual.max(mixed_residual(
                        oracle.initial_state_jacobians[future_timestep][(state_coordinate, other)],
                        initial_state_jacobian[(state_coordinate, other)],
                    ));
                }
            }
        }
    }
    Ok(residual)
}

fn level0b_mean_matches_level0a(bank: &ActionSequenceBank, initial_state: &DVector<f64>) -> Result<bool, String> {
    let level0a = Level0ACanary::world()?;
    for sequence in bank.sequences() {
        let gaussian = level0a.direct_horizon(initial_state, sequence)?;
        let action_conditioned = level0b_sequence(sequence, false)?.direct_horizon(initial_state, sequence)?;
        for timestep in 0..sequence.nrows() {
            if mixed_residual(
                action_conditioned.means[timestep].metric_distance(&gaussian.means[timestep]),
                0.0,
            ) > 0.0
            {
                return Ok(false);
            }
        }
    }
    Ok(true)
}

fn pinned_level0b_oracle_residual(
    oracle: &DirectHorizonOracle,
    sequence: &DMatrix<f64>,
    initial_state: &DVector<f64>,
) -> f64 {
    let transition = Level0ACanary::state_transition();
    let response = Level0ACanary::action_response();
    let mut residual = 0.0_f64;
    for future_timestep in 0..sequence.nrows() {
        let mut mean = matrix_power(&transition, future_timestep + 1) * initial_state;
        let mut covariance = DMatrix::zeros(2, 2);
        for action_timestep in 0..=future_timestep {
            let power = matrix_power(&transition, future_timestep - action_timestep);
            mean += &power * &response * sequence.row(action_timestep).transpose();
            let process = level0b_covariance(sequence[(action_timestep, 0)], sequence[(action_timestep, 1)]);
            covariance += &power * process * power.transpose();
        }
        for state_coordinate in 0..2 {
            residual = residual.max(mixed_residual(
                oracle.means[future_timestep][state_coordinate],
                mean[state_coordinate],
            ));
            for other in 0..2 {
                residual = residual.max(mixed_residual(
                    oracle.covariances[future_timestep][(state_coordinate, other)],
                    covariance[(state_coordinate, other)],
                ));
            }
        }
    }
    residual
}

fn detect_level0b_defects(
    bank: &ActionSequenceBank,
    initial_state: &DVector<f64>,
) -> Result<Vec<&'static str>, String> {
    let mut detected = Vec::new();
    let constant = Level0ACanary::world()?;
    let mut constant_residual = 0.0_f64;
    let mut swapped_residual = 0.0_f64;
    for sequence in bank.sequences() {
        constant_residual = constant_residual.max(pinned_level0b_oracle_residual(
            &constant.direct_horizon(initial_state, sequence)?,
            sequence,
            initial_state,
        ));
        swapped_residual = swapped_residual.max(pinned_level0b_oracle_residual(
            &level0b_sequence(sequence, true)?.direct_horizon(initial_state, sequence)?,
            sequence,
            initial_state,
        ));
    }
    if constant_residual > 0.0 {
        detected.push("constant_Q0");
    }
    if swapped_residual > 0.0 {
        detected.push("swapped_scale_rotation");
    }
    let mut wrong_b = Level0ACanary::action_response();
    wrong_b[(0, 1)] = 0.0;
    let mut mean_gap = 0.0_f64;
    for sequence in bank.sequences() {
        let truth = level0b_sequence(sequence, false)?.direct_horizon(initial_state, sequence)?;
        let mutant = ControlledLinearSequence::try_new(
            vec![Level0ACanary::state_transition(); sequence.nrows()],
            vec![wrong_b.clone(); sequence.nrows()],
            (0..sequence.nrows())
                .map(|timestep| level0b_covariance(sequence[(timestep, 0)], sequence[(timestep, 1)]))
                .collect(),
        )?
        .direct_horizon(initial_state, sequence)?;
        mean_gap = mean_gap.max((&truth.means[sequence.nrows() - 1] - &mutant.means[sequence.nrows() - 1]).norm());
    }
    if mean_gap > 0.0 {
        detected.push("wrong_B01");
    }
    let mut lagged_residual = 0.0_f64;
    for sequence in bank.sequences() {
        let mut lagged = sequence.clone();
        for timestep in (1..sequence.nrows()).rev() {
            lagged[(timestep, 0)] = sequence[(timestep - 1, 0)];
            lagged[(timestep, 1)] = sequence[(timestep - 1, 1)];
        }
        lagged[(0, 0)] = 0.0;
        lagged[(0, 1)] = 0.0;
        lagged_residual = lagged_residual.max(pinned_level0b_oracle_residual(
            &level0b_sequence(&lagged, false)?.direct_horizon(initial_state, sequence)?,
            sequence,
            initial_state,
        ));
    }
    if lagged_residual > 0.0 {
        detected.push("covariance_lag");
    }
    Ok(detected)
}

fn level0c_alpha(sequence: &DMatrix<f64>) -> Option<f64> {
    if sequence.iter().all(|value| *value == 0.0) {
        None
    } else if sequence.column(1).iter().all(|value| *value == 0.0) {
        Some(8.0)
    } else {
        Some(2.0)
    }
}

fn level0c_increment_variances() -> [f64; 8] {
    let mut values = [0.0; 8];
    let mut power = 1.0_f64;
    for value in &mut values {
        *value = Level0CCanary::SCALE * Level0CCanary::SCALE * power;
        power *= Level0CCanary::DRIFT * Level0CCanary::DRIFT;
    }
    values
}

fn level0c_radius_moments(alpha: Option<f64>) -> (f64, f64, f64) {
    let increments = level0c_increment_variances();
    let sum: f64 = increments.iter().sum();
    let sum_squares: f64 = increments.iter().map(|value| value * value).sum();
    let first = increments[0];
    match alpha {
        None => (8.0, 8.0, 8.0),
        Some(shape) => {
            let shared = 8.0 * (1.0 + 1.0 / shape);
            let independent = 8.0 + 8.0 / shape * sum_squares / (sum * sum);
            let independent_h1 = 8.0 + 8.0 / shape * (first * first) / (first * first);
            (shared, independent, independent_h1)
        }
    }
}

fn independent_radius_fourth_moment(shape: f64, horizon: usize) -> Result<f64, String> {
    if !shape.is_finite() || shape <= 0.0 {
        return Err("radial Gamma shape must be finite and positive".to_string());
    }
    if horizon == 0 {
        return Err("radial forecast horizon must be positive".to_string());
    }
    let mut variance = Level0CCanary::SCALE * Level0CCanary::SCALE;
    let mut sum = 0.0;
    let mut sum_squares = 0.0;
    for _ in 0..horizon {
        sum += variance;
        sum_squares += variance * variance;
        variance *= Level0CCanary::DRIFT * Level0CCanary::DRIFT;
    }
    Ok(8.0 + 8.0 / shape * sum_squares / (sum * sum))
}

fn level0c_cf_value(alpha: Option<f64>, radius: f64, shared: bool) -> f64 {
    let increments = level0c_increment_variances();
    let sum: f64 = increments.iter().sum();
    let radius_sq = radius * radius;
    match alpha {
        None => (-0.5 * sum * radius_sq).exp(),
        Some(shape) if shared => (1.0 + sum * radius_sq / (2.0 * shape)).powf(-shape),
        Some(shape) => increments.iter().fold(1.0_f64, |product, variance| {
            product * (1.0 + variance * radius_sq / (2.0 * shape)).powf(-shape)
        }),
    }
}

fn level0c_cf_residual(alpha: Option<f64>) -> f64 {
    let increments = level0c_increment_variances();
    let sum: f64 = increments.iter().sum();
    let mut residual = 0.0_f64;
    for radius in Level0CCanary::CF_RADII {
        let radius_sq = radius * radius;
        let shared = level0c_cf_value(alpha, radius, true);
        let independent = level0c_cf_value(alpha, radius, false);
        match alpha {
            None => {
                let gaussian = (-0.5 * sum * radius_sq).exp();
                residual = residual.max(mixed_residual(shared, gaussian));
                residual = residual.max(mixed_residual(independent, gaussian));
            }
            Some(shape) => {
                let first = increments[0];
                let one_step = (1.0 + first * radius_sq / (2.0 * shape)).powf(-shape);
                residual = residual.max(mixed_residual(
                    (1.0 + first * radius_sq / (2.0 * shape)).powf(-shape),
                    one_step,
                ));
                if radius == 2.0 && mixed_residual(independent, shared) == 0.0 {
                    residual = residual.max(1.0);
                }
            }
        }
    }
    residual
}

fn level0c_sign_pairs_match(bank: &ActionSequenceBank) -> Result<bool, String> {
    let sequences = bank.sequences();
    if sequences.len() != 5 {
        return Err(format!("Level 0C bank must have 5 sequences, got {}", sequences.len()));
    }
    for (left, right) in [(1, 2), (3, 4)] {
        if level0c_alpha(&sequences[left]) != level0c_alpha(&sequences[right]) {
            return Ok(false);
        }
        let (left_shared, left_independent, _) = level0c_radius_moments(level0c_alpha(&sequences[left]));
        let (right_shared, right_independent, _) = level0c_radius_moments(level0c_alpha(&sequences[right]));
        if mixed_residual(left_shared, right_shared) > 0.0 || mixed_residual(left_independent, right_independent) > 0.0
        {
            return Ok(false);
        }
        for radius in Level0CCanary::CF_RADII {
            if mixed_residual(
                level0c_cf_value(level0c_alpha(&sequences[left]), radius, true),
                level0c_cf_value(level0c_alpha(&sequences[right]), radius, true),
            ) > 0.0
            {
                return Ok(false);
            }
        }
    }
    Ok(true)
}

fn sorted_eigenvalues(matrix: &DMatrix<f64>) -> [f64; 2] {
    let mut values: Vec<f64> = SymmetricEigen::new(matrix.clone())
        .eigenvalues
        .iter()
        .copied()
        .collect();
    values.sort_by(|left, right| left.partial_cmp(right).unwrap());
    [values[0], values[1]]
}

fn level0b_scale_preserves_orientation() -> Result<bool, String> {
    let base = Level0ACanary::process_covariance();
    let scaled = level0b_covariance(0.5, 0.0);
    let factor = (1.0 + Level0BCanary::SCALE_GAIN * 0.5).powi(2);
    for row in 0..2 {
        for column in 0..2 {
            if mixed_residual(scaled[(row, column)], factor * base[(row, column)]) > 0.0 {
                return Ok(false);
            }
        }
    }
    Ok(true)
}

fn level0b_rotation_preserves_spectrum() -> Result<bool, String> {
    let base = sorted_eigenvalues(&Level0ACanary::process_covariance());
    let rotated = sorted_eigenvalues(&level0b_covariance(0.0, 0.5));
    Ok(mixed_residual(rotated[0], base[0]) == 0.0 && mixed_residual(rotated[1], base[1]) == 0.0)
}

fn detect_level0c_defects(bank: &ActionSequenceBank) -> Result<Vec<&'static str>, String> {
    let mut detected = Vec::new();
    let (shared_alpha2, independent_alpha2, _) = level0c_radius_moments(Some(2.0));
    if mixed_residual(8.0, shared_alpha2) > 0.0 {
        detected.push("gaussian_W");
    }
    let (shared_alpha8, _, _) = level0c_radius_moments(Some(8.0));
    if mixed_residual(shared_alpha8, shared_alpha2) > 0.0 {
        detected.push("swapped_alpha");
    }
    // Gamma(shape=2, rate=8) has E[W]=1/4, so E[R^2] becomes 0.5 instead of 2.
    if mixed_residual(Level0CCanary::SECOND_MOMENT * (2.0 / 8.0), Level0CCanary::SECOND_MOMENT) > 0.0 {
        detected.push("rate_not_shape");
    }
    if mixed_residual(independent_alpha2, shared_alpha2) > 0.0 {
        detected.push("independent_labeled_shared");
    }
    let _ = bank;
    Ok(detected)
}

fn matrix_asymmetry(matrix: &DMatrix<f64>) -> f64 {
    let mut maximum = 0.0_f64;
    for row in 0..matrix.nrows() {
        for column in 0..row {
            maximum = maximum.max((matrix[(row, column)] - matrix[(column, row)]).abs());
        }
    }
    maximum
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scalar_world() -> ControlledLinearGaussian {
        ControlledLinearGaussian::try_new(
            DMatrix::from_row_slice(1, 1, &[2.0]),
            DMatrix::from_row_slice(1, 1, &[3.0]),
            DMatrix::from_row_slice(1, 1, &[5.0]),
        )
        .unwrap()
    }

    #[test]
    fn direct_horizon_matches_the_closed_form_controlled_recurrence() {
        let oracle = scalar_world()
            .direct_horizon(
                &DVector::from_vec(vec![7.0]),
                &DMatrix::from_row_slice(2, 1, &[1.0, 4.0]),
            )
            .unwrap();

        assert_eq!(
            oracle.means,
            vec![DVector::from_vec(vec![17.0]), DVector::from_vec(vec![46.0])]
        );
        assert_eq!(
            oracle.covariances,
            vec![
                DMatrix::from_row_slice(1, 1, &[5.0]),
                DMatrix::from_row_slice(1, 1, &[25.0])
            ]
        );
        assert_eq!(
            oracle.initial_state_jacobians,
            vec![
                DMatrix::from_row_slice(1, 1, &[2.0]),
                DMatrix::from_row_slice(1, 1, &[4.0])
            ]
        );
        assert_eq!(oracle.action_jacobian.shape(), &[2, 1, 2, 1]);
        assert_eq!(oracle.action_jacobian[(0, 0, 0, 0)], 3.0);
        assert_eq!(oracle.action_jacobian[(0, 0, 1, 0)], 0.0);
        assert_eq!(oracle.action_jacobian[(1, 0, 0, 0)], 6.0);
        assert_eq!(oracle.action_jacobian[(1, 0, 1, 0)], 3.0);
        let path = oracle.joint_path_covariance().unwrap();
        assert_eq!(path.nrows(), 2);
        assert_eq!(path[(0, 0)], 5.0);
        assert_eq!(path[(1, 1)], 25.0);
        assert_eq!(path[(1, 0)], 10.0);
        assert_eq!(path[(0, 1)], 10.0);
    }

    #[test]
    fn gramian_and_inverse_action_close_the_endpoint_response_loop() {
        let world = ControlledLinearGaussian::try_new(
            DMatrix::from_row_slice(2, 2, &[0.9, 2.0, 0.0, 0.8]),
            DMatrix::from_row_slice(2, 1, &[0.0, 1.0]),
            DMatrix::zeros(2, 2),
        )
        .unwrap();
        let initial_state = DVector::from_vec(vec![0.3, -0.4]);
        let actions = DMatrix::zeros(2, 1);
        let nominal = world.direct_horizon(&initial_state, &actions).unwrap();

        assert_eq!(
            nominal.controllability_gramian(0).unwrap(),
            DMatrix::from_row_slice(2, 2, &[0.0, 0.0, 0.0, 1.0])
        );
        let expected_gramian = DMatrix::from_row_slice(2, 2, &[4.0, 1.6, 1.6, 1.64]);
        assert!((nominal.controllability_gramian(1).unwrap() - expected_gramian).norm() < 3e-16);

        let desired_displacement = DVector::from_vec(vec![0.5, -0.3]);
        let intervention = nominal
            .ridge_endpoint_action_intervention(1, &desired_displacement, 1e-10)
            .unwrap();
        let executed = world
            .direct_horizon(&initial_state, &(actions + &intervention))
            .unwrap();
        let executed_displacement = &executed.means[1] - &nominal.means[1];
        let endpoint_action_response = DMatrix::from_row_slice(2, 2, &[2.0, 0.0, 0.8, 1.0]);
        let predicted_displacement = endpoint_action_response * DVector::from_column_slice(intervention.as_slice());
        assert!((&executed_displacement - predicted_displacement).norm() < 3e-16);
        assert!((executed_displacement - desired_displacement).norm() < 2e-10);
    }

    #[test]
    fn unreachable_endpoint_direction_stays_unreachable() {
        let world = ControlledLinearGaussian::try_new(
            DMatrix::from_diagonal(&DVector::from_vec(vec![0.8, 0.7])),
            DMatrix::from_row_slice(2, 1, &[1.0, 0.0]),
            DMatrix::zeros(2, 2),
        )
        .unwrap();
        let oracle = world.direct_horizon(&DVector::zeros(2), &DMatrix::zeros(3, 1)).unwrap();
        let gramian = oracle.controllability_gramian(2).unwrap();
        let intervention = oracle
            .ridge_endpoint_action_intervention(2, &DVector::from_vec(vec![0.0, 1.0]), 1e-6)
            .unwrap();

        assert_eq!(gramian[(1, 0)], 0.0);
        assert_eq!(gramian[(1, 1)], 0.0);
        assert_eq!(intervention, DMatrix::zeros(3, 1));
    }

    #[test]
    fn multi_action_intervention_stays_timestep_major() {
        let world =
            ControlledLinearGaussian::try_new(DMatrix::identity(2, 2), DMatrix::identity(2, 2), DMatrix::zeros(2, 2))
                .unwrap();
        let oracle = world.direct_horizon(&DVector::zeros(2), &DMatrix::zeros(2, 2)).unwrap();

        let intervention = oracle
            .ridge_endpoint_action_intervention(1, &DVector::from_vec(vec![0.6, -0.3]), 1.0)
            .unwrap();

        let expected = DMatrix::from_row_slice(2, 2, &[0.2, -0.1, 0.2, -0.1]);
        assert!((intervention - expected).norm() < 2e-16);
    }

    #[test]
    fn endpoint_control_contracts_refuse_invalid_requests() {
        let oracle = scalar_world()
            .direct_horizon(&DVector::from_vec(vec![0.0]), &DMatrix::zeros(1, 1))
            .unwrap();

        assert!(
            oracle
                .controllability_gramian(1)
                .unwrap_err()
                .contains("outside horizon")
        );
        assert!(
            oracle
                .ridge_endpoint_action_intervention(0, &DVector::from_vec(vec![1.0]), 0.0)
                .unwrap_err()
                .contains("ridge must be finite and > 0")
        );
        assert!(
            oracle
                .ridge_endpoint_action_intervention(0, &DVector::from_vec(vec![1.0, 2.0]), 1e-6)
                .unwrap_err()
                .contains("desired_displacement")
        );
    }

    #[test]
    fn time_varying_sequence_uses_each_local_operator_once() {
        let world = ControlledLinearSequence::try_new(
            vec![
                DMatrix::from_row_slice(1, 1, &[2.0]),
                DMatrix::from_row_slice(1, 1, &[0.5]),
                DMatrix::from_row_slice(1, 1, &[-1.0]),
            ],
            vec![
                DMatrix::from_row_slice(1, 1, &[1.0]),
                DMatrix::from_row_slice(1, 1, &[3.0]),
                DMatrix::from_row_slice(1, 1, &[2.0]),
            ],
            vec![
                DMatrix::from_row_slice(1, 1, &[1.0]),
                DMatrix::from_row_slice(1, 1, &[2.0]),
                DMatrix::from_row_slice(1, 1, &[4.0]),
            ],
        )
        .unwrap();
        let oracle = world
            .direct_horizon(
                &DVector::from_vec(vec![4.0]),
                &DMatrix::from_row_slice(3, 1, &[1.0, 2.0, -1.0]),
            )
            .unwrap();

        assert_eq!(
            oracle.means,
            vec![
                DVector::from_vec(vec![9.0]),
                DVector::from_vec(vec![10.5]),
                DVector::from_vec(vec![-12.5]),
            ]
        );
        assert_eq!(
            oracle.covariances,
            vec![
                DMatrix::from_row_slice(1, 1, &[1.0]),
                DMatrix::from_row_slice(1, 1, &[2.25]),
                DMatrix::from_row_slice(1, 1, &[6.25]),
            ]
        );
        assert_eq!(oracle.initial_state_jacobians[0][(0, 0)], 2.0);
        assert_eq!(oracle.initial_state_jacobians[1][(0, 0)], 1.0);
        assert_eq!(oracle.initial_state_jacobians[2][(0, 0)], -1.0);
        let expected_action_jacobian = [[1.0, 0.0, 0.0], [0.5, 3.0, 0.0], [-0.5, -3.0, 2.0]];
        for future_timestep in 0..3 {
            for action_timestep in 0..3 {
                assert_eq!(
                    oracle.action_jacobian[(future_timestep, 0, action_timestep, 0)],
                    expected_action_jacobian[future_timestep][action_timestep]
                );
            }
        }
    }

    #[test]
    fn action_jacobian_keeps_each_action_at_its_true_future_step() {
        let world = scalar_world();
        assert_eq!(
            world.action_jacobian(2, 0).unwrap(),
            DMatrix::from_row_slice(1, 1, &[6.0])
        );
        assert_eq!(
            world.action_jacobian(2, 1).unwrap(),
            DMatrix::from_row_slice(1, 1, &[3.0])
        );
        assert!(
            world
                .action_jacobian(2, 2)
                .unwrap_err()
                .contains("outside direct horizon")
        );
    }

    #[test]
    fn hidden_actuator_oracle_keeps_the_exact_action_delay_and_memory() {
        let world = ScalarHiddenActuatorOracle::try_new(1.0, 2.0, 0.5, 0.5).unwrap();

        assert_eq!(world.actuator_state_after(0.0, &[2.0, 0.0]).unwrap(), 0.5);
        let oracle = world.direct_horizon(0.0, 0.0, &[1.0, 0.0, 0.0]).unwrap();

        assert_eq!(oracle.observed_states, vec![0.0, 1.0, 1.5]);
        assert_eq!(oracle.actuator_states, vec![0.5, 0.25, 0.125]);
        assert_eq!(
            oracle.observed_action_jacobian,
            DMatrix::from_row_slice(3, 3, &[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.5, 1.0, 0.0])
        );
    }

    #[test]
    fn zero_actuator_retention_forgets_every_action_before_the_latest_one() {
        let world = ScalarHiddenActuatorOracle::try_new(1.0, 1.0, 0.0, 1.0).unwrap();

        let first_history = world.actuator_state_after(7.0, &[-100.0, 3.0]).unwrap();
        let second_history = world.actuator_state_after(-9.0, &[200.0, 3.0]).unwrap();
        assert_eq!(first_history, 3.0);
        assert_eq!(second_history, 3.0);
    }

    #[test]
    fn zero_action_gain_is_the_true_no_action_history_null() {
        let world = ScalarHiddenActuatorOracle::try_new(1.0, 1.0, 0.5, 0.0).unwrap();

        let first = world.direct_horizon(2.0, 1.0, &[-100.0, 7.0, 80.0]).unwrap();
        let second = world.direct_horizon(2.0, 1.0, &[300.0, -4.0, -90.0]).unwrap();

        assert_eq!(first.observed_states, second.observed_states);
        assert_eq!(first.actuator_states, second.actuator_states);
        assert_eq!(first.observed_action_jacobian, DMatrix::zeros(3, 3));
        assert_eq!(second.observed_action_jacobian, DMatrix::zeros(3, 3));
    }

    #[test]
    fn state_dependent_control_oracle_exposes_the_exact_interaction_gain() {
        let world = ScalarStateDependentControlOracle::try_new(0.5, 1.0, 2.0).unwrap();
        let oracle = world.direct_horizon(2.0, &[3.0, 4.0]).unwrap();

        assert_eq!(oracle.states, vec![16.0, 140.0]);
        assert_eq!(
            oracle.action_jacobian,
            DMatrix::from_row_slice(2, 2, &[5.0, 0.0, 42.5, 33.0])
        );
    }

    #[test]
    fn zero_interaction_gain_is_state_independent() {
        let world = ScalarStateDependentControlOracle::try_new(0.5, 3.0, 0.0).unwrap();
        let from_small_state = world.direct_horizon(-2.0, &[1.0, 1.0]).unwrap();
        let from_large_state = world.direct_horizon(50.0, &[1.0, 1.0]).unwrap();
        let expected = DMatrix::from_row_slice(2, 2, &[3.0, 0.0, 1.5, 3.0]);

        assert_eq!(from_small_state.action_jacobian, expected);
        assert_eq!(from_large_state.action_jacobian, expected);
    }

    #[test]
    fn minimal_control_oracles_refuse_non_finite_or_unstable_parameters() {
        let retention_error = ScalarHiddenActuatorOracle::try_new(1.0, 1.0, 1.0, 1.0).unwrap_err();
        assert!(retention_error.contains("0 <= retention < 1"));

        let interaction_error = ScalarStateDependentControlOracle::try_new(1.0, f64::NAN, 1.0).unwrap_err();
        assert!(interaction_error.contains("action_offset must be finite"));
    }

    #[test]
    fn action_bank_reports_missing_excitation_as_unidentified_rank() {
        let bank = ActionSequenceBank::try_new(vec![
            DMatrix::from_row_slice(2, 2, &[-1.0, 0.0, 1.0, 0.0]),
            DMatrix::from_row_slice(2, 2, &[1.0, 0.0, -1.0, 0.0]),
        ])
        .unwrap();
        let coverage = bank.coverage(1e-12).unwrap();

        assert_eq!(coverage.sample_count, 4);
        assert_eq!(coverage.rank, 1);
        assert!(!coverage.is_full_rank());
        assert_eq!(coverage.mean, DVector::from_vec(vec![0.0, 0.0]));
    }

    #[test]
    fn action_bank_reports_full_excitation_when_both_axes_vary() {
        let bank = ActionSequenceBank::try_new(vec![
            DMatrix::from_row_slice(2, 2, &[-1.0, 0.0, 1.0, 0.0]),
            DMatrix::from_row_slice(2, 2, &[0.0, -1.0, 0.0, 1.0]),
        ])
        .unwrap();
        let coverage = bank.coverage(1e-12).unwrap();

        assert_eq!(coverage.rank, 2);
        assert!(coverage.is_full_rank());
    }

    #[test]
    fn action_bank_refuses_shape_drift_and_an_unpinned_rank_definition() {
        let shape_error = ActionSequenceBank::try_new(vec![
            DMatrix::from_row_slice(2, 1, &[0.0, 1.0]),
            DMatrix::from_row_slice(1, 1, &[0.0]),
        ])
        .unwrap_err();
        assert!(shape_error.contains("expected 2x1"));

        let bank = ActionSequenceBank::try_new(vec![DMatrix::from_row_slice(2, 1, &[0.0, 1.0])]).unwrap();
        assert!(bank.coverage(0.0).unwrap_err().contains("rank_tolerance"));
    }

    #[test]
    fn rejects_a_covariance_that_cannot_define_a_gaussian_world() {
        let error = ControlledLinearGaussian::try_new(
            DMatrix::identity(2, 2),
            DMatrix::identity(2, 1),
            DMatrix::from_row_slice(2, 2, &[1.0, 2.0, 0.0, 1.0]),
        )
        .unwrap_err();
        assert!(error.contains("symmetric"));
    }

    fn level0_geometry() -> GeometryIdentity {
        GeometryIdentity::try_new(
            2,
            vec![vec![0, 1]],
            2,
            2,
            8,
            "cpu-reference",
            GeometryCapacityClass::FullCapacity,
        )
        .unwrap()
    }

    fn exact_independent_noise() -> ProcessNoiseIdentity {
        ProcessNoiseIdentity::try_new(ProcessNoiseDependence::IndependentStep, 0, true).unwrap()
    }

    fn moment_identity(reports: Vec<OracleReportKind>) -> ControlledCaseIdentity {
        ControlledCaseIdentity::try_new(
            "level0-canary",
            OracleTier::ConditionalMoment,
            ForecastScope::DirectHorizon,
            ObservationProcess::DirectlyObservedState,
            ActionProvenance::OpenLoop,
            ActionCoverageKind::UnconditionalAction,
            ActionJacobianClaim::ExcitedSubspaceOnly,
            exact_independent_noise(),
            level0_geometry(),
            reports,
            None,
        )
        .unwrap()
    }

    #[test]
    fn complete_identity_admits_a_full_rank_open_loop_bank() {
        let bank = ActionSequenceBank::try_new(vec![
            DMatrix::from_row_slice(2, 2, &[-1.0, 0.0, 1.0, 0.0]),
            DMatrix::from_row_slice(2, 2, &[0.0, -1.0, 0.0, 1.0]),
        ])
        .unwrap();
        let admitted = moment_identity(vec![
            OracleReportKind::ConditionalMean,
            OracleReportKind::ConditionalCovariance,
            OracleReportKind::ActionJacobian,
            OracleReportKind::GaussianLogScoreProjection,
        ])
        .admit(bank.coverage(1e-12).unwrap())
        .unwrap();

        assert_eq!(admitted.identity.oracle_tier().as_str(), "conditional_moment");
        assert_eq!(admitted.identity.forecast_scope().as_str(), "direct_horizon");
        assert_eq!(admitted.action_jacobian_support, ActionJacobianSupport::FullMap);
        assert_eq!(admitted.identity.geometry().capacity_class().as_str(), "full_capacity");
    }

    #[test]
    fn identity_refuses_missing_context_tier_conflicts_and_scope_confusion() {
        assert!(
            ControlledCaseIdentity::try_new(
                "   ",
                OracleTier::ConditionalMoment,
                ForecastScope::DirectHorizon,
                ObservationProcess::DirectlyObservedState,
                ActionProvenance::OpenLoop,
                ActionCoverageKind::UnconditionalAction,
                ActionJacobianClaim::ExcitedSubspaceOnly,
                exact_independent_noise(),
                level0_geometry(),
                vec![OracleReportKind::ConditionalMean],
                None,
            )
            .unwrap_err()
            .contains("physics_context identity is required")
        );
        assert!(
            ControlledCaseIdentity::try_new(
                "level0-canary",
                OracleTier::ConditionalMoment,
                ForecastScope::DirectHorizon,
                ObservationProcess::DirectlyObservedState,
                ActionProvenance::OpenLoop,
                ActionCoverageKind::UnconditionalAction,
                ActionJacobianClaim::ExcitedSubspaceOnly,
                exact_independent_noise(),
                level0_geometry(),
                vec![OracleReportKind::RadialLaw],
                None,
            )
            .unwrap_err()
            .contains("does not authorize report radial_law")
        );
        assert!(
            ControlledCaseIdentity::try_new(
                "level0-canary",
                OracleTier::FullLaw,
                ForecastScope::DirectHorizon,
                ObservationProcess::DirectlyObservedState,
                ActionProvenance::OpenLoop,
                ActionCoverageKind::UnconditionalAction,
                ActionJacobianClaim::ExcitedSubspaceOnly,
                exact_independent_noise(),
                level0_geometry(),
                vec![OracleReportKind::TrajectoryLaw],
                None,
            )
            .unwrap_err()
            .contains("scope confusion")
        );
        assert!(
            GeometryIdentity::try_new(
                2,
                vec![vec![0, 1]],
                2,
                1,
                8,
                "cpu-reference",
                GeometryCapacityClass::FullCapacity,
            )
            .unwrap_err()
            .contains("full_capacity requires R = P")
        );
        assert!(
            ProcessNoiseIdentity::try_new(ProcessNoiseDependence::IndependentStep, 0, false)
                .unwrap_err()
                .contains("process-noise provenance")
        );
        assert!(
            ControlledCaseIdentity::try_new(
                "level0-canary",
                OracleTier::CertifiedTransform,
                ForecastScope::DirectHorizon,
                ObservationProcess::DirectlyObservedState,
                ActionProvenance::OpenLoop,
                ActionCoverageKind::UnconditionalAction,
                ActionJacobianClaim::ExcitedSubspaceOnly,
                exact_independent_noise(),
                level0_geometry(),
                vec![OracleReportKind::ConditionalMean],
                None,
            )
            .unwrap_err()
            .contains("certified_transform requires a pinned inversion error bound")
        );
    }

    #[test]
    fn radial_law_refuses_a_single_process_noise_draw() {
        let one_draw = ProcessNoiseIdentity::try_new(ProcessNoiseDependence::Persistent, 1, false).unwrap();
        let error = ControlledCaseIdentity::try_new(
            "level0-canary",
            OracleTier::FullLaw,
            ForecastScope::DirectHorizon,
            ObservationProcess::DirectlyObservedState,
            ActionProvenance::OpenLoop,
            ActionCoverageKind::UnconditionalAction,
            ActionJacobianClaim::ExcitedSubspaceOnly,
            one_draw,
            level0_geometry(),
            vec![OracleReportKind::RadialLaw],
            None,
        )
        .unwrap_err();
        assert!(error.contains("replicate_count >= 2"));
    }

    #[test]
    fn missing_excitation_cannot_support_a_full_action_jacobian_claim() {
        let bank = ActionSequenceBank::try_new(vec![
            DMatrix::from_row_slice(2, 2, &[-1.0, 0.0, 1.0, 0.0]),
            DMatrix::from_row_slice(2, 2, &[1.0, 0.0, -1.0, 0.0]),
        ])
        .unwrap();
        let coverage = bank.coverage(1e-12).unwrap();
        let full_claim = ControlledCaseIdentity::try_new(
            "level0-canary",
            OracleTier::ConditionalMoment,
            ForecastScope::DirectHorizon,
            ObservationProcess::DirectlyObservedState,
            ActionProvenance::OpenLoop,
            ActionCoverageKind::UnconditionalAction,
            ActionJacobianClaim::FullMap,
            exact_independent_noise(),
            level0_geometry(),
            vec![OracleReportKind::ActionJacobian],
            None,
        )
        .unwrap();
        assert!(full_claim.admit(coverage.clone()).unwrap_err().contains("unidentified"));

        let subspace_claim = moment_identity(vec![OracleReportKind::ActionJacobian]);
        let admitted = subspace_claim.admit(coverage).unwrap();
        assert_eq!(
            admitted.action_jacobian_support,
            ActionJacobianSupport::ExcitedSubspace {
                rank: 1,
                unidentified_dimension: 1,
            }
        );
    }

    #[test]
    fn level0a_canary_is_falsifiable() {
        let report = Level0ACanary::evaluate().unwrap();
        assert_eq!(report.instantaneous_rank, 2);
        assert_eq!(report.sequence_design_rank, 16);
        assert_eq!(report.closed_form_mixed_residual, 0.0);
        assert!(report.block_seam_nats < 1e-10);
        assert!(report.orientation_capacity_nats < 1e-10);
        assert_eq!(
            report.detected_defects,
            vec![
                "wrong_B01",
                "diagonal_Q",
                "shifted_impulse",
                "dropped_action_axis",
                "split_block_seam"
            ]
        );
    }

    #[test]
    fn level0_geometry_analysis_parameters_match_the_rust_canaries() {
        const DECLARATION_JSON: &str = include_str!("../../analysis_declarations/level0_geometry_ledger.json");
        let declaration: serde_json::Value = serde_json::from_str(DECLARATION_JSON).unwrap();
        let parameters = &declaration["parameters"];
        let state_transition = Level0ACanary::state_transition();
        let process_covariance = Level0ACanary::process_covariance();
        for row in 0..2 {
            for column in 0..2 {
                assert_eq!(
                    parameters["level0a"]["state_transition"][row][column].as_f64().unwrap(),
                    state_transition[(row, column)]
                );
                assert_eq!(
                    parameters["level0a"]["process_covariance"][row][column]
                        .as_f64()
                        .unwrap(),
                    process_covariance[(row, column)]
                );
            }
        }
        assert_eq!(
            parameters["level0b"]["scale_gain"].as_f64().unwrap(),
            Level0BCanary::SCALE_GAIN
        );
        assert_eq!(
            parameters["level0b"]["angle_gain"].as_f64().unwrap(),
            Level0BCanary::ANGLE_GAIN
        );
        assert_eq!(
            parameters["level0a"]["endpoint_horizons"],
            serde_json::json!([1, 2, 4, Level0ACanary::HORIZON])
        );
        assert_eq!(
            parameters["level0b"]["one_step_actions"],
            serde_json::json!({
                "zero": [0.0, 0.0],
                "scale_plus": [Level0ACanary::IMPULSE_AMPLITUDE, 0.0],
                "scale_minus": [-Level0ACanary::IMPULSE_AMPLITUDE, 0.0],
                "rotate_plus": [0.0, Level0ACanary::IMPULSE_AMPLITUDE],
                "rotate_minus": [0.0, -Level0ACanary::IMPULSE_AMPLITUDE]
            })
        );
    }

    #[test]
    fn level0b_path_covariance_declaration_matches_the_rust_canaries() {
        const DECLARATION_JSON: &str = include_str!("../../analysis_declarations/level0b_path_covariance.json");
        let declaration: serde_json::Value = serde_json::from_str(DECLARATION_JSON).unwrap();
        let parameters = &declaration["parameters"];
        assert_eq!(parameters["scope"], "joint_future_path_covariance");
        assert_eq!(parameters["horizon"], Level0ACanary::HORIZON);
        assert_eq!(parameters["sequence_count"], 33);
        assert_eq!(parameters["zero_action_sequence_index"], 0);
        assert_eq!(parameters["scale_gain"].as_f64().unwrap(), Level0BCanary::SCALE_GAIN);
        assert_eq!(parameters["angle_gain"].as_f64().unwrap(), Level0BCanary::ANGLE_GAIN);
        assert_eq!(
            parameters["impulse_amplitude"].as_f64().unwrap(),
            Level0ACanary::IMPULSE_AMPLITUDE
        );
        assert_eq!(parameters["trains_fern"], false);
        assert_eq!(parameters["fern_patch_width"], Level0ACanary::HORIZON);
        assert_eq!(parameters["fern_block_axis"], "timestep_major_per_channel");
        let state_transition = Level0ACanary::state_transition();
        let process_covariance = Level0ACanary::process_covariance();
        let action_response = Level0ACanary::action_response();
        let initial_state = Level0ACanary::initial_state();
        for row in 0..2 {
            for column in 0..2 {
                assert_eq!(
                    parameters["state_transition"][row][column].as_f64().unwrap(),
                    state_transition[(row, column)]
                );
                assert_eq!(
                    parameters["process_covariance"][row][column].as_f64().unwrap(),
                    process_covariance[(row, column)]
                );
                assert_eq!(
                    parameters["action_response"][row][column].as_f64().unwrap(),
                    action_response[(row, column)]
                );
            }
            assert_eq!(parameters["initial_state"][row].as_f64().unwrap(), initial_state[row]);
        }
    }

    #[test]
    fn level0b_canary_is_falsifiable() {
        let report = Level0BCanary::evaluate().unwrap();
        assert!(report.mean_matches_level0a);
        assert_eq!(report.closed_form_mixed_residual, 0.0);
        assert!(report.orientation_capacity_nats < 1e-10);
        assert_eq!(
            report.detected_defects,
            vec!["constant_Q0", "swapped_scale_rotation", "wrong_B01", "covariance_lag"]
        );
    }

    #[test]
    fn level0c_canary_is_falsifiable() {
        let report = Level0CCanary::evaluate().unwrap();
        assert_eq!(report.shared_fourth_moments, [8.0, 9.0, 12.0]);
        assert_eq!(
            report.detected_defects,
            vec![
                "gaussian_W",
                "swapped_alpha",
                "rate_not_shape",
                "independent_labeled_shared"
            ]
        );
        assert!(report.independent_horizon8_fourth_moments[1] > 8.0);
        assert!(report.independent_horizon8_fourth_moments[2] > report.independent_horizon8_fourth_moments[1]);
        assert_eq!(report.second_moment, 2.0);
        assert_eq!(report.characteristic_function_residual, 0.0);
        assert!(report.sign_matched);
    }

    #[test]
    fn action_independent_gamma_attenuation_has_equal_covariance_and_fading_tails() {
        let report = ActionIndependentGammaAttenuationCanary::evaluate().unwrap();
        assert_eq!(report.one_step_second_moment, 2.0);
        assert_eq!(report.one_step_fourth_moment, 10.0);
        assert_eq!(report.horizon8_fourth_moment, 8.689128941802204);
        assert_eq!(report.horizon24_fourth_moment, 8.684563808640855);
        assert_eq!(
            report.process_covariance,
            DMatrix::from_diagonal(&DVector::from_vec(vec![0.0225, 0.0225]))
        );
        assert!(report.horizon8_fourth_moment > 8.0);
        assert!(report.horizon8_fourth_moment < report.one_step_fourth_moment);
        assert!(report.horizon24_fourth_moment < report.horizon8_fourth_moment);
    }

    #[test]
    fn action_independent_gaussian_control_matches_covariance_without_gamma_tails() {
        let gaussian = ActionIndependentGaussianAttenuationCanary::evaluate().unwrap();
        let gamma = ActionIndependentGammaAttenuationCanary::evaluate().unwrap();
        assert_eq!(gaussian.one_step_second_moment, 2.0);
        assert_eq!(gaussian.one_step_fourth_moment, 8.0);
        assert_eq!(gaussian.horizon8_fourth_moment, 8.0);
        assert_eq!(gaussian.process_covariance, gamma.process_covariance);
        assert!(gamma.one_step_fourth_moment > gaussian.one_step_fourth_moment);
        assert!(gamma.horizon8_fourth_moment > gaussian.horizon8_fourth_moment);
    }
}
