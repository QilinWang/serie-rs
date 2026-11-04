//! Exact discrete RK4 response oracle for smooth controlled ODEs.
//!
//! The action is held constant during each RK4 step. The returned `A_t` and
//! `B_t` differentiate the complete discrete step map; they are not the
//! continuous vector-field Jacobians. This distinction is load-bearing for
//! Fern Scientific III's local, rolled, and finite-intervention comparisons.

use nalgebra::{DMatrix, DVector};
use ndarray::Array4;

/// Lorenz-63 drift parameters. Control enters separately through the oracle's
/// constant rectangular control matrix.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Lorenz63Field {
    sigma: f64,
    rho: f64,
    beta: f64,
}

impl Lorenz63Field {
    pub fn try_new(sigma: f64, rho: f64, beta: f64) -> Result<Self, String> {
        for (name, value) in [("sigma", sigma), ("rho", rho), ("beta", beta)] {
            if !value.is_finite() {
                return Err(format!("{name} must be finite, got {value}"));
            }
        }
        if sigma <= 0.0 || beta <= 0.0 {
            return Err("Lorenz sigma and beta must be > 0".to_string());
        }
        Ok(Self { sigma, rho, beta })
    }
}

/// Rössler drift parameters. Control enters separately through the oracle's
/// constant rectangular control matrix.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RosslerField {
    a: f64,
    b: f64,
    c: f64,
}

impl RosslerField {
    pub fn try_new(a: f64, b: f64, c: f64) -> Result<Self, String> {
        for (name, value) in [("a", a), ("b", b), ("c", c)] {
            if !value.is_finite() {
                return Err(format!("{name} must be finite, got {value}"));
            }
        }
        Ok(Self { a, b, c })
    }
}

/// Autonomous Duffing drift. The action supplied to the oracle is the external
/// forcing, so no undeclared periodic driver is hidden in this field.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DuffingField {
    alpha: f64,
    beta: f64,
    damping: f64,
}

impl DuffingField {
    pub fn try_new(alpha: f64, beta: f64, damping: f64) -> Result<Self, String> {
        for (name, value) in [("alpha", alpha), ("beta", beta), ("damping", damping)] {
            if !value.is_finite() {
                return Err(format!("{name} must be finite, got {value}"));
            }
        }
        if damping < 0.0 {
            return Err("Duffing damping must be >= 0".to_string());
        }
        Ok(Self { alpha, beta, damping })
    }
}

/// Autonomous Van der Pol drift parameters.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct VanDerPolField {
    mu: f64,
}

impl VanDerPolField {
    pub fn try_new(mu: f64) -> Result<Self, String> {
        if !mu.is_finite() || mu < 0.0 {
            return Err(format!("Van der Pol mu must be finite and >= 0, got {mu}"));
        }
        Ok(Self { mu })
    }
}

/// Periodic Lorenz-96 lattice with a declared, fixed state dimension.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Lorenz96Field {
    dimension: usize,
    forcing: f64,
}

impl Lorenz96Field {
    pub fn try_new(dimension: usize, forcing: f64) -> Result<Self, String> {
        if dimension < 4 {
            return Err(format!("Lorenz-96 dimension must be >= 4, got {dimension}"));
        }
        if !forcing.is_finite() {
            return Err(format!("forcing must be finite, got {forcing}"));
        }
        Ok(Self { dimension, forcing })
    }
}

/// Typed smooth vector fields admitted by the analytic RK4 oracle.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum AutonomousControlledField {
    Lorenz63(Lorenz63Field),
    Rossler(RosslerField),
    Duffing(DuffingField),
    VanDerPol(VanDerPolField),
    Lorenz96(Lorenz96Field),
}

impl AutonomousControlledField {
    pub fn state_dimension(&self) -> usize {
        match self {
            Self::Lorenz63(_) | Self::Rossler(_) => 3,
            Self::Duffing(_) | Self::VanDerPol(_) => 2,
            Self::Lorenz96(field) => field.dimension,
        }
    }

    fn value_and_state_jacobian(&self, state: &DVector<f64>) -> (DVector<f64>, DMatrix<f64>) {
        match self {
            Self::Lorenz63(field) => {
                let (x, y, z) = (state[0], state[1], state[2]);
                (
                    DVector::from_vec(vec![
                        field.sigma * (y - x),
                        x * (field.rho - z) - y,
                        x * y - field.beta * z,
                    ]),
                    DMatrix::from_row_slice(
                        3,
                        3,
                        &[
                            -field.sigma,
                            field.sigma,
                            0.0,
                            field.rho - z,
                            -1.0,
                            -x,
                            y,
                            x,
                            -field.beta,
                        ],
                    ),
                )
            }
            Self::Rossler(field) => {
                let (x, y, z) = (state[0], state[1], state[2]);
                (
                    DVector::from_vec(vec![-y - z, x + field.a * y, field.b + z * (x - field.c)]),
                    DMatrix::from_row_slice(3, 3, &[0.0, -1.0, -1.0, 1.0, field.a, 0.0, z, 0.0, x - field.c]),
                )
            }
            Self::Duffing(field) => {
                let (position, velocity) = (state[0], state[1]);
                (
                    DVector::from_vec(vec![
                        velocity,
                        -field.damping * velocity - field.alpha * position - field.beta * position.powi(3),
                    ]),
                    DMatrix::from_row_slice(
                        2,
                        2,
                        &[
                            0.0,
                            1.0,
                            -field.alpha - 3.0 * field.beta * position.powi(2),
                            -field.damping,
                        ],
                    ),
                )
            }
            Self::VanDerPol(field) => {
                let (position, velocity) = (state[0], state[1]);
                (
                    DVector::from_vec(vec![
                        velocity,
                        field.mu * (1.0 - position * position) * velocity - position,
                    ]),
                    DMatrix::from_row_slice(
                        2,
                        2,
                        &[
                            0.0,
                            1.0,
                            -2.0 * field.mu * position * velocity - 1.0,
                            field.mu * (1.0 - position * position),
                        ],
                    ),
                )
            }
            Self::Lorenz96(field) => {
                let mut value = DVector::zeros(field.dimension);
                let mut state_jacobian = DMatrix::zeros(field.dimension, field.dimension);
                for coordinate in 0..field.dimension {
                    let previous_two = (coordinate + field.dimension - 2) % field.dimension;
                    let previous_one = (coordinate + field.dimension - 1) % field.dimension;
                    let next_one = (coordinate + 1) % field.dimension;
                    value[coordinate] = (state[next_one] - state[previous_two]) * state[previous_one]
                        - state[coordinate]
                        + field.forcing;
                    // Add rather than assign: at the minimum dimension,
                    // modular indices can coincide and derivative terms must
                    // accumulate.
                    state_jacobian[(coordinate, next_one)] += state[previous_one];
                    state_jacobian[(coordinate, previous_two)] -= state[previous_one];
                    state_jacobian[(coordinate, previous_one)] += state[next_one] - state[previous_two];
                    state_jacobian[(coordinate, coordinate)] -= 1.0;
                }
                (value, state_jacobian)
            }
        }
    }
}

/// One state plus the exact derivative of one declared discrete RK4 step.
#[derive(Clone, Debug, PartialEq)]
pub struct ControlledRk4Step {
    pub next_state: DVector<f64>,
    pub local_state_jacobian: DMatrix<f64>,
    pub local_action_jacobian: DMatrix<f64>,
}

/// Exact smooth state path and accumulated response operators.
#[derive(Clone, Debug, PartialEq)]
pub struct ControlledRk4Horizon {
    pub states: Vec<DVector<f64>>,
    pub local_state_jacobians: Vec<DMatrix<f64>>,
    pub local_action_jacobians: Vec<DMatrix<f64>>,
    pub initial_state_jacobians: Vec<DMatrix<f64>>,
    pub action_jacobian: Array4<f64>,
}

/// Zero-order-hold controlled RK4 oracle.
#[derive(Clone, Debug, PartialEq)]
pub struct ControlledRk4Oracle {
    field: AutonomousControlledField,
    timestep: f64,
    control_matrix: DMatrix<f64>,
}

impl ControlledRk4Oracle {
    pub fn try_new(
        field: AutonomousControlledField,
        timestep: f64,
        control_matrix: DMatrix<f64>,
    ) -> Result<Self, String> {
        if !timestep.is_finite() || timestep <= 0.0 {
            return Err(format!("timestep must be finite and > 0, got {timestep}"));
        }
        if control_matrix.nrows() != field.state_dimension() || control_matrix.ncols() == 0 {
            return Err(format!(
                "control_matrix must have nonempty shape {}xM, got {}x{}",
                field.state_dimension(),
                control_matrix.nrows(),
                control_matrix.ncols()
            ));
        }
        if let Some(value) = control_matrix.iter().copied().find(|value| !value.is_finite()) {
            return Err(format!("control_matrix must contain only finite values, found {value}"));
        }
        Ok(Self {
            field,
            timestep,
            control_matrix,
        })
    }

    pub fn state_dimension(&self) -> usize {
        self.field.state_dimension()
    }

    pub fn action_dimension(&self) -> usize {
        self.control_matrix.ncols()
    }

    fn stage(
        &self,
        state: &DVector<f64>,
        action: &DVector<f64>,
        state_sensitivity: &DMatrix<f64>,
        action_sensitivity: &DMatrix<f64>,
    ) -> (DVector<f64>, DMatrix<f64>, DMatrix<f64>) {
        let (field_value, field_state_jacobian) = self.field.value_and_state_jacobian(state);
        let value = field_value + &self.control_matrix * action;
        let value_from_state = &field_state_jacobian * state_sensitivity;
        let value_from_action = &field_state_jacobian * action_sensitivity + &self.control_matrix;
        (value, value_from_state, value_from_action)
    }

    /// Advance one interval and differentiate the complete RK4 map.
    pub fn step(&self, state: &DVector<f64>, action: &DVector<f64>) -> Result<ControlledRk4Step, String> {
        if state.len() != self.state_dimension() || action.len() != self.action_dimension() {
            return Err(format!(
                "state/action dimensions must be {}/{}, got {}/{}",
                self.state_dimension(),
                self.action_dimension(),
                state.len(),
                action.len()
            ));
        }
        if let Some(value) = state
            .iter()
            .chain(action.iter())
            .copied()
            .find(|value| !value.is_finite())
        {
            return Err(format!(
                "state and action must contain only finite values, found {value}"
            ));
        }

        let identity = DMatrix::identity(self.state_dimension(), self.state_dimension());
        let zero_action_sensitivity = DMatrix::zeros(self.state_dimension(), self.action_dimension());
        let (k1, k1_state, k1_action) = self.stage(state, action, &identity, &zero_action_sensitivity);
        let half_step = 0.5 * self.timestep;
        let stage_2_state_sensitivity = &identity + &k1_state * half_step;
        let stage_2_action_sensitivity = &k1_action * half_step;
        let (k2, k2_state, k2_action) = self.stage(
            &(state + &k1 * half_step),
            action,
            &stage_2_state_sensitivity,
            &stage_2_action_sensitivity,
        );
        let stage_3_state_sensitivity = &identity + &k2_state * half_step;
        let stage_3_action_sensitivity = &k2_action * half_step;
        let (k3, k3_state, k3_action) = self.stage(
            &(state + &k2 * half_step),
            action,
            &stage_3_state_sensitivity,
            &stage_3_action_sensitivity,
        );
        let stage_4_state_sensitivity = &identity + &k3_state * self.timestep;
        let stage_4_action_sensitivity = &k3_action * self.timestep;
        let (k4, k4_state, k4_action) = self.stage(
            &(state + &k3 * self.timestep),
            action,
            &stage_4_state_sensitivity,
            &stage_4_action_sensitivity,
        );

        let weight = self.timestep / 6.0;
        let next_state = state + (k1 + k2 * 2.0 + k3 * 2.0 + k4) * weight;
        let local_state_jacobian = identity + (k1_state + k2_state * 2.0 + k3_state * 2.0 + k4_state) * weight;
        let local_action_jacobian = (k1_action + k2_action * 2.0 + k3_action * 2.0 + k4_action) * weight;
        if let Some(value) = next_state.iter().copied().find(|value| !value.is_finite()) {
            return Err(format!("RK4 transition produced a non-finite state value {value}"));
        }
        Ok(ControlledRk4Step {
            next_state,
            local_state_jacobian,
            local_action_jacobian,
        })
    }

    /// Roll one complete action path and accumulate `Phi/G` in causal layout.
    pub fn direct_horizon(
        &self,
        initial_state: &DVector<f64>,
        future_actions: &DMatrix<f64>,
    ) -> Result<ControlledRk4Horizon, String> {
        if initial_state.len() != self.state_dimension() {
            return Err(format!(
                "initial_state must have {} values, got {}",
                self.state_dimension(),
                initial_state.len()
            ));
        }
        if future_actions.ncols() != self.action_dimension() {
            return Err(format!(
                "future_actions must have {} columns, got {}",
                self.action_dimension(),
                future_actions.ncols()
            ));
        }
        if let Some(value) = initial_state
            .iter()
            .chain(future_actions.iter())
            .copied()
            .find(|value| !value.is_finite())
        {
            return Err(format!(
                "initial state and actions must contain only finite values, found {value}"
            ));
        }

        let horizon = future_actions.nrows();
        let mut state = initial_state.clone();
        let mut states = Vec::with_capacity(horizon);
        let mut local_state_jacobians = Vec::with_capacity(horizon);
        let mut local_action_jacobians = Vec::with_capacity(horizon);
        let mut initial_state_jacobians = Vec::with_capacity(horizon);
        let mut initial_state_sensitivity = DMatrix::identity(self.state_dimension(), self.state_dimension());
        let mut action_sensitivities = vec![DMatrix::zeros(self.state_dimension(), self.action_dimension()); horizon];
        let mut action_jacobian = Array4::zeros((horizon, self.state_dimension(), horizon, self.action_dimension()));

        for timestep in 0..horizon {
            let action = future_actions.row(timestep).transpose();
            let step = self.step(&state, &action)?;
            initial_state_sensitivity = &step.local_state_jacobian * initial_state_sensitivity;
            for sensitivity in &mut action_sensitivities[..timestep] {
                *sensitivity = &step.local_state_jacobian * &*sensitivity;
            }
            action_sensitivities[timestep] = step.local_action_jacobian.clone();
            for action_timestep in 0..=timestep {
                for state_coordinate in 0..self.state_dimension() {
                    for action_coordinate in 0..self.action_dimension() {
                        action_jacobian[(timestep, state_coordinate, action_timestep, action_coordinate)] =
                            action_sensitivities[action_timestep][(state_coordinate, action_coordinate)];
                    }
                }
            }
            state = step.next_state;
            states.push(state.clone());
            local_state_jacobians.push(step.local_state_jacobian);
            local_action_jacobians.push(step.local_action_jacobian);
            initial_state_jacobians.push(initial_state_sensitivity.clone());
        }

        Ok(ControlledRk4Horizon {
            states,
            local_state_jacobians,
            local_action_jacobians,
            initial_state_jacobians,
            action_jacobian,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const FINITE_DIFFERENCE_STEP: f64 = 1e-6;
    const DERIVATIVE_TOLERANCE: f64 = 3e-8;

    fn assert_step_derivatives(oracle: &ControlledRk4Oracle, state: DVector<f64>, action: DVector<f64>) {
        let step = oracle.step(&state, &action).unwrap();
        for coordinate in 0..oracle.state_dimension() {
            let mut plus_state = state.clone();
            let mut minus_state = state.clone();
            plus_state[coordinate] += FINITE_DIFFERENCE_STEP;
            minus_state[coordinate] -= FINITE_DIFFERENCE_STEP;
            let finite_difference = (oracle.step(&plus_state, &action).unwrap().next_state
                - oracle.step(&minus_state, &action).unwrap().next_state)
                / (2.0 * FINITE_DIFFERENCE_STEP);
            assert!((step.local_state_jacobian.column(coordinate) - finite_difference).norm() < DERIVATIVE_TOLERANCE);
        }
        for coordinate in 0..oracle.action_dimension() {
            let mut plus_action = action.clone();
            let mut minus_action = action.clone();
            plus_action[coordinate] += FINITE_DIFFERENCE_STEP;
            minus_action[coordinate] -= FINITE_DIFFERENCE_STEP;
            let finite_difference = (oracle.step(&state, &plus_action).unwrap().next_state
                - oracle.step(&state, &minus_action).unwrap().next_state)
                / (2.0 * FINITE_DIFFERENCE_STEP);
            assert!((step.local_action_jacobian.column(coordinate) - finite_difference).norm() < DERIVATIVE_TOLERANCE);
        }
    }

    #[test]
    fn every_smooth_field_matches_an_independent_finite_difference() {
        let cases = [
            (
                AutonomousControlledField::Lorenz63(Lorenz63Field::try_new(10.0, 28.0, 8.0 / 3.0).unwrap()),
                DVector::from_vec(vec![1.0, 0.98, 1.1]),
                DMatrix::from_row_slice(3, 1, &[1.0, 0.0, 0.25]),
                DVector::from_vec(vec![0.3]),
            ),
            (
                AutonomousControlledField::Rossler(RosslerField::try_new(0.2, 0.2, 5.7).unwrap()),
                DVector::from_vec(vec![1.0, 0.4, 0.2]),
                DMatrix::from_row_slice(3, 1, &[0.0, 1.0, 0.1]),
                DVector::from_vec(vec![-0.3]),
            ),
            (
                AutonomousControlledField::Duffing(DuffingField::try_new(1.0, 0.5, 0.2).unwrap()),
                DVector::from_vec(vec![0.4, -0.7]),
                DMatrix::from_row_slice(2, 1, &[0.0, 1.2]),
                DVector::from_vec(vec![-0.2]),
            ),
            (
                AutonomousControlledField::VanDerPol(VanDerPolField::try_new(2.0).unwrap()),
                DVector::from_vec(vec![-0.6, 0.8]),
                DMatrix::from_row_slice(2, 1, &[0.0, 0.9]),
                DVector::from_vec(vec![0.5]),
            ),
            (
                AutonomousControlledField::Lorenz96(Lorenz96Field::try_new(4, 8.0).unwrap()),
                DVector::from_vec(vec![1.2, -0.4, 0.7, 2.1]),
                DMatrix::from_row_slice(4, 1, &[1.0, 0.0, 0.25, -0.2]),
                DVector::from_vec(vec![0.4]),
            ),
            (
                AutonomousControlledField::Lorenz96(Lorenz96Field::try_new(6, 8.0).unwrap()),
                DVector::from_vec(vec![1.01, 1.0, 1.0, 1.0, 1.0, 1.0]),
                DMatrix::from_row_slice(6, 1, &[1.0, 0.0, 0.0, 0.0, 0.0, 0.2]),
                DVector::from_vec(vec![-0.1]),
            ),
        ];

        for (field, state, control_matrix, action) in cases {
            let oracle = ControlledRk4Oracle::try_new(field, 0.01, control_matrix).unwrap();
            assert_step_derivatives(&oracle, state, action);
        }
    }

    #[test]
    fn minimum_lorenz96_field_jacobian_accumulates_wrapped_terms() {
        let field = AutonomousControlledField::Lorenz96(Lorenz96Field::try_new(4, 8.0).unwrap());
        let state = DVector::from_vec(vec![1.2, -0.4, 0.7, 2.1]);
        let (_, state_jacobian) = field.value_and_state_jacobian(&state);

        for coordinate in 0..field.state_dimension() {
            let mut plus_state = state.clone();
            let mut minus_state = state.clone();
            plus_state[coordinate] += FINITE_DIFFERENCE_STEP;
            minus_state[coordinate] -= FINITE_DIFFERENCE_STEP;
            let (plus_value, _) = field.value_and_state_jacobian(&plus_state);
            let (minus_value, _) = field.value_and_state_jacobian(&minus_state);
            let finite_difference = (plus_value - minus_value) / (2.0 * FINITE_DIFFERENCE_STEP);
            assert!((state_jacobian.column(coordinate) - finite_difference).norm() < 2e-9);
        }
    }

    #[test]
    fn rossler_and_lorenz96_canaries_match_python() {
        let rossler = ControlledRk4Oracle::try_new(
            AutonomousControlledField::Rossler(RosslerField::try_new(0.2, 0.2, 5.7).unwrap()),
            0.01,
            DMatrix::from_row_slice(3, 1, &[0.0, 1.0, 0.1]),
        )
        .unwrap();
        let rossler_step = rossler
            .step(&DVector::from_vec(vec![1.0, 0.4, 0.2]), &DVector::from_vec(vec![-0.3]))
            .unwrap();
        let expected_rossler = DVector::from_vec(vec![0.9939989973514264, 0.40777778225122085, 0.19247239040430125]);
        assert!((rossler_step.next_state - expected_rossler).norm() < 3e-15);

        let lorenz96 = ControlledRk4Oracle::try_new(
            AutonomousControlledField::Lorenz96(Lorenz96Field::try_new(4, 8.0).unwrap()),
            0.01,
            DMatrix::from_row_slice(4, 1, &[1.0, 0.0, 0.25, -0.2]),
        )
        .unwrap();
        let lorenz96_step = lorenz96
            .step(
                &DVector::from_vec(vec![1.2, -0.4, 0.7, 2.1]),
                &DVector::from_vec(vec![0.4]),
            )
            .unwrap();
        let expected_lorenz96 = DVector::from_vec(vec![
            1.2482369641061575,
            -0.3334677969764246,
            0.7703113026582239,
            2.1695469062617065,
        ]);
        assert!((lorenz96_step.next_state - expected_lorenz96).norm() < 4e-15);
    }

    #[test]
    fn invalid_field_contracts_are_refused() {
        assert!(
            Lorenz96Field::try_new(3, 8.0)
                .unwrap_err()
                .contains("dimension must be >= 4")
        );
        assert!(
            Lorenz96Field::try_new(4, f64::NAN)
                .unwrap_err()
                .contains("forcing must be finite")
        );
        assert!(
            RosslerField::try_new(0.2, f64::INFINITY, 5.7)
                .unwrap_err()
                .contains("b must be finite")
        );
    }

    #[test]
    fn lorenz_canary_matches_python_and_horizon_action_response_is_causal() {
        let oracle = ControlledRk4Oracle::try_new(
            AutonomousControlledField::Lorenz63(Lorenz63Field::try_new(10.0, 28.0, 8.0 / 3.0).unwrap()),
            0.01,
            DMatrix::from_row_slice(3, 1, &[1.0, 0.0, 0.25]),
        )
        .unwrap();
        let initial_state = DVector::from_vec(vec![1.0, 0.98, 1.1]);
        let actions = DMatrix::from_row_slice(3, 1, &[0.3, -0.2, 0.1]);
        let first_step = oracle.step(&initial_state, &DVector::from_vec(vec![0.3])).unwrap();
        let expected_state = DVector::from_vec(vec![1.0134846388174992, 1.2392546924982812, 1.0828033068138028]);
        assert!((first_step.next_state - expected_state).norm() < 3e-15);

        let horizon = oracle.direct_horizon(&initial_state, &actions).unwrap();
        for action_timestep in 0..actions.nrows() {
            for future_timestep in 0..action_timestep {
                for state_coordinate in 0..oracle.state_dimension() {
                    assert_eq!(
                        horizon.action_jacobian[(future_timestep, state_coordinate, action_timestep, 0)],
                        0.0
                    );
                }
            }
            let mut plus_actions = actions.clone();
            let mut minus_actions = actions.clone();
            plus_actions[(action_timestep, 0)] += FINITE_DIFFERENCE_STEP;
            minus_actions[(action_timestep, 0)] -= FINITE_DIFFERENCE_STEP;
            let plus = oracle.direct_horizon(&initial_state, &plus_actions).unwrap();
            let minus = oracle.direct_horizon(&initial_state, &minus_actions).unwrap();
            for future_timestep in action_timestep..actions.nrows() {
                let finite_difference =
                    (&plus.states[future_timestep] - &minus.states[future_timestep]) / (2.0 * FINITE_DIFFERENCE_STEP);
                for state_coordinate in 0..oracle.state_dimension() {
                    assert!(
                        (horizon.action_jacobian[(future_timestep, state_coordinate, action_timestep, 0)]
                            - finite_difference[state_coordinate])
                            .abs()
                            < DERIVATIVE_TOLERANCE
                    );
                }
            }
        }
    }
}
