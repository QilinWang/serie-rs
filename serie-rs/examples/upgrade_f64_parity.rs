//! Emit exact f64 bits for existing deterministic test fixtures, to compare
//! toolchains/dependencies. Inputs come from the crate's existing canary tests;
//! this does not train a model or introduce a new experiment configuration.
use nalgebra::{DMatrix, DVector};
use serde_json::{Value, json};
use serie_rs::controlled_linear::{Level0ACanary, Level0BCanary};
use serie_rs::controlled_ode::{AutonomousControlledField, ControlledRk4Oracle, Lorenz63Field};
use serie_rs::controlled_pendulum::{ControlledPendulum, PendulumState};

fn bits<'a>(values: impl Iterator<Item = &'a f64>) -> Vec<u64> {
    values.map(|value| value.to_bits()).collect()
}

// nalgebra's iterator order is column-major; record dimensions alongside bits.
fn matrix(value: &DMatrix<f64>) -> Value {
    json!({"shape": [value.nrows(), value.ncols()], "column_major_bits": bits(value.iter())})
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let world = Level0ACanary::world()?;
    let bank = Level0ACanary::action_bank()?;
    let coverage = bank.coverage(Level0ACanary::RANK_TOLERANCE)?;
    let mut linear = Vec::new();
    for actions in bank.sequences() {
        let out = world.direct_horizon(&Level0ACanary::initial_state(), actions)?;
        linear.push(json!({
            "means": out.means.iter().map(|x| bits(x.iter())).collect::<Vec<_>>(),
            "covariances": out.covariances.iter().map(matrix).collect::<Vec<_>>(),
            "local_state_jacobians": out.local_state_jacobians.iter().map(matrix).collect::<Vec<_>>(),
            "local_action_jacobians": out.local_action_jacobians.iter().map(matrix).collect::<Vec<_>>(),
            "initial_state_jacobians": out.initial_state_jacobians.iter().map(matrix).collect::<Vec<_>>(),
            "action_jacobian_shape": out.action_jacobian.shape(),
            "action_jacobian_logical_order_bits": bits(out.action_jacobian.iter()),
        }));
    }
    let a = Level0ACanary::evaluate()?;
    let b = Level0BCanary::evaluate()?;
    let ode = ControlledRk4Oracle::try_new(
        AutonomousControlledField::Lorenz63(Lorenz63Field::try_new(10.0, 28.0, 8.0 / 3.0)?),
        0.01,
        DMatrix::from_row_slice(3, 1, &[1.0, 0.0, 0.25]),
    )?;
    let ode = ode.direct_horizon(
        &DVector::from_vec(vec![1.0, 0.98, 1.1]),
        &DMatrix::from_row_slice(3, 1, &[0.3, -0.2, 0.1]),
    )?;
    let pendulum = ControlledPendulum::try_new(0.05, 0.15, 9.81, 0.8)?.direct_horizon(
        PendulumState {
            angle: -0.2,
            angular_velocity: 0.4,
        },
        &[-0.7, 0.2, 0.9, -0.1],
    )?;
    let result = json!({
        "linear": linear,
        "linear_coverage": {
            "sample_count": coverage.sample_count,
            "action_dimension": coverage.action_dimension,
            "mean_bits": bits(coverage.mean.iter()),
            "covariance": matrix(&coverage.covariance),
            "singular_values_bits": bits(coverage.singular_values.iter()),
            "rank": coverage.rank,
            "rank_tolerance_bits": coverage.rank_tolerance.to_bits(),
        },
        "canary_a": {
            "instantaneous_rank": a.instantaneous_rank,
            "sequence_design_rank": a.sequence_design_rank,
            "closed_form_mixed_residual_bits": a.closed_form_mixed_residual.to_bits(),
            "block_seam_nats_bits": a.block_seam_nats.to_bits(),
            "orientation_capacity_nats_bits": a.orientation_capacity_nats.to_bits(),
            "detected_defects": a.detected_defects,
        },
        "canary_b": {
            "mean_matches_level0a": b.mean_matches_level0a,
            "closed_form_mixed_residual_bits": b.closed_form_mixed_residual.to_bits(),
            "block_seam_nats_bits": b.block_seam_nats.to_bits(),
            "orientation_capacity_nats_bits": b.orientation_capacity_nats.to_bits(),
            "detected_defects": b.detected_defects,
        },
        "ode": {
            "states": ode.states.iter().map(|x| bits(x.iter())).collect::<Vec<_>>(),
            "local_state_jacobians": ode.local_state_jacobians.iter().map(matrix).collect::<Vec<_>>(),
            "local_action_jacobians": ode.local_action_jacobians.iter().map(matrix).collect::<Vec<_>>(),
            "initial_state_jacobians": ode.initial_state_jacobians.iter().map(matrix).collect::<Vec<_>>(),
            "action_jacobian_shape": ode.action_jacobian.shape(),
            "action_jacobian_logical_order_bits": bits(ode.action_jacobian.iter()),
        },
        "pendulum": {
            "states_bits": pendulum.states.iter().map(|s| [s.angle.to_bits(), s.angular_velocity.to_bits()]).collect::<Vec<_>>(),
            "local_state_jacobians_column_major_bits": pendulum.local_state_jacobians.iter().map(|x| bits(x.iter())).collect::<Vec<_>>(),
            "local_action_jacobians_bits": pendulum.local_action_jacobians.iter().map(|x| bits(x.iter())).collect::<Vec<_>>(),
            "initial_state_jacobians_column_major_bits": pendulum.initial_state_jacobians.iter().map(|x| bits(x.iter())).collect::<Vec<_>>(),
            "action_jacobian_shape": pendulum.action_jacobian.shape(),
            "action_jacobian_logical_order_bits": bits(pendulum.action_jacobian.iter()),
        },
    });
    println!("{}", serde_json::to_string(&result)?);
    Ok(())
}
