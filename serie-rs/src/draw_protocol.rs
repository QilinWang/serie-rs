//! Draw-bank protocol shared by staged Rust data flows.
//!
//! This mirrors the public Python `serie` semantics without requiring Rust to
//! own the full Python API. The important rule is statistical: deterministic
//! systems use one trajectory draw crossed with observation-noise draws, while
//! stochastic systems use paired trajectory/noise draws by ordinal by default.

use crate::stage_synthetic::TrajectorySpec;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DrawRef {
    pub seed_base: u64,
    pub draw_ordinal: usize,
    pub draw_seed: u64,
}

impl DrawRef {
    pub fn new(seed_base: u64, draw_ordinal: usize) -> Self {
        Self {
            seed_base,
            draw_ordinal,
            draw_seed: draw_seed(seed_base, draw_ordinal),
        }
    }

    pub fn try_from_parts(seed_base: u64, draw_ordinal: usize, provided_draw_seed: u64) -> Result<Self, String> {
        let expected = draw_seed(seed_base, draw_ordinal);
        if provided_draw_seed != expected {
            return Err(format!(
                "draw_seed mismatch for seed_base={seed_base}, draw_ordinal={draw_ordinal}: got {provided_draw_seed}, expected {expected}"
            ));
        }
        Ok(Self {
            seed_base,
            draw_ordinal,
            draw_seed: provided_draw_seed,
        })
    }
}

pub fn spawn_draw_seeds(seed_base: u64, n_draws: usize) -> Result<Vec<u64>, String> {
    if n_draws == 0 {
        return Err("n_draws must be >= 1".to_string());
    }
    Ok((0..n_draws)
        .map(|draw_ordinal| draw_seed(seed_base, draw_ordinal))
        .collect())
}

fn draw_seed(seed_base: u64, draw_ordinal: usize) -> u64 {
    // Stable SplitMix64-style derivation, truncated to a uint32-compatible
    // child seed. The active Python transition worker mirrors this rule in
    // serie.spawn_draw_seeds while data generation is still Python-backed.
    let mut x = seed_base ^ 0x9E37_79B9_7F4A_7C15u64.wrapping_mul((draw_ordinal as u64).wrapping_add(1));
    x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    (x ^ (x >> 31)) & 0xFFFF_FFFF
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DrawDesign {
    Auto,
    NoiseOnly,
    Joint,
    TrajectoryFixed,
    CrossProduct,
}

#[derive(Debug, Clone, PartialEq)]
pub struct DgpDrawBank {
    pub source_name: String,
    pub is_stochastic: bool,
    pub trajectory_seed_base: u64,
    pub noise_seed_base: u64,
    pub n_trajectory_draws: usize,
    pub n_noise_draws: usize,
    pub rows: usize,
    pub dim: usize,
}

impl DgpDrawBank {
    pub fn new(
        source_name: impl Into<String>,
        is_stochastic: bool,
        trajectory_seed_base: u64,
        noise_seed_base: u64,
        n_trajectory_draws: usize,
        n_noise_draws: usize,
        rows: usize,
        dim: usize,
    ) -> Result<Self, String> {
        if n_trajectory_draws == 0 {
            return Err("n_trajectory_draws must be >= 1".to_string());
        }
        if n_noise_draws == 0 {
            return Err("n_noise_draws must be >= 1".to_string());
        }
        Ok(Self {
            source_name: source_name.into(),
            is_stochastic,
            trajectory_seed_base,
            noise_seed_base,
            n_trajectory_draws,
            n_noise_draws,
            rows,
            dim,
        })
    }

    pub fn from_trajectory_spec(
        spec: &TrajectorySpec,
        trajectory_seed_base: u64,
        noise_seed_base: u64,
        n_trajectory_draws: usize,
        n_noise_draws: usize,
    ) -> Result<Self, String> {
        Self::new(
            spec.tag(),
            spec.system.is_stochastic(),
            trajectory_seed_base,
            noise_seed_base,
            n_trajectory_draws,
            n_noise_draws,
            spec.steps + 1 - spec.burn_in,
            spec.dim,
        )
    }

    pub fn trajectory_draw(&self, draw_ordinal: usize) -> Result<DrawRef, String> {
        if draw_ordinal >= self.n_trajectory_draws {
            return Err(format!(
                "trajectory_draw_ordinal={draw_ordinal} outside [0, {})",
                self.n_trajectory_draws
            ));
        }
        Ok(DrawRef::new(self.trajectory_seed_base, draw_ordinal))
    }

    pub fn noise_draw(&self, draw_ordinal: usize) -> Result<DrawRef, String> {
        if draw_ordinal >= self.n_noise_draws {
            return Err(format!(
                "noise_draw_ordinal={draw_ordinal} outside [0, {})",
                self.n_noise_draws
            ));
        }
        Ok(DrawRef::new(self.noise_seed_base, draw_ordinal))
    }

    pub fn cell(
        &self,
        noise_level: f32,
        trajectory_draw_ordinal: usize,
        noise_draw_ordinal: usize,
    ) -> Result<TitrationCell, String> {
        Ok(TitrationCell {
            source_name: self.source_name.clone(),
            noise_level,
            trajectory_draw: self.trajectory_draw(trajectory_draw_ordinal)?,
            noise_draw: self.noise_draw(noise_draw_ordinal)?,
            rows: self.rows,
            dim: self.dim,
        })
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct TitrationCell {
    pub source_name: String,
    pub noise_level: f32,
    pub trajectory_draw: DrawRef,
    pub noise_draw: DrawRef,
    pub rows: usize,
    pub dim: usize,
}

impl TitrationCell {
    pub fn tag(&self) -> String {
        format!(
            "{}_traj{}_sig{}_noise{}",
            self.source_name,
            self.trajectory_draw.draw_ordinal,
            format_tag_float(f64::from(self.noise_level)),
            self.noise_draw.draw_ordinal
        )
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct TitrationPlan {
    pub noise_levels: Vec<f32>,
    pub n_trajectory_draws: usize,
    pub n_noise_draws: usize,
    pub draw_design: DrawDesign,
}

impl TitrationPlan {
    pub fn new(noise_levels: Vec<f32>, n_trajectory_draws: usize, n_noise_draws: usize) -> Result<Self, String> {
        Self::with_design(noise_levels, n_trajectory_draws, n_noise_draws, DrawDesign::Auto)
    }

    pub fn with_design(
        noise_levels: Vec<f32>,
        n_trajectory_draws: usize,
        n_noise_draws: usize,
        draw_design: DrawDesign,
    ) -> Result<Self, String> {
        if noise_levels.is_empty() {
            return Err("TitrationPlan.noise_levels cannot be empty".to_string());
        }
        if n_trajectory_draws == 0 {
            return Err("n_trajectory_draws must be >= 1".to_string());
        }
        if n_noise_draws == 0 {
            return Err("n_noise_draws must be >= 1".to_string());
        }
        Ok(Self {
            noise_levels,
            n_trajectory_draws,
            n_noise_draws,
            draw_design,
        })
    }

    pub fn iter_cells(&self, bank: &DgpDrawBank) -> Result<Vec<TitrationCell>, String> {
        let design = self.resolved_draw_design(bank);
        self.require_available(bank)?;
        let mut cells = Vec::new();

        match design {
            DrawDesign::NoiseOnly => {
                if self.n_trajectory_draws != 1 {
                    return Err(
                        "noise_only requires n_trajectory_draws=1; use cross_product for an explicit full grid"
                            .to_string(),
                    );
                }
                for &noise_level in &self.noise_levels {
                    for noise_ordinal in 0..self.n_noise_draws {
                        cells.push(bank.cell(noise_level, 0, noise_ordinal)?);
                    }
                }
            }
            DrawDesign::Joint => {
                if !bank.is_stochastic {
                    return Err("joint requires a stochastic DGP bank".to_string());
                }
                if self.n_trajectory_draws != self.n_noise_draws {
                    return Err(format!(
                        "joint pairs trajectory and noise draws by spawn ordinal, so n_trajectory_draws must equal n_noise_draws; got {} and {}",
                        self.n_trajectory_draws, self.n_noise_draws
                    ));
                }
                for &noise_level in &self.noise_levels {
                    for ordinal in 0..self.n_trajectory_draws {
                        cells.push(bank.cell(noise_level, ordinal, ordinal)?);
                    }
                }
            }
            DrawDesign::TrajectoryFixed => {
                if !bank.is_stochastic {
                    return Err("trajectory_fixed requires a stochastic DGP bank".to_string());
                }
                if self.n_trajectory_draws != 1 {
                    return Err(
                        "trajectory_fixed is the conditional diagnostic with one trajectory path; set n_trajectory_draws=1"
                            .to_string(),
                    );
                }
                for &noise_level in &self.noise_levels {
                    for noise_ordinal in 0..self.n_noise_draws {
                        cells.push(bank.cell(noise_level, 0, noise_ordinal)?);
                    }
                }
            }
            DrawDesign::CrossProduct => {
                for trajectory_ordinal in 0..self.n_trajectory_draws {
                    for &noise_level in &self.noise_levels {
                        for noise_ordinal in 0..self.n_noise_draws {
                            cells.push(bank.cell(noise_level, trajectory_ordinal, noise_ordinal)?);
                        }
                    }
                }
            }
            DrawDesign::Auto => unreachable!("resolved_draw_design never returns Auto"),
        }

        Ok(cells)
    }

    fn resolved_draw_design(&self, bank: &DgpDrawBank) -> DrawDesign {
        match self.draw_design {
            DrawDesign::Auto if bank.is_stochastic => DrawDesign::Joint,
            DrawDesign::Auto => DrawDesign::NoiseOnly,
            design => design,
        }
    }

    fn require_available(&self, bank: &DgpDrawBank) -> Result<(), String> {
        if bank.n_trajectory_draws < self.n_trajectory_draws {
            return Err(format!(
                "bank has {} trajectory draws, plan requires {}",
                bank.n_trajectory_draws, self.n_trajectory_draws
            ));
        }
        if bank.n_noise_draws < self.n_noise_draws {
            return Err(format!(
                "bank has {} noise draws, plan requires {}",
                bank.n_noise_draws, self.n_noise_draws
            ));
        }
        Ok(())
    }
}

fn format_tag_float(value: f64) -> String {
    let mut formatted = format!("{value:.12}");
    while formatted.contains('.') && formatted.ends_with('0') {
        formatted.pop();
    }
    if formatted.ends_with('.') {
        formatted.pop();
    }
    formatted
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn spawned_draw_seeds_are_stable_and_not_consecutive() {
        let seeds = spawn_draw_seeds(42, 3).unwrap();
        assert_eq!(seeds, spawn_draw_seeds(42, 3).unwrap());
        assert_ne!(seeds, vec![42, 43, 44]);
        assert_ne!(seeds[0], seeds[1]);
    }

    #[test]
    fn draw_ref_validates_seed_recipe() {
        let draw = DrawRef::new(1000, 2);
        assert_eq!(DrawRef::try_from_parts(1000, 2, draw.draw_seed).unwrap(), draw);
        assert!(DrawRef::try_from_parts(1000, 2, draw.draw_seed + 1).is_err());
    }

    #[test]
    fn deterministic_auto_uses_one_trajectory_many_noise_draws() {
        let bank = DgpDrawBank::new("lorenz", false, 7, 1000, 1, 3, 100, 3).unwrap();
        let plan = TitrationPlan::new(vec![0.0, 1.0], 1, 3).unwrap();
        let cells = plan.iter_cells(&bank).unwrap();
        let pairs: Vec<_> = cells
            .iter()
            .map(|cell| (cell.trajectory_draw.draw_ordinal, cell.noise_draw.draw_ordinal))
            .collect();
        assert_eq!(pairs, vec![(0, 0), (0, 1), (0, 2), (0, 0), (0, 1), (0, 2)]);
    }

    #[test]
    fn deterministic_auto_rejects_multiple_trajectory_draws() {
        let bank = DgpDrawBank::new("lorenz", false, 7, 1000, 2, 2, 100, 3).unwrap();
        let plan = TitrationPlan::new(vec![0.0], 2, 2).unwrap();
        assert!(plan.iter_cells(&bank).unwrap_err().contains("noise_only"));
    }

    #[test]
    fn stochastic_auto_pairs_trajectory_and_noise_draws() {
        let bank = DgpDrawBank::new("ou", true, 7, 1000, 3, 3, 100, 1).unwrap();
        let plan = TitrationPlan::new(vec![0.0, 1.0], 3, 3).unwrap();
        let cells = plan.iter_cells(&bank).unwrap();
        let pairs: Vec<_> = cells
            .iter()
            .map(|cell| (cell.trajectory_draw.draw_ordinal, cell.noise_draw.draw_ordinal))
            .collect();
        assert_eq!(pairs, vec![(0, 0), (1, 1), (2, 2), (0, 0), (1, 1), (2, 2)]);
    }

    #[test]
    fn stochastic_auto_requires_equal_draw_counts() {
        let bank = DgpDrawBank::new("ou", true, 7, 1000, 1, 3, 100, 1).unwrap();
        let plan = TitrationPlan::new(vec![0.0], 1, 3).unwrap();
        assert!(plan.iter_cells(&bank).unwrap_err().contains("must equal"));
    }

    #[test]
    fn stochastic_cross_product_is_explicit() {
        let bank = DgpDrawBank::new("ou", true, 7, 1000, 2, 3, 100, 1).unwrap();
        let plan = TitrationPlan::with_design(vec![0.0], 2, 3, DrawDesign::CrossProduct).unwrap();
        let cells = plan.iter_cells(&bank).unwrap();
        assert_eq!(cells.len(), 6);
    }
}
