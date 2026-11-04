// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// bundle transitions (stage fn change)
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// run_diagnostics(...): no bundle boundary change (CLI-facing text diagnostics)

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// diagnostics cli
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// cli hook structs

use crate::stage_diag_text;
use crate::stage_partition::PartitionStats;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DiagnoseConfig {
    pub col_limit: usize,
    pub drift_thr: f64,
    pub sparse_thr: f64,
    pub heavy_thr: f64,
}

impl Default for DiagnoseConfig {
    fn default() -> Self {
        Self {
            col_limit: 10,
            drift_thr: 0.3,
            sparse_thr: 0.3,
            heavy_thr: 1.5,
        }
    }
}

// cli hook stage fn
pub fn run_diagnostics(train: &PartitionStats, val: &PartitionStats, test: &PartitionStats, cfg: DiagnoseConfig) {
    stage_diag_text::print_diagnostics(train, val, test, &cfg);
}
