// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// bundle transitions (stage fn change)
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// print_diagnostics(...): no bundle boundary change (text diagnostics renderer)

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// diagnostics text
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// text diagnostics stage fn

use crate::stage_diag_cli::DiagnoseConfig;
use crate::stage_partition::PartitionStats;

use colored::Colorize;
use tabled::{
    Table, Tabled,
    settings::{Panel, Style},
};

// partition diagnostics
#[derive(Tabled)]
struct ChannelPreviewRow {
    #[tabled(rename = "Ch")]
    ch: usize,
    #[tabled(rename = "Mean")]
    mean: String,
    #[tabled(rename = "Std")]
    std: String,
    #[tabled(rename = "Median")]
    median: String,
    #[tabled(rename = "MAD")]
    mad: String,
    #[tabled(rename = "Min")]
    min: String,
    #[tabled(rename = "Max")]
    max: String,
}

pub fn print_diagnostics(train: &PartitionStats, val: &PartitionStats, test: &PartitionStats, cfg: &DiagnoseConfig) {
    let col_limit = cfg.col_limit;
    let drift_thr = cfg.drift_thr;
    let sparse_thr = cfg.sparse_thr;
    let heavy_thr = cfg.heavy_thr;
    let dim = train.dim();
    let total = train.count + val.count + test.count;
    let limit = dim.min(col_limit);

    // ── Counts ──
    // ══════════════════════════════════════════════
    //  Counts table
    // ══════════════════════════════════════════════

    #[derive(Tabled)]
    struct CountRow {
        #[tabled(rename = "Split")]
        split: String,
        #[tabled(rename = "Count")]
        count: String,
        #[tabled(rename = "Fraction")]
        frac: String,
    }

    let count_rows = vec![
        CountRow {
            split: "Train".green().bold().to_string(),
            count: format!("{}", train.count),
            frac: format!("{:.1}%", 100.0 * train.count as f64 / total as f64),
        },
        CountRow {
            split: "Val".yellow().bold().to_string(),
            count: format!("{}", val.count),
            frac: format!("{:.1}%", 100.0 * val.count as f64 / total as f64),
        },
        CountRow {
            split: "Test".red().bold().to_string(),
            count: format!("{}", test.count),
            frac: format!("{:.1}%", 100.0 * test.count as f64 / total as f64),
        },
        CountRow {
            split: "Total".dimmed().to_string(),
            count: format!("{}", total),
            frac: "100.0%".to_string(),
        },
    ];

    let count_table = Table::new(count_rows)
        .with(Style::rounded())
        .with(Panel::header("Split Sizes"))
        .to_string();

    // ══════════════════════════════════════════════
    //  Health table
    // ══════════════════════════════════════════════

    let heavy = train.heavy_tail_channels(heavy_thr);
    let constant = train.const_channels();
    let sparse = train.sparse_channels(sparse_thr);

    fn fmt_channels(channels: &[usize], limit: usize) -> String {
        if channels.is_empty() {
            "none".green().to_string()
        } else {
            let shown: Vec<String> = channels.iter().take(limit).map(|c| c.to_string()).collect();
            let s = format!("[{}]", shown.join(", "));
            if channels.len() > limit {
                format!("{}{}", s, "...").red().to_string()
            } else {
                s.red().to_string()
            }
        }
    }

    #[derive(Tabled)]
    struct HealthRow {
        #[tabled(rename = "Issue")]
        issue: String,
        #[tabled(rename = "Count")]
        count: String,
        #[tabled(rename = "Channels")]
        channels: String,
    }

    let health_rows = vec![
        HealthRow {
            issue: "Heavy tails".to_string(),
            count: format!("{}/{}", heavy.len(), dim),
            channels: fmt_channels(&heavy, 10),
        },
        HealthRow {
            issue: "Near constant".to_string(),
            count: format!("{}/{}", constant.len(), dim),
            channels: fmt_channels(&constant, 10),
        },
        HealthRow {
            issue: format!("Sparse (>{:.0}% zeros)", sparse_thr * 100.0),
            count: format!("{}/{}", sparse.len(), dim),
            channels: fmt_channels(&sparse, 10),
        },
    ];

    let health_table = Table::new(health_rows)
        .with(Style::rounded())
        .with(Panel::header("Distribution Health — Train"))
        .to_string();

    // ══════════════════════════════════════════════
    //  Drift table
    // ══════════════════════════════════════════════

    let val_shift = val.mean_shift_from(train);
    let test_shift = test.mean_shift_from(train);
    let val_ratio = val.std_ratio_from(train);
    let test_ratio = test.std_ratio_from(train);
    let has_drift = (val_ratio - 1.0).abs() > drift_thr || (test_ratio - 1.0).abs() > drift_thr;

    let status = if has_drift { "⚠ drift" } else { "✓" };

    fn style_ratio(ratio: f64, thr: f64) -> String {
        let s = format!("{:.3}", ratio);
        if (ratio - 1.0).abs() > thr {
            s.red().bold().to_string()
        } else {
            s
        }
    }

    #[derive(Tabled)]
    struct DriftRow {
        #[tabled(rename = "Metric")]
        metric: String,
        #[tabled(rename = "Val")]
        val: String,
        #[tabled(rename = "Test")]
        test: String,
    }

    let drift_rows = vec![
        DriftRow {
            metric: "Mean shift / train std".to_string(),
            val: format!("{:.4}", val_shift),
            test: format!("{:.4}", test_shift),
        },
        DriftRow {
            metric: "Std ratio (median)".to_string(),
            val: style_ratio(val_ratio, drift_thr),
            test: style_ratio(test_ratio, drift_thr),
        },
    ];

    let drift_table = Table::new(drift_rows)
        .with(Style::rounded())
        .with(Panel::header(format!("Cross-Split Drift (ref: Train)  [{}]", status)))
        .to_string();

    // ── Stats ──
    println!("\n── Channel Statistics — Train (D={}) ──", dim);
    let rows: Vec<ChannelPreviewRow> = (0..limit)
        .map(|i| ChannelPreviewRow {
            ch: i,
            mean: format!("{:.3}", train.mean[i]),
            std: format!("{:.3}", train.std[i]),
            median: format!("{:.3}", train.median[i]),
            mad: format!("{:.3}", train.mad[i]),
            min: format!("{:.3}", train.min[i]),
            max: format!("{:.3}", train.max[i]),
        })
        .collect();
    println!("{}", Table::new(rows));
    if dim > limit {
        println!("  ... +{} more", dim - limit);
    }

    // ══════════════════════════════════════════════
    //  Stats table
    // ══════════════════════════════════════════════

    #[derive(Tabled)]
    struct ChannelRow {
        #[tabled(rename = "Ch")]
        ch: String,
        #[tabled(rename = "Mean")]
        mean: String,
        #[tabled(rename = "Std")]
        std: String,
        #[tabled(rename = "Median")]
        median: String,
        #[tabled(rename = "MAD")]
        mad: String,
        #[tabled(rename = "Min")]
        min: String,
        #[tabled(rename = "Max")]
        max: String,
    }

    let mut stat_rows: Vec<ChannelRow> = (0..limit)
        .map(|i| ChannelRow {
            ch: format!("{}", i),
            mean: format!("{:.3}", train.mean[i]),
            std: format!("{:.3}", train.std[i]),
            median: format!("{:.3}", train.median[i]),
            mad: format!("{:.3}", train.mad[i]),
            min: format!("{:.3}", train.min[i]),
            max: format!("{:.3}", train.max[i]),
        })
        .collect();

    if dim > limit {
        stat_rows.push(ChannelRow {
            ch: format!("... +{}", dim - limit).dimmed().to_string(),
            mean: String::new(),
            std: String::new(),
            median: String::new(),
            mad: String::new(),
            min: String::new(),
            max: String::new(),
        });
    }

    let stats_table = Table::new(stat_rows)
        .with(Style::rounded())
        .with(Panel::header(format!("Channel Statistics — Train (D={})", dim)))
        .to_string();

    // ══════════════════════════════════════════════
    //  Print side by side (Python: Columns([...], padding=(0, 3)))
    // ══════════════════════════════════════════════

    // Side-by-side: split each table into lines, pad and zip
    let tables_side_by_side = side_by_side(&[&count_table, &health_table, &drift_table], 3);

    println!();
    println!("{}", "── Data Diagnostics ──".blue().bold());
    println!("{}", tables_side_by_side);
    println!("{}", stats_table);

    if has_drift {
        println!("{}", "⚠  Significant variance drift detected".red().bold());
    }
}

/// Render multiple table strings side by side with padding.
/// Python equivalent: rich.columns.Columns([t1, t2, t3], padding=(0, 3))
fn side_by_side(tables: &[&str], padding: usize) -> String {
    // Split each table into lines
    let table_lines: Vec<Vec<&str>> = tables.iter().map(|t| t.lines().collect()).collect();

    // Find max height and width per table
    let max_height = table_lines.iter().map(|t| t.len()).max().unwrap_or(0);
    let widths: Vec<usize> = table_lines
        .iter()
        .map(|t| t.iter().map(|l| l.chars().count()).max().unwrap_or(0))
        .collect();

    let pad = " ".repeat(padding);
    let mut result = String::new();

    for row in 0..max_height {
        for (i, lines) in table_lines.iter().enumerate() {
            let line = if row < lines.len() { lines[row] } else { "" };
            // Pad line to table width so columns align
            let visible_width = line.chars().count();
            let extra = widths[i].saturating_sub(visible_width);
            result.push_str(line);
            result.push_str(&" ".repeat(extra));
            if i < table_lines.len() - 1 {
                result.push_str(&pad);
            }
        }
        result.push('\n');
    }
    result
}
