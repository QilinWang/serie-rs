use std::fs::File;
use std::hint::black_box;
use std::io::{BufWriter, Read, Write};
use std::time::Instant;

// The benchmark runner generates this file by extracting the original function
// and return type directly from serie-rs/src/stage_ready.rs.
include!(env!("SERIE_WINDOW_SOURCE"));

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    let n: usize = args[3].parse()?;
    let d: usize = args[4].parse()?;
    let seq: usize = args[5].parse()?;
    let pred: usize = args[6].parse()?;
    let repeats: usize = args[7].parse()?;
    let mut bytes = Vec::new();
    File::open(&args[1])?.read_to_end(&mut bytes)?;
    assert_eq!(bytes.len(), n * d * 4);
    let input: Vec<f32> = bytes
        .chunks_exact(4)
        .map(|x| f32::from_le_bytes(x.try_into().unwrap()))
        .collect();
    for _ in 0..2 {
        black_box(sliding_window(black_box(&input), n, d, seq, pred));
    }
    let mut samples = Vec::with_capacity(repeats);
    for _ in 0..repeats {
        let start = Instant::now();
        let output = black_box(sliding_window(black_box(&input), n, d, seq, pred));
        let elapsed = start.elapsed().as_nanos();
        samples.push(elapsed);
        drop(output);
    }
    for elapsed in samples {
        println!("{elapsed}");
    }
    // Validation and file I/O are outside all timed regions.
    let output = sliding_window(&input, n, d, seq, pred);
    assert_eq!(output.n_windows, n - seq - pred + 1);
    let mut writer = BufWriter::new(File::create(&args[2])?);
    for &value in output.x.iter().chain(&output.y) {
        writer.write_all(&value.to_le_bytes())?;
    }
    writer.flush()?;
    Ok(())
}
