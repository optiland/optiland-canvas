//! Timings for docs/performance.md: the 2D transform, and the share of it
//! spent bridging `Cx` to rustfft's complex type (ADR 0003 promised to measure
//! this; above 15% the internals would switch type).
//!
//! `cargo run -p oc-wave --release --example bench_wave`

use oc_wave::{Cx, Fft2, Grid2};
use std::hint::black_box;
use std::time::Instant;

fn median_ms(mut f: impl FnMut(), reps: usize) -> f64 {
    let mut t: Vec<f64> = (0..reps)
        .map(|_| {
            let t0 = Instant::now();
            f();
            t0.elapsed().as_secs_f64() * 1e3
        })
        .collect();
    t.sort_by(|a, b| a.partial_cmp(b).unwrap());
    t[reps / 2]
}

fn main() {
    println!("| n² | 2D FFT (ms) | bridge copy (ms) | bridge share |");
    println!("|---|---|---|---|");
    for n in [128usize, 256, 512, 1024, 2048] {
        let g = Grid2::square(n, 1e-3).unwrap();
        let mut data: Vec<Cx> = (0..g.len()).map(|k| Cx::new(k as f64, 1.0)).collect();
        let mut fft = Fft2::new();
        fft.forward(&mut data, g.n, 1.0); // plan once
        let reps = if n >= 1024 { 7 } else { 21 };
        let t_fft = median_ms(|| fft.forward(black_box(&mut data), g.n, 1.0), reps);
        // The bridge: one copy in, one copy out, the same as `Fft2` does.
        let mut buf: Vec<rustfft::num_complex::Complex64> = Vec::with_capacity(g.len());
        let t_copy = median_ms(
            || {
                buf.clear();
                buf.extend(data.iter().map(|z| rustfft::num_complex::Complex64::new(z.re, z.im)));
                for (z, c) in data.iter_mut().zip(buf.iter()) {
                    *z = Cx::new(c.re, c.im);
                }
                black_box(&data);
            },
            reps,
        );
        println!(
            "| {n}² | {t_fft:.2} | {t_copy:.2} | {:.1}% |",
            100.0 * t_copy / t_fft
        );
    }

    // One band-limited angular-spectrum step: pad 2×, transform, multiply,
    // transform back, crop. The unit the wave path is budgeted in.
    println!();
    println!("| field n² (padded) | band-limited ASM step (ms) |");
    println!("|---|---|");
    for n in [128usize, 256, 512, 1024] {
        let g = Grid2::square(n, 0.005).unwrap();
        let u = oc_wave::beams::gaussian(g, 0.5, 1.0, 0.1 * n as f64 / 256.0, 0.0, 1.0, [0.0, 0.0]).unwrap();
        let mut fft = Fft2::new();
        let how = oc_wave::Propagator::AngularSpectrum(oc_wave::AsmOptions::default());
        let _ = oc_wave::propagate(&u, 10.0, how, &mut fft).unwrap();
        let reps = if n >= 1024 { 5 } else { 15 };
        let t = median_ms(
            || {
                black_box(oc_wave::propagate(black_box(&u), 10.0, how, &mut fft).unwrap());
            },
            reps,
        );
        println!("| {n}² ({}²) | {t:.1} |", 2 * n);
    }
}
