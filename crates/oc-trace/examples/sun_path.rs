//! The sun's path over a year at one site, from the same NREL SPA the
//! canvas's sun rig uses, written as a table for a sun-path diagram.
//!
//! Every `STEP_MIN` minutes (default 10) of every day of `YEAR` (default
//! 2026), on the UTC clock, while the sun is up: azimuth (degrees clockwise
//! from north) and apparent elevation (refraction included).
//!
//! The site defaults to the Deutsches Optisches Museum, Jena (50.92781 N,
//! 11.57928 E, 160 m above sea level). `HEIGHT_M` raises the point above
//! that, a heliostat on the roof say: it only changes the parallax and the
//! pressure the refraction sees, and the run prints by how much.
//!
//! `cargo run --release -p oc-trace --example sun_path -- out.csv`

use oc_core::solar::{solar_position, SolarSite, UtcInstant};

fn env_f64(name: &str, default: f64) -> f64 {
    std::env::var(name).ok().and_then(|v| v.parse().ok()).unwrap_or(default)
}

fn site(height_m: f64) -> SolarSite {
    let elevation_m = env_f64("SITE_ELEV_M", 160.0) + height_m;
    SolarSite {
        latitude_deg: env_f64("LAT", 50.92781),
        longitude_deg: env_f64("LON", 11.57928),
        elevation_m,
        // As the canvas's sun panel sets it: a standard atmosphere's pressure
        // at that height.
        pressure_mbar: 1013.25 * (-elevation_m / 8434.0).exp(),
        temperature_c: 10.0,
        ..SolarSite::default()
    }
}

fn days_in(year: i32, month: u32) -> u32 {
    match month {
        2 if (year % 4 == 0 && year % 100 != 0) || year % 400 == 0 => 29,
        2 => 28,
        4 | 6 | 9 | 11 => 30,
        _ => 31,
    }
}

fn main() {
    let out = std::env::args().nth(1).unwrap_or_else(|| "sun_path.csv".into());
    let year = env_f64("YEAR", 2026.0) as i32;
    let step = env_f64("STEP_MIN", 10.0).max(1.0) as u32;
    let height = env_f64("HEIGHT_M", 0.0);
    let (at_floor, raised) = (site(0.0), site(height));

    let mut csv = String::from("month,day,utc_minutes,azimuth_deg,elevation_deg\n");
    let mut worst: f64 = 0.0;
    for month in 1..=12 {
        for day in 1..=days_in(year, month) {
            let mut m = 0;
            while m < 24 * 60 {
                let t = UtcInstant::new(year, month, day, m / 60, m % 60, 0.0);
                let p = solar_position(&raised, &t);
                if p.elevation_deg > -1.0 {
                    csv.push_str(&format!("{month},{day},{m},{:.4},{:.4}\n", p.azimuth_deg, p.elevation_deg));
                    let q = solar_position(&at_floor, &t);
                    if p.elevation_deg > 5.0 {
                        worst = worst.max((p.elevation_deg - q.elevation_deg).abs());
                    }
                }
                m += step;
            }
        }
    }
    std::fs::write(&out, csv).expect("write the table");
    eprintln!(
        "wrote {out}: {year}, every {step} min, {:.1} m above sea level; {height} m above the site moves the elevation by at most {:.2e} deg above 5 deg",
        raised.elevation_m, worst
    );
}
