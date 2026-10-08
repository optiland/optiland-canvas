//! Solar position and spectrum against published numbers.
//!
//! Two independent references:
//!
//! - The SPA report's own worked example, Reda & Andreas, NREL/TP-560-34302
//!   (revised January 2008), Appendix A.5, Table A5.1. Values are copied as
//!   printed there; the tolerances are half a unit in the last printed digit
//!   (or 1e-5° where the report prints more). Use the 2008 revision: the
//!   June 2003 original prints different Earth sums (L = 24.0182635175°,
//!   R = 0.9965421031 AU), which the revision corrected; the final zenith
//!   and azimuth are the same to the printed digits in both.
//! - pvlib-python 0.15.2, `pvlib.solarposition.spa_python(..., how="numpy",
//!   atmos_refract=0.5667)` with explicit ΔT and pressure. pvlib implements the
//!   same algorithm independently, so these agree far below SPA's own accuracy
//!   claim; they guard the transcription of the tables and every sign
//!   convention across hemispheres, seasons, centuries and a sun below the
//!   horizon.

use oc_core::solar;
use oc_core::solar::{
    am15_direct_lines, am15_direct_lines_equal_energy, am15_fraction, solar_position,
    sun_direction_enu, sun_half_angle_rad, SolarPosition, SolarSite, UtcInstant,
    AM15_DIRECT_TOTAL_W_M2, SUN_HALF_ANGLE_RAD,
};

fn close(got: f64, want: f64, tol: f64, what: &str) {
    assert!(
        (got - want).abs() <= tol,
        "{what}: got {got}, want {want} (|Δ| = {:e}, tol {tol:e})",
        (got - want).abs()
    );
}

fn spa_example() -> (SolarSite, UtcInstant, SolarPosition) {
    let site = SolarSite {
        latitude_deg: 39.742476,
        longitude_deg: -105.1786,
        elevation_m: 1830.14,
        pressure_mbar: 820.0,
        temperature_c: 11.0,
        delta_t_s: 67.0,
        atmos_refract_deg: 0.5667,
    };
    let t = UtcInstant::from_local(2003, 10, 17, 12, 30, 30.0, -7.0);
    let p = solar_position(&site, &t);
    (site, t, p)
}

#[test]
fn local_time_rolls_to_utc() {
    let (_, t, _) = spa_example();
    assert_eq!(t, UtcInstant::new(2003, 10, 17, 19, 30, 30.0));
    // Across a year boundary, both directions, and a half-hour zone.
    assert_eq!(
        UtcInstant::from_local(2023, 12, 31, 20, 0, 0.0, -5.0),
        UtcInstant::new(2024, 1, 1, 1, 0, 0.0)
    );
    assert_eq!(
        UtcInstant::from_local(2024, 3, 1, 2, 15, 7.25, 5.5),
        UtcInstant::new(2024, 2, 29, 20, 45, 7.25)
    );
}

/// Table A5.1, Julian day and the heliocentric position of the Earth.
#[test]
fn spa_report_heliocentric() {
    let (_, _, p) = spa_example();
    close(p.julian_day, 2452930.312847, 5e-7, "JD");
    close(p.heliocentric_longitude_deg, 24.0182616917, 5e-11, "L");
    close(p.heliocentric_latitude_deg, -0.0001011219, 5e-11, "B");
    close(p.earth_sun_distance_au, 0.9965422974, 5e-11, "R");
}

/// Table A5.1, nutation, obliquity, apparent longitude, geocentric sun.
#[test]
fn spa_report_nutation_and_geocentric() {
    let (_, _, p) = spa_example();
    close(p.nutation_longitude_deg, -0.00399840, 5e-9, "Δψ");
    close(p.nutation_obliquity_deg, 0.00166657, 5e-9, "Δε");
    close(p.true_obliquity_deg, 23.440465, 5e-7, "ε");
    close(p.apparent_sun_longitude_deg, 204.0085519281, 5e-10, "λ");
    close(p.right_ascension_deg, 202.22741, 5e-6, "α");
    close(p.declination_deg, -9.31434, 5e-6, "δ");
    // Printed as 11.105900; the unrounded value is 11.1059020 (pvlib agrees).
    close(p.hour_angle_deg, 11.105900, 3e-6, "H");
    close(p.sun_mean_longitude_deg, 205.8971722516, 5e-10, "M");
}

/// Table A5.1, the topocentric answer, the incidence on the example's tilted
/// surface (slope 30°, azimuth rotation −10°), and the equation of time.
#[test]
fn spa_report_topocentric() {
    let (_, _, p) = spa_example();
    // The report prints H′ = 11.10629, but its own H and α′ − α give
    // 11.105900 + 0.00037 = 11.10627; pvlib and this code both get
    // 11.1062705. The printed digit is off by 2e-5, so it is held to 3e-5.
    close(p.topocentric_hour_angle_deg, 11.10629, 3e-5, "H′");
    close(p.topocentric_right_ascension_deg, 202.22704, 5e-6, "α′");
    close(p.topocentric_declination_deg, -9.316179, 5e-7, "δ′");
    close(p.zenith_deg, 50.11162, 5e-6, "θ");
    close(p.azimuth_deg, 194.34024, 5e-6, "Φ");
    close(p.incidence_angle_deg(30.0, -10.0), 25.18700, 5e-6, "I");
    // The report prints 14.641503 min; pvlib and this code both give
    // 14.6415108, 8e-6 min (0.5 ms) apart. Held to 1e-5 min.
    close(p.equation_of_time_min, 14.641503, 1e-5, "E");
}

/// pvlib `spa_python` reference values, generated 2026-09-24 with pvlib 0.15.2:
/// `(utc, lat, lon, elev_m, pressure_mbar, temp_c, ΔT_s,
///   apparent_zenith, zenith, azimuth, equation_of_time)`.
#[allow(clippy::type_complexity)]
const PVLIB: [((i32, u32, u32, u32, u32, f64), [f64; 6], [f64; 4]); 9] = [
    (
        (2003, 10, 17, 19, 30, 30.0),
        [39.742476, -105.1786, 1830.14, 820.0, 11.0, 67.0],
        [
            50.111622024037,
            50.127954096160,
            194.340240510240,
            14.641510770822,
        ],
    ),
    (
        (2024, 6, 21, 10, 15, 0.0),
        [52.52, 13.405, 34.0, 1013.25, 20.0, 69.2],
        [
            30.798649502413,
            30.808410113619,
            155.620421963420,
            -1.902709061091,
        ],
    ),
    (
        (2019, 12, 3, 2, 47, 13.5),
        [-33.8688, 151.2093, 58.0, 1008.0, 24.0, 69.4],
        [
            18.140807393532,
            18.146064049705,
            306.656152588903,
            10.486070528783,
        ],
    ),
    (
        (1987, 3, 20, 16, 0, 0.0),
        [64.1466, -21.9426, 10.0, 990.0, -3.0, 55.5],
        [
            69.530049844421,
            69.575962359558,
            219.021405639669,
            -7.591282646626,
        ],
    ),
    // Night in Rio: the sun 46° below the horizon, so no refraction.
    (
        (2045, 9, 1, 23, 59, 59.0),
        [-22.9519, -43.2105, 700.0, 930.0, 28.0, 75.0],
        [
            135.953405265880,
            135.953405265880,
            257.009070460738,
            0.278795580205,
        ],
    ),
    (
        (1912, 1, 15, 12, 0, 0.0),
        [0.0, 0.0, 0.0, 1013.25, 12.0, 13.5],
        [
            21.433958058033,
            21.440561001197,
            174.121811374010,
            -9.210542377725,
        ],
    ),
    (
        (2031, 2, 11, 7, 5, 44.0),
        [23.5, 90.4, 5.0, 1015.0, 18.0, 71.0],
        [
            39.733038114186,
            39.746763641413,
            200.409720677233,
            -14.175594756693,
        ],
    ),
    // Just after sunrise in Zurich: refraction is 0.146° here.
    (
        (2026, 9, 24, 5, 52, 0.0),
        [47.3769, 8.5417, 408.0, 965.0, 9.0, 69.0],
        [
            84.459812869296,
            84.605710476865,
            96.612407213507,
            7.892418908676,
        ],
    ),
    // Svalbard, midnight sun.
    (
        (2011, 7, 4, 3, 0, 0.0),
        [78.2232, 15.6267, 20.0, 1005.0, 4.0, 66.3],
        [
            73.338360348017,
            73.394608046352,
            55.957853922083,
            -4.307444049933,
        ],
    ),
];

#[test]
fn matches_pvlib_spa_python() {
    let mut worst: f64 = 0.0;
    for ((y, mo, d, h, mi, s), [lat, lon, el, p, t, dt], [z_app, z, az, eot]) in PVLIB {
        let site = SolarSite {
            latitude_deg: lat,
            longitude_deg: lon,
            elevation_m: el,
            pressure_mbar: p,
            temperature_c: t,
            delta_t_s: dt,
            atmos_refract_deg: 0.5667,
        };
        let pos = solar_position(&site, &UtcInstant::new(y, mo, d, h, mi, s));
        let tag = format!("{y}-{mo:02}-{d:02}");
        close(
            pos.zenith_deg,
            z_app,
            1e-9,
            &format!("{tag} apparent zenith"),
        );
        close(
            pos.zenith_no_refraction_deg,
            z,
            1e-9,
            &format!("{tag} zenith"),
        );
        close(pos.azimuth_deg, az, 1e-9, &format!("{tag} azimuth"));
        close(pos.equation_of_time_min, eot, 1e-9, &format!("{tag} EoT"));
        worst = worst
            .max((pos.zenith_deg - z_app).abs())
            .max((pos.zenith_no_refraction_deg - z).abs())
            .max((pos.azimuth_deg - az).abs());
    }
    eprintln!("worst angle difference vs pvlib: {worst:e} deg");
}

#[test]
fn default_site_is_documented_values() {
    let s = SolarSite::default();
    assert_eq!(s.pressure_mbar, 1013.25);
    assert_eq!(s.temperature_c, 12.0);
    assert_eq!(s.delta_t_s, 69.0);
    assert_eq!(s.atmos_refract_deg, 0.5667);
}

#[test]
fn sun_direction_is_a_unit_vector_in_enu() {
    let (_, _, p) = spa_example();
    let v = sun_direction_enu(&p);
    close(v[0] * v[0] + v[1] * v[1] + v[2] * v[2], 1.0, 1e-15, "norm");
    close(
        v[2].asin().to_degrees(),
        90.0 - p.zenith_deg,
        1e-12,
        "elevation",
    );
    // Azimuth 194°: a little west of due south, so east < 0 and north < 0.
    assert!(v[0] < 0.0 && v[1] < 0.0 && v[2] > 0.0);
    close(
        v[0].atan2(v[1]).to_degrees().rem_euclid(360.0),
        p.azimuth_deg,
        1e-12,
        "azimuth",
    );
}

#[test]
fn sun_half_angle_follows_the_distance() {
    close(SUN_HALF_ANGLE_RAD, 4.6524e-3, 5e-8, "mean half angle");
    // Perihelion (early January) larger, aphelion (early July) smaller, by
    // about 1.7% each way.
    let jan = sun_half_angle_rad(&UtcInstant::new(2025, 1, 4, 0, 0, 0.0));
    let jul = sun_half_angle_rad(&UtcInstant::new(2025, 7, 3, 0, 0, 0.0));
    close(jan / SUN_HALF_ANGLE_RAD, 1.0 / 0.98333, 2e-4, "perihelion");
    close(jul / SUN_HALF_ANGLE_RAD, 1.0 / 1.01668, 2e-4, "aphelion");
    let (_, _, p) = spa_example();
    close(
        p.half_angle_rad(),
        SUN_HALF_ANGLE_RAD / 0.9965422974,
        1e-12,
        "example",
    );
}

// ---------------------------------------------------------------------------
// ASTM G173-03
// ---------------------------------------------------------------------------

#[test]
fn am15_direct_total() {
    // ASTM G173-03 quotes 900.1 W/m² for direct + circumsolar.
    close(AM15_DIRECT_TOTAL_W_M2, 900.1, 0.05, "published total");
    let whole = am15_direct_lines(1, 0.28, 4.0);
    close(whole[0].1, AM15_DIRECT_TOTAL_W_M2, 1e-9, "one band = total");
    close(
        am15_fraction(0.28, 4.0),
        1.0,
        1e-15,
        "fraction of everything",
    );
    // Outside the table there is nothing.
    close(am15_fraction(0.1, 5.0), 1.0, 1e-15, "saturates");
    assert_eq!(am15_fraction(0.1, 0.28), 0.0);
    assert_eq!(am15_fraction(1.0, 1.0), 0.0);
}

#[test]
fn am15_fractions_add_up() {
    let a = am15_fraction(0.28, 0.7);
    let b = am15_fraction(0.7, 1.4);
    let c = am15_fraction(1.4, 4.0);
    close(a + b + c, 1.0, 1e-12, "split");
    // A 0.3–2.5 µm sun carries almost all of the direct beam.
    let f = am15_fraction(0.3, 2.5);
    assert!(f > 0.98 && f < 1.0, "0.3–2.5 µm share {f}");
}

#[test]
fn am15_band_powers_sum_to_the_band() {
    for (lo, hi, n) in [
        (0.28, 4.0, 1),
        (0.3, 2.5, 7),
        (0.4, 0.7, 30),
        (0.3001, 1.2345, 13),
    ] {
        let want = am15_fraction(lo, hi) * AM15_DIRECT_TOTAL_W_M2;
        for lines in [
            am15_direct_lines(n, lo, hi),
            am15_direct_lines_equal_energy(n, lo, hi),
        ] {
            assert_eq!(lines.len(), n);
            let sum: f64 = lines.iter().map(|l| l.1).sum();
            assert!(
                (sum - want).abs() <= 1e-9 * want,
                "[{lo}, {hi}] x{n}: {sum} vs {want}"
            );
        }
    }
}

#[test]
fn am15_centroids_sit_inside_their_bands() {
    let (lo, hi, n) = (0.3, 2.5, 22);
    let w = (hi - lo) / n as f64;
    for (i, (c, p)) in am15_direct_lines(n, lo, hi).into_iter().enumerate() {
        let (a, b) = (lo + w * i as f64, lo + w * (i + 1) as f64);
        assert!(
            c >= a - 1e-12 && c <= b + 1e-12,
            "band {i}: {c} not in [{a}, {b}]"
        );
        assert!(p > 0.0);
    }
    // Equal-energy bands: equal powers, increasing centroids.
    let eq = am15_direct_lines_equal_energy(10, lo, hi);
    let each = eq.iter().map(|l| l.1).sum::<f64>() / 10.0;
    for w in eq.windows(2) {
        assert!(w[0].0 < w[1].0);
    }
    for (c, p) in &eq {
        assert!((p - each).abs() <= 1e-9 * each, "{p} vs {each}");
        assert!(*c > lo && *c < hi);
    }
    assert!(am15_direct_lines(0, 0.3, 2.5).is_empty());
    assert!(am15_direct_lines(3, 2.5, 0.3).is_empty());
}

/// The continuum is the table at its own resolution: its power adds up to
/// the band's, every line sits inside its interval, and the lines equal-width
/// sampling gives are the ones the G173 functions always gave.
#[test]
fn the_g173_continuum_is_the_table_itself() {
    let t = solar::g173_direct();
    let lines = solar::table_continuum(t, 0.3, 2.5);
    // 0.5 nm steps to 400 nm, 1 nm to 1700 nm, then 5 nm: 1661 intervals.
    assert_eq!(lines.len(), 1661);
    let sum: f64 = lines.iter().map(|(_, p)| p).sum();
    assert!((sum - solar::table_power(t, 0.3, 2.5)).abs() < 1e-9 * sum);
    assert!(lines.windows(2).all(|w| w[0].0 < w[1].0));
    assert!(lines.iter().all(|(l, _)| (0.3..=2.5).contains(l)));
    assert_eq!(solar::table_lines(t, 6, 0.3, 2.5), solar::am15_direct_lines(6, 0.3, 2.5));
    assert!((solar::table_fraction(t, 0.3, 2.5) - solar::am15_fraction(0.3, 2.5)).abs() < 1e-12);
}

/// A SMARTS-style file: a header naming its columns, several irradiance
/// columns, and the direct one with the circumsolar part picked.
#[test]
fn a_smarts_file_reads_its_direct_column() {
    let text = "Wvlgth Extraterrestrial_spectrm Global_tilted_irradiance Beam_normal_+circumsolar\n\
                300 0.45 0.01 0.002\n400 1.60 1.10 0.80\n500 1.95 1.50 1.30\n";
    let p = solar::parse_spectrum_table(text).unwrap();
    assert_eq!(p.rows, vec![[300.0, 0.002], [400.0, 0.80], [500.0, 1.30]]);
    assert!(p.summary.contains("beam_normal_+circumsolar"), "{}", p.summary);
    // The trapezoids between the rows.
    assert!(p.summary.contains("W/m²"));
    assert!((solar::table_power(&p.rows, 0.3, 0.5) - (0.5 * 100.0 * 0.802 + 0.5 * 100.0 * 2.1)).abs() < 1e-9);
}

/// Two bare columns in micrometres and W/m²/µm, commas, a comment.
#[test]
fn micrometres_per_micrometre_read_as_nanometres() {
    let text = "# a spectroradiometer\nwavelength_um, irradiance_W/m2/um\n0.4, 800\n0.5, 1300\n";
    let p = solar::parse_spectrum_table(text).unwrap();
    assert_eq!(p.rows, vec![[400.0, 0.8], [500.0, 1.3]]);
}

#[test]
fn a_table_that_is_not_one_is_refused() {
    assert!(solar::parse_spectrum_table("just words\n").is_err());
    assert!(solar::parse_spectrum_table("400 1\n").is_err());
    assert!(solar::parse_spectrum_table("400 1\n400 2\n").is_err());
    assert!(solar::parse_spectrum_table("400 1\n500 -2\n").is_err());
    assert!(solar::parse_spectrum_table("400 0\n500 0\n").is_err());
}
