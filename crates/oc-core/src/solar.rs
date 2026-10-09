//! The sun as a light source: where it is, how big it looks, and what it
//! emits.
//!
//! This module is pure numbers for a later heliostat feature. It knows nothing
//! about scenes or the tracer; it answers three questions a solar source needs
//! answered before a single ray is launched.
//!
//! ## Where the sun is: NREL SPA
//!
//! [`solar_position`] is the NREL Solar Position Algorithm, Reda & Andreas,
//! *Solar Position Algorithm for Solar Radiation Applications*, NREL/TP-560-34302
//! (June 2003, revised January 2008), implemented step for step: the full VSOP87
//! periodic terms for the Earth's heliocentric longitude, latitude and radius
//! vector, the 63-term nutation series, aberration, parallax for the observer's
//! elevation, and the report's atmospheric refraction model. The report claims
//! ±0.0003° over the years −2000 to 6000; the test suite reproduces its worked
//! example (Table A5.1) to the digits it prints (two rounding slips in the
//! table aside, see the tests) and pvlib-python's `spa_python` to 1e-12°.
//!
//! What is *not* exact:
//!
//! - **ΔT = TT − UT** is an input, not something the algorithm can know. It is
//!   measured, drifts by roughly a second every few years, and cannot be
//!   predicted far ahead. [`SolarSite::default`] uses 69 s, close to the value
//!   for the mid-2020s; one second of ΔT error moves the sun by about 0.004°.
//! - **UT1 − UTC** (below 0.9 s by definition of UTC) is ignored: the instant is
//!   treated as UT1. Worst case that is another 0.004°.
//! - **Refraction** is the report's standard-atmosphere formula (Sæmundsson's,
//!   scaled by pressure and temperature). It is good to a few arcminutes near the
//!   horizon and much better high up; a heliostat field does not work at the
//!   horizon anyway.
//!
//! Angles in the public API are degrees because that is how every solar table,
//! site survey and SPA report is written. The one exception is the sun's
//! angular radius, which feeds straight into a cone sampler and is radians.
//!
//! ## How big it looks
//!
//! [`SUN_HALF_ANGLE_RAD`] is the mean angular radius of the solar disc, and
//! [`SolarPosition::half_angle_rad`] the value at a given instant from the
//! Earth–Sun distance SPA already computes. They are the disc edge only;
//! circumsolar scattering (the aureole a pyrheliometer also sees) is not a
//! disc and is not modelled here.
//!
//! ## What it emits: ASTM G173-03
//!
//! The spectrum is the ASTM G173-03 reference **direct normal + circumsolar**
//! irradiance at air mass 1.5, the standard for concentrating collectors. It is
//! a tabulated standard, not a model of any particular day, and it is embedded
//! as tabulated (2002 rows, 280–4000 nm). Between rows the irradiance is taken
//! as linear, so every integral here ([`am15_fraction`], the band powers of
//! [`am15_direct_lines`]) is the exact integral of that piecewise-linear
//! interpolant, and the band powers telescope to the total.
//!
//! ## Frames
//!
//! [`sun_direction_enu`] returns East-North-Up. Mapping ENU onto a scene's axes
//! is the caller's business, and belongs with whatever decides where north is
//! in that scene.

mod astm_g173;
mod spa_tables;
pub mod sunshape;

use astm_g173::G173_DIRECT;
use spa_tables::*;

/// The sun's radius seen from 1 AU, 959.63 arcsec (Auwers' value, the one the
/// Astronomical Almanac uses).
pub const SUN_RADIUS_ARCSEC_AT_1AU: f64 = 959.63;

/// Mean angular radius of the solar disc, in radians: 959.63″ = 4.652 mrad
/// = 0.26656°. Commonly rounded to 4.65 mrad.
///
/// This is the value at exactly 1 AU, which is also the annual mean: over a
/// Kepler orbit the time average of 1/R is 1/a, and a = 1.000001 AU.
/// Over the year the Earth–Sun distance swings by ±1.7%, and the disc with it
/// (4.58 mrad in early July, 4.73 mrad in early January); use
/// [`SolarPosition::half_angle_rad`] when the date is known.
pub const SUN_HALF_ANGLE_RAD: f64 =
    SUN_RADIUS_ARCSEC_AT_1AU / 3600.0 * (std::f64::consts::PI / 180.0);

/// The sun's semi-diameter SPA uses in its refraction switch, in degrees.
const SPA_SUN_RADIUS_DEG: f64 = 0.26667;

/// An observer on the ground.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SolarSite {
    /// Geodetic latitude, degrees, north positive.
    pub latitude_deg: f64,
    /// Longitude, degrees, **east positive** (Golden, Colorado is −105.18).
    pub longitude_deg: f64,
    /// Height above the reference ellipsoid, metres. Only the parallax sees
    /// it, and only barely.
    pub elevation_m: f64,
    /// Annual mean local pressure, millibar (= hPa). Refraction only.
    pub pressure_mbar: f64,
    /// Annual mean local temperature, °C. Refraction only.
    pub temperature_c: f64,
    /// ΔT = TT − UT, seconds. See the module docs: it is measured, not
    /// computed, and drifts.
    pub delta_t_s: f64,
    /// Atmospheric refraction at the horizon, degrees. SPA's default 0.5667°;
    /// it only sets the depth below the horizon at which refraction switches
    /// off.
    pub atmos_refract_deg: f64,
}

impl Default for SolarSite {
    /// Sea level on the equator at the Greenwich meridian, in the standard
    /// atmosphere (1013.25 mbar), 12 °C, ΔT = 69 s (the mid-2020s value; it
    /// was 63.8 s in 2000 and 66.1 s in 2010, and future values are forecasts).
    fn default() -> Self {
        Self {
            latitude_deg: 0.0,
            longitude_deg: 0.0,
            elevation_m: 0.0,
            pressure_mbar: 1013.25,
            temperature_c: 12.0,
            delta_t_s: 69.0,
            atmos_refract_deg: 0.5667,
        }
    }
}

/// A calendar instant in UTC (treated as UT1, see the module docs), Gregorian
/// calendar from 15 October 1582 and Julian before, as SPA does.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct UtcInstant {
    pub year: i32,
    /// 1..=12
    pub month: u32,
    /// 1..=31
    pub day: u32,
    /// 0..=23
    pub hour: u32,
    /// 0..=59
    pub minute: u32,
    /// Seconds, fractional allowed, `[0, 60)`.
    pub second: f64,
}

impl UtcInstant {
    pub fn new(year: i32, month: u32, day: u32, hour: u32, minute: u32, second: f64) -> Self {
        Self {
            year,
            month,
            day,
            hour,
            minute,
            second,
        }
    }

    /// Convert a local clock reading to UTC. `utc_offset_hours` is the zone's
    /// offset *from* UTC, the way it is written: −7 for Mountain Standard Time,
    /// +5.5 for India. The date rolls over as needed.
    ///
    /// The calendar arithmetic is exact (proleptic Gregorian day numbers, no
    /// Julian-day round trip), so `12:30:30` at −7 is exactly `19:30:30`. The
    /// offset is rounded to a whole second, which every real zone already is;
    /// the fractional part of `second` passes through untouched.
    pub fn from_local(
        year: i32,
        month: u32,
        day: u32,
        hour: u32,
        minute: u32,
        second: f64,
        utc_offset_hours: f64,
    ) -> Self {
        let whole = second.floor();
        let frac = second - whole;
        let offset_s = (utc_offset_hours * 3600.0).round() as i64;
        let secs = (hour as i64) * 3600 + (minute as i64) * 60 + whole as i64 - offset_s;
        let days = days_from_civil(year as i64, month, day) + secs.div_euclid(86_400);
        let sod = secs.rem_euclid(86_400);
        let (y, m, d) = civil_from_days(days);
        Self {
            year: y as i32,
            month: m,
            day: d,
            hour: (sod / 3600) as u32,
            minute: ((sod % 3600) / 60) as u32,
            second: (sod % 60) as f64 + frac,
        }
    }

    /// Julian Day (UT), SPA eq. 4, including the switch to the Gregorian
    /// calendar at JD 2299160.
    pub fn julian_day(&self) -> f64 {
        let (mut y, mut m) = (self.year as f64, self.month as f64);
        if m < 3.0 {
            y -= 1.0;
            m += 12.0;
        }
        let d = self.day as f64
            + (self.hour as f64 + (self.minute as f64 + self.second / 60.0) / 60.0) / 24.0;
        let mut jd = (365.25 * (y + 4716.0)).floor() + (30.6001 * (m + 1.0)).floor() + d - 1524.5;
        if jd > 2_299_160.0 {
            let a = (y / 100.0).floor();
            jd += 2.0 - a + (a / 4.0).floor();
        }
        jd
    }
}

/// Days since 1970-01-01 in the proleptic Gregorian calendar (Hinnant's
/// algorithm). Only used to roll local dates over to UTC.
fn days_from_civil(y: i64, m: u32, d: u32) -> i64 {
    let y = if m <= 2 { y - 1 } else { y };
    let era = y.div_euclid(400);
    let yoe = y - era * 400;
    let m = m as i64;
    let doy = (153 * (if m > 2 { m - 3 } else { m + 9 }) + 2) / 5 + d as i64 - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    era * 146_097 + doe - 719_468
}

fn civil_from_days(z: i64) -> (i64, u32, u32) {
    let z = z + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = (doy - (153 * mp + 2) / 5 + 1) as u32;
    let m = if mp < 10 { mp + 3 } else { mp - 9 } as u32;
    let y = yoe + era * 400 + if m <= 2 { 1 } else { 0 };
    (y, m, d)
}

/// Everything SPA computes on the way to the answer. The intermediates are
/// public because the report tabulates them and the tests check them; a caller
/// usually wants [`zenith_deg`](Self::zenith_deg), [`azimuth_deg`](Self::azimuth_deg)
/// and [`sun_direction_enu`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SolarPosition {
    /// Julian Day (UT).
    pub julian_day: f64,
    /// Julian Ephemeris Day (TT): `julian_day + ΔT/86400`.
    pub julian_ephemeris_day: f64,
    /// Earth heliocentric longitude L, degrees, `[0, 360)`.
    pub heliocentric_longitude_deg: f64,
    /// Earth heliocentric latitude B, degrees.
    pub heliocentric_latitude_deg: f64,
    /// Earth–Sun distance R, astronomical units.
    pub earth_sun_distance_au: f64,
    /// Nutation in longitude Δψ, degrees.
    pub nutation_longitude_deg: f64,
    /// Nutation in obliquity Δε, degrees.
    pub nutation_obliquity_deg: f64,
    /// True obliquity of the ecliptic ε, degrees.
    pub true_obliquity_deg: f64,
    /// Apparent sun longitude λ, degrees.
    pub apparent_sun_longitude_deg: f64,
    /// Geocentric right ascension α, degrees, `[0, 360)`.
    pub right_ascension_deg: f64,
    /// Geocentric declination δ, degrees.
    pub declination_deg: f64,
    /// Observer local hour angle H, degrees, `[0, 360)`, westward from south.
    pub hour_angle_deg: f64,
    /// Topocentric right ascension α′, degrees.
    pub topocentric_right_ascension_deg: f64,
    /// Topocentric declination δ′, degrees.
    pub topocentric_declination_deg: f64,
    /// Topocentric local hour angle H′, degrees.
    pub topocentric_hour_angle_deg: f64,
    /// Topocentric elevation without refraction e₀, degrees.
    pub elevation_no_refraction_deg: f64,
    /// Apparent topocentric elevation e (with refraction), degrees.
    pub elevation_deg: f64,
    /// Topocentric zenith angle without refraction, `90 − e₀`, degrees.
    pub zenith_no_refraction_deg: f64,
    /// Apparent topocentric zenith angle θ (with refraction), `90 − e`,
    /// degrees. This is where the sun *looks* to be, which is where a mirror
    /// has to be aimed.
    pub zenith_deg: f64,
    /// Topocentric azimuth Φ, degrees clockwise from north, `[0, 360)`.
    /// Refraction does not change it.
    pub azimuth_deg: f64,
    /// Topocentric astronomers' azimuth Γ, degrees westward from south,
    /// `[0, 360)`. SPA's surface-incidence formula is written in it.
    pub astronomers_azimuth_deg: f64,
    /// Sun mean longitude M, degrees, `[0, 360)`.
    pub sun_mean_longitude_deg: f64,
    /// Equation of time, minutes: apparent minus mean solar time.
    pub equation_of_time_min: f64,
}

impl SolarPosition {
    /// Angular radius of the solar disc at this instant, radians:
    /// 959.63″ / R.
    pub fn half_angle_rad(&self) -> f64 {
        SUN_HALF_ANGLE_RAD / self.earth_sun_distance_au
    }

    /// Incidence angle on a tilted plane, degrees (SPA eq. 47). `slope_deg` is
    /// the tilt from horizontal; `surface_azimuth_rotation_deg` is SPA's γ,
    /// measured from **south**, westward positive (so a surface facing
    /// south-east has −45). Uses the apparent (refracted) zenith, as SPA does.
    pub fn incidence_angle_deg(&self, slope_deg: f64, surface_azimuth_rotation_deg: f64) -> f64 {
        let th = self.zenith_deg.to_radians();
        let w = slope_deg.to_radians();
        let g = (self.astronomers_azimuth_deg - surface_azimuth_rotation_deg).to_radians();
        (th.cos() * w.cos() + w.sin() * th.sin() * g.cos())
            .clamp(-1.0, 1.0)
            .acos()
            .to_degrees()
    }
}

/// `Σ A cos(B + C·x)` over one table.
fn periodic_sum(rows: &[[f64; 3]], x: f64) -> f64 {
    rows.iter().map(|r| r[0] * (r[1] + r[2] * x).cos()).sum()
}

/// `(t0 + t1 x + t2 x² + ...) / 1e8` over a family of tables (SPA eqs. 9–11).
fn vsop_series(tables: &[&[[f64; 3]]], x: f64) -> f64 {
    let mut acc = 0.0;
    let mut p = 1.0;
    for t in tables {
        acc += periodic_sum(t, x) * p;
        p *= x;
    }
    acc / 1e8
}

fn limit_degrees(d: f64) -> f64 {
    d.rem_euclid(360.0)
}

/// The sun as seen from `site` at `instant`: SPA, full precision.
///
/// Refraction is added only while the unrefracted sun is above
/// `−(0.26667° + atmos_refract_deg)`, i.e. while any of the disc could still be
/// lifted over the horizon, exactly as SPA does; below that the apparent and
/// geometric positions coincide.
pub fn solar_position(site: &SolarSite, instant: &UtcInstant) -> SolarPosition {
    // 3.1 Julian day, ephemeris day, centuries, millennium.
    let jd = instant.julian_day();
    let jde = jd + site.delta_t_s / 86_400.0;
    let jc = (jd - 2_451_545.0) / 36_525.0;
    let jce = (jde - 2_451_545.0) / 36_525.0;
    let jme = jce / 10.0;

    // 3.2 Earth heliocentric longitude, latitude, radius vector.
    let l_deg = limit_degrees(vsop_series(&[&L0, &L1, &L2, &L3, &L4, &L5], jme).to_degrees());
    let b_deg = vsop_series(&[&B0, &B1], jme).to_degrees();
    let r_au = vsop_series(&[&R0, &R1, &R2, &R3, &R4], jme);

    // 3.3 Geocentric longitude and latitude.
    let theta = limit_degrees(l_deg + 180.0);
    let beta = -b_deg;

    // 3.4 Nutation.
    let x = [
        297.85036 + 445_267.111480 * jce - 0.0019142 * jce * jce + jce.powi(3) / 189_474.0,
        357.52772 + 35_999.050340 * jce - 0.0001603 * jce * jce - jce.powi(3) / 300_000.0,
        134.96298 + 477_198.867398 * jce + 0.0086972 * jce * jce + jce.powi(3) / 56_250.0,
        93.27191 + 483_202.017538 * jce - 0.0036825 * jce * jce + jce.powi(3) / 327_270.0,
        125.04452 - 1934.136261 * jce + 0.0020708 * jce * jce + jce.powi(3) / 450_000.0,
    ];
    let (mut dpsi, mut deps) = (0.0, 0.0);
    for (y, [a, b, c, d]) in NUTATION.iter() {
        let arg = (0..5).map(|i| x[i] * y[i]).sum::<f64>().to_radians();
        dpsi += (a + b * jce) * arg.sin();
        deps += (c + d * jce) * arg.cos();
    }
    let dpsi = dpsi / 36_000_000.0;
    let deps = deps / 36_000_000.0;

    // 3.5 True obliquity of the ecliptic.
    let u = jme / 10.0;
    let eps0 = [
        84_381.448, -4680.93, -1.55, 1999.25, -51.38, -249.67, -39.05, 7.12, 27.87, 5.79, 2.45,
    ]
    .iter()
    .rev()
    .fold(0.0, |acc, c| acc * u + c);
    let eps = eps0 / 3600.0 + deps;

    // 3.6–3.7 Aberration, apparent sun longitude.
    let dtau = -20.4898 / (3600.0 * r_au);
    let lambda = theta + dpsi + dtau;

    // 3.8 Apparent sidereal time at Greenwich.
    let nu0 = limit_degrees(
        280.46061837 + 360.98564736629 * (jd - 2_451_545.0) + 0.000387933 * jc * jc
            - jc.powi(3) / 38_710_000.0,
    );
    let eps_r = eps.to_radians();
    let nu = nu0 + dpsi * eps_r.cos();

    // 3.9–3.10 Geocentric right ascension and declination.
    let lam_r = lambda.to_radians();
    let beta_r = beta.to_radians();
    let alpha = limit_degrees(
        (lam_r.sin() * eps_r.cos() - beta_r.tan() * eps_r.sin())
            .atan2(lam_r.cos())
            .to_degrees(),
    );
    let delta = (beta_r.sin() * eps_r.cos() + beta_r.cos() * eps_r.sin() * lam_r.sin())
        .asin()
        .to_degrees();

    // 3.11 Observer local hour angle.
    let h = limit_degrees(nu + site.longitude_deg - alpha);

    // 3.12 Topocentric right ascension (parallax).
    let xi = (8.794 / (3600.0 * r_au)).to_radians();
    let phi = site.latitude_deg.to_radians();
    let uu = (0.99664719 * phi.tan()).atan();
    let xx = uu.cos() + site.elevation_m / 6_378_140.0 * phi.cos();
    let yy = 0.99664719 * uu.sin() + site.elevation_m / 6_378_140.0 * phi.sin();
    let h_r = h.to_radians();
    let delta_r = delta.to_radians();
    let dalpha_r = (-xx * xi.sin() * h_r.sin()).atan2(delta_r.cos() - xx * xi.sin() * h_r.cos());
    let dalpha = dalpha_r.to_degrees();
    let alpha_p = alpha + dalpha;
    let delta_p_r = ((delta_r.sin() - yy * xi.sin()) * dalpha_r.cos())
        .atan2(delta_r.cos() - xx * xi.sin() * h_r.cos());

    // 3.13 Topocentric local hour angle.
    let h_p = h - dalpha;
    let h_p_r = h_p.to_radians();

    // 3.14 Topocentric zenith angle.
    let e0 = (phi.sin() * delta_p_r.sin() + phi.cos() * delta_p_r.cos() * h_p_r.cos())
        .asin()
        .to_degrees();
    let de = if e0 >= -(SPA_SUN_RADIUS_DEG + site.atmos_refract_deg) {
        (site.pressure_mbar / 1010.0) * (283.0 / (273.0 + site.temperature_c)) * 1.02
            / (60.0 * (e0 + 10.3 / (e0 + 5.11)).to_radians().tan())
    } else {
        0.0
    };
    let e = e0 + de;

    // 3.15 Topocentric azimuth.
    let gamma = limit_degrees(
        h_p_r
            .sin()
            .atan2(h_p_r.cos() * phi.sin() - delta_p_r.tan() * phi.cos())
            .to_degrees(),
    );
    let azimuth = limit_degrees(gamma + 180.0);

    // A.1 Equation of time.
    let m = limit_degrees(
        280.4664567 + 360_007.6982779 * jme + 0.03032028 * jme * jme + jme.powi(3) / 49_931.0
            - jme.powi(4) / 15_300.0
            - jme.powi(5) / 2_000_000.0,
    );
    let eot = 4.0 * (m - 0.0057183 - alpha + dpsi * eps_r.cos());
    // SPA limits E to ±20 minutes by adding or removing a whole day; the
    // wrap below is the same for any real value.
    let mut eot = eot.rem_euclid(1440.0);
    if eot > 720.0 {
        eot -= 1440.0;
    }

    SolarPosition {
        julian_day: jd,
        julian_ephemeris_day: jde,
        heliocentric_longitude_deg: l_deg,
        heliocentric_latitude_deg: b_deg,
        earth_sun_distance_au: r_au,
        nutation_longitude_deg: dpsi,
        nutation_obliquity_deg: deps,
        true_obliquity_deg: eps,
        apparent_sun_longitude_deg: lambda,
        right_ascension_deg: alpha,
        declination_deg: delta,
        hour_angle_deg: h,
        topocentric_right_ascension_deg: alpha_p,
        topocentric_declination_deg: delta_p_r.to_degrees(),
        topocentric_hour_angle_deg: h_p,
        elevation_no_refraction_deg: e0,
        elevation_deg: e,
        zenith_no_refraction_deg: 90.0 - e0,
        zenith_deg: 90.0 - e,
        azimuth_deg: azimuth,
        astronomers_azimuth_deg: gamma,
        sun_mean_longitude_deg: m,
        equation_of_time_min: eot,
    }
}

/// Unit vector from the site **toward** the sun, East-North-Up, using the
/// apparent (refracted) elevation, because that is the direction light
/// actually arrives from. The direction the light *travels* is its negative.
pub fn sun_direction_enu(pos: &SolarPosition) -> [f64; 3] {
    let az = pos.azimuth_deg.to_radians();
    let el = pos.elevation_deg.to_radians();
    [az.sin() * el.cos(), az.cos() * el.cos(), el.sin()]
}

/// Angular radius of the solar disc at `instant`, radians. Only the
/// Earth–Sun distance matters, so the site is irrelevant and ΔT is taken
/// from [`SolarSite::default`] (a minute of ΔT error changes R by 1e-8).
pub fn sun_half_angle_rad(instant: &UtcInstant) -> f64 {
    solar_position(&SolarSite::default(), instant).half_angle_rad()
}

// ---------------------------------------------------------------------------
// ASTM G173-03 direct + circumsolar
// ---------------------------------------------------------------------------

/// First and last tabulated wavelengths of ASTM G173-03, micrometres.
pub const AM15_LAMBDA_MIN_UM: f64 = 0.28;
pub const AM15_LAMBDA_MAX_UM: f64 = 4.0;

/// Direct normal + circumsolar irradiance of ASTM G173-03 integrated over the
/// whole table (280–4000 nm) by the trapezoid rule on the tabulated rows,
/// W/m². The standard quotes 900.1 W/m²; the test suite recomputes this from
/// the embedded table.
pub const AM15_DIRECT_TOTAL_W_M2: f64 = 900.139_329_284_214_9;

/// The ASTM G173-03 direct + circumsolar table, rows `[nm, W/m²/nm]`.
pub fn g173_direct() -> &'static [[f64; 2]] {
    &G173_DIRECT
}

fn g173_cumulative(lambda_nm: f64) -> (f64, f64) {
    table_cumulative(&G173_DIRECT, lambda_nm)
}

/// Integrals of a piecewise-linear spectrum `table` (rows `[nm, W/m²/nm]`,
/// wavelength increasing) from its first row to `lambda_nm`: `(∫E dλ,
/// ∫λ E dλ)` in W/m² and W/m²·nm. Zero below the table; the spectrum is zero
/// outside it.
pub fn table_cumulative(table: &[[f64; 2]], lambda_nm: f64) -> (f64, f64) {
    let (mut e, mut le) = (0.0, 0.0);
    for w in table.windows(2) {
        let ([x0, y0], [x1, y1]) = (w[0], w[1]);
        if lambda_nm <= x0 {
            break;
        }
        let b = lambda_nm.min(x1);
        let yb = if b >= x1 {
            y1
        } else {
            y0 + (y1 - y0) * (b - x0) / (x1 - x0)
        };
        let (a, ya) = (x0, y0);
        e += 0.5 * (b - a) * (ya + yb);
        le += (b - a) / 6.0 * (a * (2.0 * ya + yb) + b * (ya + 2.0 * yb));
        if b < x1 {
            break;
        }
    }
    (e, le)
}

/// Fraction of the ASTM G173-03 direct + circumsolar irradiance between
/// `lo_um` and `hi_um` (micrometres). Wavelengths outside 0.28–4.0 µm carry
/// nothing, so `am15_fraction(0.28, 4.0) == 1` and wider bands saturate.
/// Returns 0 for an empty or inverted band.
pub fn am15_fraction(lo_um: f64, hi_um: f64) -> f64 {
    if hi_um <= lo_um {
        return 0.0;
    }
    let (lo, _) = g173_cumulative(lo_um * 1000.0);
    let (hi, _) = g173_cumulative(hi_um * 1000.0);
    let (tot, _) = g173_cumulative(AM15_LAMBDA_MAX_UM * 1000.0);
    (hi - lo) / tot
}

/// Power of `table` between `lo_um` and `hi_um`, W/m².
pub fn table_power(table: &[[f64; 2]], lo_um: f64, hi_um: f64) -> f64 {
    if hi_um <= lo_um {
        return 0.0;
    }
    table_cumulative(table, hi_um * 1000.0).0 - table_cumulative(table, lo_um * 1000.0).0
}

/// Share of all of `table`'s power that lies between `lo_um` and `hi_um`.
pub fn table_fraction(table: &[[f64; 2]], lo_um: f64, hi_um: f64) -> f64 {
    let (Some(first), Some(last)) = (table.first(), table.last()) else {
        return 0.0;
    };
    let all = table_power(table, first[0] / 1000.0, last[0] / 1000.0);
    if all > 0.0 {
        table_power(table, lo_um, hi_um) / all
    } else {
        0.0
    }
}

/// `table` between `lo_um` and `hi_um` at its own resolution: one line per
/// tabulated interval (clipped to the band), at the interval's power-weighted
/// wavelength and carrying its power, W/m². What a continuous source draws a
/// wavelength per ray from; ASTM G173 gives 1661 of them over 0.3–2.5 µm,
/// half a nanometre apart in the ultraviolet and a nanometre in the visible.
pub fn table_continuum(table: &[[f64; 2]], lo_um: f64, hi_um: f64) -> Vec<(f64, f64)> {
    let (lo, hi) = (lo_um * 1000.0, hi_um * 1000.0);
    // Each interval integrated on its own segment: the same numbers as
    // differences of `table_cumulative`, in one pass rather than one pass per
    // interval, which a solar atlas of 70 000 rows cannot afford.
    table
        .windows(2)
        .filter_map(|w| {
            let ([x0, y0], [x1, y1]) = (w[0], w[1]);
            let (a, b) = (x0.max(lo), x1.min(hi));
            if b <= a {
                return None;
            }
            let at = |x: f64| y0 + (y1 - y0) * (x - x0) / (x1 - x0);
            let (ya, yb) = (at(a), at(b));
            let p = 0.5 * (b - a) * (ya + yb);
            let le = (b - a) / 6.0 * (a * (2.0 * ya + yb) + b * (ya + 2.0 * yb));
            let c = if p > 0.0 { (le / p).clamp(a, b) } else { 0.5 * (a + b) };
            Some((c / 1000.0, p))
        })
        .filter(|(_, p)| *p > 0.0)
        .collect()
}

/// [`am15_direct_lines`] for any table.
pub fn table_lines(table: &[[f64; 2]], n_bands: usize, lo_um: f64, hi_um: f64) -> Vec<(f64, f64)> {
    if n_bands == 0 || hi_um <= lo_um {
        return Vec::new();
    }
    let (lo, hi) = (lo_um * 1000.0, hi_um * 1000.0);
    let w = (hi - lo) / n_bands as f64;
    (0..n_bands)
        .map(|i| {
            let a = lo + w * i as f64;
            let b = if i + 1 == n_bands { hi } else { a + w };
            table_band_line(table, a, b)
        })
        .collect()
}

fn table_band_line(table: &[[f64; 2]], a_nm: f64, b_nm: f64) -> (f64, f64) {
    let (e0, l0) = table_cumulative(table, a_nm);
    let (e1, l1) = table_cumulative(table, b_nm);
    let p = e1 - e0;
    let c = if p > 0.0 {
        ((l1 - l0) / p).clamp(a_nm, b_nm)
    } else {
        0.5 * (a_nm + b_nm)
    };
    (c / 1000.0, p)
}

/// A spectrum read from text: its rows `[nm, W/m²/nm]` and a line saying what
/// was read.
pub struct ParsedSpectrum {
    pub rows: Vec<[f64; 2]>,
    pub summary: String,
}

/// Read a spectral irradiance table from delimited text: a SMARTS output file,
/// a spectroradiometer export, or two columns typed by hand.
///
/// Commas, semicolons, tabs and spaces all separate fields, and lines starting
/// with `#`, `;` or `//` are comments. A header row, if any, picks the columns:
/// the wavelength is the first column whose name mentions a wavelength
/// (`wvl`, `wave`, `lambda`, `nm`, `µm`), else the first; the irradiance is
/// the column naming direct light with the circumsolar part (`circumsolar`),
/// else any direct or beam column (`direct`, `beam`, `dni`), else the second.
/// Wavelengths all below 30 are taken as micrometres; an irradiance column
/// whose name says per micrometre is divided by 1000 into per nanometre.
pub fn parse_spectrum_table(text: &str) -> Result<ParsedSpectrum, String> {
    let split = |line: &str| -> Vec<String> {
        line.split(|c: char| c == ',' || c == ';' || c == '\t' || c.is_whitespace())
            .filter(|t| !t.is_empty())
            .map(str::to_string)
            .collect()
    };
    let mut header: Option<Vec<String>> = None;
    let mut rows: Vec<Vec<f64>> = Vec::new();
    for line in text.lines() {
        let t = line.trim();
        if t.is_empty() || t.starts_with('#') || t.starts_with(';') || t.starts_with("//") {
            continue;
        }
        let fields = split(t);
        let nums: Vec<Option<f64>> = fields.iter().map(|f| f.parse::<f64>().ok()).collect();
        if nums.len() >= 2 && nums.iter().all(Option::is_some) {
            rows.push(nums.into_iter().flatten().collect());
        } else if rows.is_empty() {
            header = Some(fields.iter().map(|f| f.to_lowercase()).collect());
        }
    }
    if rows.len() < 2 {
        return Err("fewer than two rows of numbers".into());
    }
    let width = rows.iter().map(Vec::len).min().unwrap_or(0);
    let find = |keys: &[&str]| -> Option<usize> {
        header
            .as_ref()?
            .iter()
            .position(|h| keys.iter().any(|k| h.contains(k)))
            .filter(|i| *i < width)
    };
    let wl = find(&["wvl", "wave", "lambda", "nm", "µm", "um"]).unwrap_or(0);
    let ir = find(&["circumsolar"])
        .or_else(|| find(&["direct", "beam", "dni"]))
        .filter(|i| *i != wl)
        .unwrap_or(if wl == 0 { 1 } else { 0 });
    let name = |i: usize| {
        header
            .as_ref()
            .and_then(|h| h.get(i).cloned())
            .unwrap_or_else(|| format!("column {}", i + 1))
    };
    let micrometres = rows.iter().all(|r| r[wl] < 30.0);
    let per_um = header
        .as_ref()
        .and_then(|h| h.get(ir))
        // Per micrometre, said as a unit: "/um", "um-1", "per um". A bare "um"
        // would match "circumsolar".
        .is_some_and(|h| {
            ["/um", "/µm", "um-1", "µm-1", "um^-1", "per_um", "per um", "per µm"]
                .iter()
                .any(|k| h.contains(k))
        });
    let mut out: Vec<[f64; 2]> = rows
        .iter()
        .map(|r| {
            let nm = if micrometres { r[wl] * 1000.0 } else { r[wl] };
            let e = if per_um { r[ir] / 1000.0 } else { r[ir] };
            [nm, e]
        })
        .collect();
    out.sort_by(|a, b| a[0].total_cmp(&b[0]));
    if out.iter().any(|r| !r[0].is_finite() || !r[1].is_finite() || r[0] <= 0.0) {
        return Err("a wavelength or an irradiance is not a finite number".into());
    }
    if out.iter().any(|r| r[1] < 0.0) {
        return Err("an irradiance is negative".into());
    }
    if out.windows(2).any(|w| w[1][0] <= w[0][0]) {
        return Err("two rows share a wavelength".into());
    }
    let total = table_cumulative(&out, out[out.len() - 1][0]).0;
    if total <= 0.0 {
        return Err("the table carries no power".into());
    }
    let summary = format!(
        "{} rows, {:.0}–{:.0} nm{}, irradiance from '{}'{}: {:.1} W/m²",
        out.len(),
        out[0][0],
        out[out.len() - 1][0],
        if micrometres { " (read as µm)" } else { "" },
        name(ir),
        if per_um { " per µm" } else { "" },
        total
    );
    Ok(ParsedSpectrum { rows: out, summary })
}

/// One band `[a, b]` (nm) as a line: power-weighted centroid in µm, and power.
/// A band with no power (below 280 nm, say) is placed at its midpoint.
fn band_line(a_nm: f64, b_nm: f64) -> (f64, f64) {
    let (e0, l0) = g173_cumulative(a_nm);
    let (e1, l1) = g173_cumulative(b_nm);
    let p = e1 - e0;
    let c = if p > 0.0 {
        ((l1 - l0) / p).clamp(a_nm, b_nm)
    } else {
        0.5 * (a_nm + b_nm)
    };
    (c / 1000.0, p)
}

/// Split `[lo_um, hi_um]` into `n_bands` **equal-width** bands and represent
/// each by one line: `(centroid λ in µm, band power in W/m²)`.
///
/// The centroid is the power-weighted mean wavelength of the band, and the
/// weight is the band's integrated direct + circumsolar irradiance, so the
/// weights sum to the irradiance inside `[lo, hi]` and the result drops
/// straight into `Spectrum::Lines`. Equal width is the default because it
/// samples every part of the band, including the weak infrared tail and the
/// water-absorption gaps, the way a dispersive element sees them; for fewer
/// wasted rays per watt use [`am15_direct_lines_equal_energy`].
///
/// Returns an empty vector for `n_bands == 0` or `hi_um <= lo_um`.
pub fn am15_direct_lines(n_bands: usize, lo_um: f64, hi_um: f64) -> Vec<(f64, f64)> {
    if n_bands == 0 || hi_um <= lo_um {
        return Vec::new();
    }
    let (lo, hi) = (lo_um * 1000.0, hi_um * 1000.0);
    let w = (hi - lo) / n_bands as f64;
    (0..n_bands)
        .map(|i| {
            let a = lo + w * i as f64;
            let b = if i + 1 == n_bands { hi } else { a + w };
            band_line(a, b)
        })
        .collect()
}

/// As [`am15_direct_lines`], but the band edges are chosen so every band
/// carries the same power (to about 1e-10 relative, by bisection on the cumulative
/// integral). The lines crowd into the visible and near infrared where the
/// sun is bright, so each traced ray carries the same weight.
pub fn am15_direct_lines_equal_energy(n_bands: usize, lo_um: f64, hi_um: f64) -> Vec<(f64, f64)> {
    if n_bands == 0 || hi_um <= lo_um {
        return Vec::new();
    }
    let (lo, hi) = (lo_um * 1000.0, hi_um * 1000.0);
    let e_lo = g173_cumulative(lo).0;
    let e_hi = g173_cumulative(hi).0;
    let mut edges = Vec::with_capacity(n_bands + 1);
    edges.push(lo);
    for i in 1..n_bands {
        let target = e_lo + (e_hi - e_lo) * i as f64 / n_bands as f64;
        let (mut a, mut b) = (lo, hi);
        for _ in 0..200 {
            let m = 0.5 * (a + b);
            if g173_cumulative(m).0 < target {
                a = m;
            } else {
                b = m;
            }
            if b - a <= 1e-12 * hi {
                break;
            }
        }
        edges.push(0.5 * (a + b));
    }
    edges.push(hi);
    edges.windows(2).map(|w| band_line(w[0], w[1])).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn civil_day_numbers_round_trip() {
        for z in [-800_000_i64, -1, 0, 1, 11_016, 20_000, 100_000] {
            let (y, m, d) = civil_from_days(z);
            assert_eq!(days_from_civil(y, m, d), z);
        }
        assert_eq!(days_from_civil(1970, 1, 1), 0);
        assert_eq!(civil_from_days(days_from_civil(2000, 2, 29)), (2000, 2, 29));
    }

    #[test]
    fn cumulative_integral_is_the_trapezoid_sum() {
        let trap: f64 = G173_DIRECT
            .windows(2)
            .map(|w| 0.5 * (w[1][0] - w[0][0]) * (w[0][1] + w[1][1]))
            .sum();
        let (cum, _) = g173_cumulative(4000.0);
        assert!((cum - trap).abs() <= 1e-12 * trap);
    }
}
