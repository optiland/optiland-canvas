//! Measured and supplied coating data with an angle axis (coating-fidelity
//! spec CF-09, the earlier spec's M4a).
//!
//! A vendor curve is a power table measured at some angle, for some
//! polarization, from some side, of an interface or of a whole part. None of
//! that can be guessed from the numbers, so the table carries it:
//!
//! - the axes: wavelength in µm and angle of incidence in degrees, both
//!   strictly increasing; a table measured at one angle has one;
//! - the powers, at the resolution the data have: `Sp` (R and T for s and p
//!   apart), `Unpolarized` (the mean of the two, measured with unpolarized
//!   light) or `Scalar` (one number, polarization not stated);
//! - no phase: a power table says nothing about the phase between s and p,
//!   and the tracer takes it as zero and says so;
//! - the scope: one `Interface`, or a `FiniteAssembly` that already includes
//!   the substrate and its rear face, which cannot then sit on a body face
//!   (that would count the glass twice);
//! - the side the light came from, and the media either side;
//! - where the data came from, and what to do outside it.
//!
//! Lookup is bilinear in `(λ, cos θ)`. Outside the data the table refuses with
//! a sentence that says why ("Data supplied at 45 degrees only; this ray is at
//! 52 degrees") under `numerics: Strict` or when the table itself says
//! `Refuse`; otherwise it clamps to the nearest edge and the result is flagged.
//! Values are stored row-major by wavelength: entry `i * aoi_deg.len() + j` is
//! wavelength `i` at angle `j`.

use serde::{Deserialize, Serialize};

use crate::fidelity::Numerics;
use crate::thinfilm::{Coating, FilmContext, FilmError, Provenance, SourceKind};

/// How finely the powers are resolved in polarization.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PowerResolution {
    /// R and T for s and p apart: `rs`, `rp`, `ts`, `tp`.
    Sp,
    /// Measured with unpolarized light: `r` and `t` are the means of s and p.
    Unpolarized,
    /// One number per point, the polarization not stated: `r` and `t`.
    Scalar,
}

/// What the data say about phase. A power table says nothing.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum PhaseInformation {
    #[default]
    Unknown,
}

/// What the measurement covers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TableScope {
    /// One coated interface.
    Interface,
    /// A whole part: the substrate, its bulk loss and its rear face included.
    FiniteAssembly,
}

/// The side the light came from: the coated `Front`, the `Back`, or `Either`
/// when the data are declared to hold from both (a lossless stack reflects and
/// transmits the same from either side).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TableSide {
    Front,
    Back,
    Either,
}

/// What to do with a ray outside the data.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum OutOfDomain {
    /// Refuse it, with the reason, whatever the profile says.
    Refuse,
    /// Read the nearest edge and flag the result, unless the profile is
    /// `Strict`.
    Clamp,
}

/// Is what the surface reflects a path, or stray light (an AR's residual,
/// which is a ghost for the same reason bare glass's reflection is)?
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum ReflectionRole {
    #[default]
    Path,
    Stray,
}

/// The media the data were measured between: the incident side, and the side
/// the transmitted light leaves into.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct TableMedia {
    pub incident: f64,
    pub exit: f64,
}

/// How far the media a table is used between may be from those it was
/// measured between. A glass's own dispersion moves its index by about 0.02
/// over the visible, which is not what this is for: it catches a table
/// measured in air used in water, or measured from the glass used from air.
pub const MEDIA_TOLERANCE: f64 = 0.05;

/// How far past the first or last angle a ray may be and still read the edge
/// without a flag, degrees. A one-angle table holds within this of its angle.
pub const ANGLE_TOLERANCE_DEG: f64 = 0.5;

/// A table of R and T over wavelength and angle of incidence, with what the
/// numbers mean.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AngularTable {
    pub lambda_um: Vec<f64>,
    pub aoi_deg: Vec<f64>,
    pub power_resolution: PowerResolution,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rs: Option<Vec<f64>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rp: Option<Vec<f64>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ts: Option<Vec<f64>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tp: Option<Vec<f64>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub r: Option<Vec<f64>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub t: Option<Vec<f64>>,
    #[serde(default)]
    pub phase_information: PhaseInformation,
    pub scope: TableScope,
    pub side: TableSide,
    pub media: TableMedia,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub provenance: Option<Provenance>,
    /// `None` follows the profile: `Strict` refuses, `Permissive` clamps.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub out_of_domain: Option<OutOfDomain>,
    #[serde(default)]
    pub reflection: ReflectionRole,
}

/// R and T for s and p at one point. A table without s and p apart gives the
/// same number to both, which is what an unpolarized or scalar measurement
/// can say.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TablePowers {
    pub rs: f64,
    pub rp: f64,
    pub ts: f64,
    pub tp: f64,
}

impl TablePowers {
    /// The unpolarized reflectance: the mean of the s and p **powers**.
    pub fn r(&self) -> f64 {
        0.5 * (self.rs + self.rp)
    }
    /// The unpolarized transmittance.
    pub fn t(&self) -> f64 {
        0.5 * (self.ts + self.tp)
    }
}

/// A lookup: the powers, and whether reading them took a clamp.
#[derive(Debug, Clone, PartialEq)]
pub struct TableLookup {
    pub powers: TablePowers,
    /// Why the data were read at their edge, when they were.
    pub clamped: Option<String>,
}

/// Format a number of degrees or nanometres the way a person writes it: whole
/// when it is, one decimal otherwise.
fn num(v: f64) -> String {
    if (v - v.round()).abs() < 0.05 {
        format!("{}", v.round())
    } else {
        format!("{v:.1}")
    }
}

impl AngularTable {
    /// A one-angle, unpolarized table.
    pub fn one_angle(lambda_um: Vec<f64>, aoi_deg: f64, r: Vec<f64>, t: Vec<f64>, media: TableMedia) -> Self {
        Self {
            lambda_um,
            aoi_deg: vec![aoi_deg],
            power_resolution: PowerResolution::Unpolarized,
            rs: None,
            rp: None,
            ts: None,
            tp: None,
            r: Some(r),
            t: Some(t),
            phase_information: PhaseInformation::Unknown,
            scope: TableScope::Interface,
            side: TableSide::Front,
            media,
            provenance: None,
            out_of_domain: None,
            reflection: ReflectionRole::Path,
        }
    }

    fn len(&self) -> usize {
        self.lambda_um.len() * self.aoi_deg.len()
    }

    /// The arrays the resolution calls for, by name, in a fixed order.
    fn arrays(&self) -> Vec<(&'static str, Option<&Vec<f64>>)> {
        match self.power_resolution {
            PowerResolution::Sp => vec![
                ("rs", self.rs.as_ref()),
                ("rp", self.rp.as_ref()),
                ("ts", self.ts.as_ref()),
                ("tp", self.tp.as_ref()),
            ],
            PowerResolution::Unpolarized | PowerResolution::Scalar => {
                vec![("r", self.r.as_ref()), ("t", self.t.as_ref())]
            }
        }
    }

    /// Axes strictly increasing and finite, wavelengths positive, angles in
    /// [0, 90), every array the resolution calls for present at the right
    /// length and no other, every value in [0, 1], R + T ≤ 1 for each
    /// polarization, the media positive. Invalid data are refused, never
    /// projected onto something plausible.
    pub fn validate(&self) -> Result<(), String> {
        let increasing = |v: &[f64]| v.windows(2).all(|w| w[0] < w[1]);
        if self.lambda_um.is_empty() || self.aoi_deg.is_empty() {
            return Err("a table needs at least one wavelength and one angle".into());
        }
        if !self.lambda_um.iter().all(|l| l.is_finite() && *l > 0.0) || !increasing(&self.lambda_um) {
            return Err("lambda_um must be finite, positive and strictly increasing".into());
        }
        if !self.aoi_deg.iter().all(|a| a.is_finite() && (0.0..90.0).contains(a)) || !increasing(&self.aoi_deg) {
            return Err("aoi_deg must be in [0, 90) and strictly increasing".into());
        }
        if !(self.media.incident.is_finite() && self.media.incident > 0.0 && self.media.exit.is_finite() && self.media.exit > 0.0)
        {
            return Err("media indices must be finite and positive".into());
        }
        let wanted = self.arrays();
        for (name, a) in &wanted {
            let Some(a) = a else {
                return Err(format!("{:?} data need {name}", self.power_resolution));
            };
            if a.len() != self.len() {
                return Err(format!(
                    "{name} has {} values for {} wavelengths by {} angles",
                    a.len(),
                    self.lambda_um.len(),
                    self.aoi_deg.len()
                ));
            }
            if let Some(i) = a.iter().position(|v| !(v.is_finite() && (0.0..=1.0).contains(v))) {
                return Err(format!("{name}[{i}] = {} is outside [0, 1]", a[i]));
            }
        }
        let stray = [
            ("rs", self.rs.is_some()),
            ("rp", self.rp.is_some()),
            ("ts", self.ts.is_some()),
            ("tp", self.tp.is_some()),
            ("r", self.r.is_some()),
            ("t", self.t.is_some()),
        ];
        for (name, present) in stray {
            if present && !wanted.iter().any(|(w, _)| *w == name) {
                return Err(format!("{:?} data carry no {name}", self.power_resolution));
            }
        }
        let pairs: Vec<(&Vec<f64>, &Vec<f64>)> = match self.power_resolution {
            PowerResolution::Sp => vec![
                (self.rs.as_ref().unwrap(), self.ts.as_ref().unwrap()),
                (self.rp.as_ref().unwrap(), self.tp.as_ref().unwrap()),
            ],
            _ => vec![(self.r.as_ref().unwrap(), self.t.as_ref().unwrap())],
        };
        for (r, t) in pairs {
            if let Some(i) = (0..r.len()).find(|&i| r[i] + t[i] > 1.0 + 1e-9) {
                let (li, ai) = (i / self.aoi_deg.len(), i % self.aoi_deg.len());
                return Err(format!(
                    "R + T = {} exceeds 1 at {} nm, {} degrees",
                    r[i] + t[i],
                    num(self.lambda_um[li] * 1000.0),
                    num(self.aoi_deg[ai])
                ));
            }
        }
        Ok(())
    }

    /// Does this table refuse a ray outside its data under `numerics`?
    pub fn refuses(&self, numerics: Numerics) -> bool {
        numerics == Numerics::Strict || self.out_of_domain == Some(OutOfDomain::Refuse)
    }

    /// Why a ray at `lambda_um` and `aoi_deg` is outside the data, or `None`
    /// when it is inside.
    pub fn outside(&self, lambda_um: f64, aoi_deg: f64) -> Option<String> {
        let (a0, a1) = (self.aoi_deg[0], self.aoi_deg[self.aoi_deg.len() - 1]);
        if aoi_deg < a0 - ANGLE_TOLERANCE_DEG || aoi_deg > a1 + ANGLE_TOLERANCE_DEG {
            return Some(if self.aoi_deg.len() == 1 {
                format!("Data supplied at {} degrees only; this ray is at {} degrees", num(a0), num(aoi_deg))
            } else {
                format!("Data supplied from {} to {} degrees; this ray is at {} degrees", num(a0), num(a1), num(aoi_deg))
            });
        }
        let (l0, l1) = (self.lambda_um[0], self.lambda_um[self.lambda_um.len() - 1]);
        if lambda_um < l0 - 1e-9 || lambda_um > l1 + 1e-9 {
            return Some(if self.lambda_um.len() == 1 {
                format!("Data supplied at {} nm only; this ray is at {} nm", num(l0 * 1000.0), num(lambda_um * 1000.0))
            } else {
                format!(
                    "Data supplied from {} to {} nm; this ray is at {} nm",
                    num(l0 * 1000.0),
                    num(l1 * 1000.0),
                    num(lambda_um * 1000.0)
                )
            });
        }
        None
    }

    /// Why the side or the media a ray meets the table from are not the
    /// ones it was measured from, or `None`. `from` is the side of the part
    /// this copy of the table answers for.
    pub fn mismatch(&self, from: TableSide, n_incident: f64, n_exit: f64) -> Option<String> {
        if self.side != TableSide::Either && from != TableSide::Either && self.side != from {
            let (data, ray) = match self.side {
                TableSide::Front => ("front", "back"),
                _ => ("back", "front"),
            };
            return Some(format!("Data supplied for the {data} side only; this ray meets the {ray}"));
        }
        if (n_incident - self.media.incident).abs() > MEDIA_TOLERANCE || (n_exit - self.media.exit).abs() > MEDIA_TOLERANCE
        {
            return Some(format!(
                "Data measured from n {:.3} into n {:.3}; this ray goes from n {n_incident:.3} into n {n_exit:.3}",
                self.media.incident, self.media.exit
            ));
        }
        None
    }

    /// The powers at `(lambda_um, cos_i)`, the point first clamped into the
    /// data: bilinear in wavelength and in `cos θ`. Exact at the knots.
    pub fn powers_clamped(&self, lambda_um: f64, cos_i: f64) -> TablePowers {
        let (i0, i1, u) = bracket(&self.lambda_um, lambda_um);
        let cos: Vec<f64> = self.aoi_deg.iter().map(|a| a.to_radians().cos()).collect();
        // `cos θ` falls as the angle rises: bracket on its negative, which
        // rises with it.
        let neg: Vec<f64> = cos.iter().map(|c| -c).collect();
        let (j0, j1, v) = bracket(&neg, -cos_i.clamp(-1.0, 1.0));
        let n = self.aoi_deg.len();
        let at = |a: &Vec<f64>| {
            let f = |i: usize, j: usize| a[i * n + j];
            let lo = f(i0, j0) + v * (f(i0, j1) - f(i0, j0));
            let hi = f(i1, j0) + v * (f(i1, j1) - f(i1, j0));
            lo + u * (hi - lo)
        };
        match self.power_resolution {
            PowerResolution::Sp => TablePowers {
                rs: at(self.rs.as_ref().expect("validated")),
                rp: at(self.rp.as_ref().expect("validated")),
                ts: at(self.ts.as_ref().expect("validated")),
                tp: at(self.tp.as_ref().expect("validated")),
            },
            _ => {
                let r = at(self.r.as_ref().expect("validated"));
                let t = at(self.t.as_ref().expect("validated"));
                TablePowers { rs: r, rp: r, ts: t, tp: t }
            }
        }
    }

    /// The lookup a ray makes: refused with the reason when it is outside the
    /// data (or met from the wrong side or media) and `refuse` is set,
    /// otherwise read at the nearest edge and flagged.
    pub fn lookup(
        &self,
        lambda_um: f64,
        cos_i: f64,
        from: TableSide,
        n_incident: f64,
        n_exit: f64,
        refuse: bool,
    ) -> Result<TableLookup, String> {
        let aoi = cos_i.clamp(-1.0, 1.0).abs().acos().to_degrees();
        let why = self.mismatch(from, n_incident, n_exit).or_else(|| self.outside(lambda_um, aoi));
        match why {
            Some(w) if refuse => Err(w),
            why => Ok(TableLookup { powers: self.powers_clamped(lambda_um, cos_i.abs()), clamped: why }),
        }
    }

    /// The table a coating's transfer matrix gives on a grid: s and p powers
    /// between `n_incident` and `n_exit`, provenance `Computed`. At its knots
    /// it is the solver.
    pub fn from_coating(
        c: &Coating,
        n_incident: f64,
        n_exit: f64,
        lambda_um: &[f64],
        aoi_deg: &[f64],
        ctx: &FilmContext,
    ) -> Result<Self, FilmError> {
        let mut rs = Vec::new();
        let mut rp = Vec::new();
        let mut ts = Vec::new();
        let mut tp = Vec::new();
        for &l in lambda_um {
            for &a in aoi_deg {
                let (v, _) = crate::thinfilm::coating_eval(c, n_incident, n_exit, l, a.to_radians().cos(), ctx)?;
                rs.push(v.r_s);
                rp.push(v.r_p);
                ts.push(v.t_s);
                tp.push(v.t_p);
            }
        }
        Ok(Self {
            lambda_um: lambda_um.to_vec(),
            aoi_deg: aoi_deg.to_vec(),
            power_resolution: PowerResolution::Sp,
            rs: Some(rs),
            rp: Some(rp),
            ts: Some(ts),
            tp: Some(tp),
            r: None,
            t: None,
            phase_information: PhaseInformation::Unknown,
            scope: TableScope::Interface,
            side: TableSide::Front,
            media: TableMedia { incident: n_incident, exit: n_exit },
            provenance: Some(Provenance {
                source_kind: SourceKind::Computed,
                reference: String::new(),
                url: String::new(),
                license: String::new(),
                retrieved: String::new(),
                range_um: Some([lambda_um[0], lambda_um[lambda_um.len() - 1]]),
                note: "computed by the canvas's transfer matrix".into(),
            }),
            out_of_domain: None,
            reflection: if c.reflection_is_stray() { ReflectionRole::Stray } else { ReflectionRole::Path },
        })
    }

    /// The table as CSV in long form, one row per wavelength and angle:
    /// `lambda_um,aoi_deg,` then the arrays the resolution has, as fractions.
    /// Every number is written in its shortest round-trip form, so reading it
    /// back with [`CsvMap::for_table`] gives the same table.
    pub fn to_csv(&self) -> String {
        let arrays = self.arrays();
        let mut out = String::from("lambda_um,aoi_deg");
        for (name, _) in &arrays {
            out.push(',');
            out.push_str(name);
        }
        out.push('\n');
        let n = self.aoi_deg.len();
        for (i, l) in self.lambda_um.iter().enumerate() {
            for (j, a) in self.aoi_deg.iter().enumerate() {
                out.push_str(&format!("{l:?},{a:?}"));
                for (_, v) in &arrays {
                    out.push_str(&format!(",{:?}", v.expect("validated")[i * n + j]));
                }
                out.push('\n');
            }
        }
        out
    }
}

/// The two knots either side of `x` on an increasing axis and the weight of
/// the second, `x` clamped into the axis. A one-point axis is its point.
fn bracket(axis: &[f64], x: f64) -> (usize, usize, f64) {
    let n = axis.len();
    if n == 1 || x <= axis[0] {
        return (0, 0, 0.0);
    }
    if x >= axis[n - 1] {
        return (n - 1, n - 1, 0.0);
    }
    let i = axis.partition_point(|&v| v <= x).clamp(1, n - 1) - 1;
    let u = (x - axis[i]) / (axis[i + 1] - axis[i]);
    (i, i + 1, u)
}

/// The wavelength column's unit.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum LambdaUnit {
    Nm,
    Um,
}

/// How the values are written.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ValueUnit {
    /// 0 to 1.
    Fraction,
    /// 0 to 100.
    Percent,
}

/// Where the angle comes from: one stated angle for the whole file, or a
/// column (long form, one row per wavelength and angle).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum CsvAoi {
    Fixed(f64),
    Column(usize),
}

/// Which columns hold the powers, by zero-based index.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CsvColumns {
    Sp { rs: usize, rp: usize, ts: usize, tp: usize },
    Unpolarized { r: usize, t: usize },
    Scalar { r: usize, t: usize },
    /// Only T was measured, with unpolarized light, and R is declared to be
    /// 1 − T: a lossless filter. The table's provenance says so.
    UnpolarizedLosslessT { t: usize },
}

/// Everything a CSV file does not say and a table must: the columns and their
/// units, the angle, the side, the scope and the media. The importer asks
/// for all of it.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CsvMap {
    pub lambda_column: usize,
    pub lambda_unit: LambdaUnit,
    pub aoi: CsvAoi,
    pub columns: CsvColumns,
    pub value_unit: ValueUnit,
    pub side: TableSide,
    pub scope: TableScope,
    pub media: TableMedia,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub provenance: Option<Provenance>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub out_of_domain: Option<OutOfDomain>,
    #[serde(default)]
    pub reflection: ReflectionRole,
}

impl CsvMap {
    /// The map that reads [`AngularTable::to_csv`] back into `table`.
    pub fn for_table(table: &AngularTable) -> Self {
        Self {
            lambda_column: 0,
            lambda_unit: LambdaUnit::Um,
            aoi: CsvAoi::Column(1),
            columns: match table.power_resolution {
                PowerResolution::Sp => CsvColumns::Sp { rs: 2, rp: 3, ts: 4, tp: 5 },
                PowerResolution::Unpolarized => CsvColumns::Unpolarized { r: 2, t: 3 },
                PowerResolution::Scalar => CsvColumns::Scalar { r: 2, t: 3 },
            },
            value_unit: ValueUnit::Fraction,
            side: table.side,
            scope: table.scope,
            media: table.media,
            provenance: table.provenance.clone(),
            out_of_domain: table.out_of_domain,
            reflection: table.reflection,
        }
    }
}

/// Split one CSV line on commas, semicolons, tabs, or runs of spaces.
fn fields(line: &str) -> Vec<&str> {
    match [',', ';', '\t'].into_iter().find(|c| line.contains(*c)) {
        Some(sep) => line.split(sep).map(str::trim).collect(),
        None => line.split_whitespace().collect(),
    }
}

/// Read a CSV file into a table, with `map` saying what the columns are.
///
/// Lines that are empty, start with `#`, or do not parse as numbers in the
/// columns the map names are skipped as headers or comments. In long form
/// (`CsvAoi::Column`) the rows may come in any order but must cover every
/// wavelength at every angle exactly once. The table is validated before it
/// is returned.
pub fn parse_csv(text: &str, map: &CsvMap) -> Result<AngularTable, String> {
    let value_cols: Vec<usize> = match map.columns {
        CsvColumns::Sp { rs, rp, ts, tp } => vec![rs, rp, ts, tp],
        CsvColumns::Unpolarized { r, t } | CsvColumns::Scalar { r, t } => vec![r, t],
        CsvColumns::UnpolarizedLosslessT { t } => vec![t],
    };
    let scale = match map.value_unit {
        ValueUnit::Fraction => 1.0,
        ValueUnit::Percent => 0.01,
    };
    let mut rows: Vec<(f64, f64, Vec<f64>)> = Vec::new();
    for (ln, line) in text.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let f = fields(line);
        let get = |c: usize| f.get(c).and_then(|s| s.parse::<f64>().ok());
        let Some(lam) = get(map.lambda_column) else {
            if rows.is_empty() {
                continue;
            }
            return Err(format!("line {}: no number in the wavelength column", ln + 1));
        };
        let aoi = match map.aoi {
            CsvAoi::Fixed(a) => a,
            CsvAoi::Column(c) => get(c).ok_or_else(|| format!("line {}: no number in the angle column", ln + 1))?,
        };
        let vals: Option<Vec<f64>> = value_cols.iter().map(|&c| get(c).map(|v| v * scale)).collect();
        let vals = vals.ok_or_else(|| format!("line {}: a value column is missing or not a number", ln + 1))?;
        let lam_um = match map.lambda_unit {
            LambdaUnit::Nm => lam / 1000.0,
            LambdaUnit::Um => lam,
        };
        rows.push((lam_um, aoi, vals));
    }
    if rows.is_empty() {
        return Err("no data rows".into());
    }
    let axis = |mut v: Vec<f64>| {
        v.sort_by(f64::total_cmp);
        v.dedup();
        v
    };
    let lambda_um = axis(rows.iter().map(|r| r.0).collect());
    let aoi_deg = axis(rows.iter().map(|r| r.1).collect());
    let (nl, na) = (lambda_um.len(), aoi_deg.len());
    let mut grid: Vec<Option<Vec<f64>>> = vec![None; nl * na];
    for (lam, aoi, vals) in rows {
        let i = lambda_um.binary_search_by(|v| v.total_cmp(&lam)).expect("on the axis");
        let j = aoi_deg.binary_search_by(|v| v.total_cmp(&aoi)).expect("on the axis");
        if grid[i * na + j].replace(vals).is_some() {
            return Err(format!("two rows for {} nm at {} degrees", num(lam * 1000.0), num(aoi)));
        }
    }
    let mut cols: Vec<Vec<f64>> = vec![Vec::with_capacity(nl * na); value_cols.len()];
    for (k, cell) in grid.iter().enumerate() {
        let Some(v) = cell else {
            return Err(format!(
                "no row for {} nm at {} degrees: a table needs every wavelength at every angle",
                num(lambda_um[k / na] * 1000.0),
                num(aoi_deg[k % na])
            ));
        };
        for (c, x) in cols.iter_mut().zip(v) {
            c.push(*x);
        }
    }
    let mut provenance = map.provenance.clone();
    let mut table = AngularTable {
        lambda_um,
        aoi_deg,
        power_resolution: PowerResolution::Unpolarized,
        rs: None,
        rp: None,
        ts: None,
        tp: None,
        r: None,
        t: None,
        phase_information: PhaseInformation::Unknown,
        scope: map.scope,
        side: map.side,
        media: map.media,
        provenance: None,
        out_of_domain: map.out_of_domain,
        reflection: map.reflection,
    };
    let mut it = cols.into_iter();
    match map.columns {
        CsvColumns::Sp { .. } => {
            table.power_resolution = PowerResolution::Sp;
            table.rs = it.next();
            table.rp = it.next();
            table.ts = it.next();
            table.tp = it.next();
        }
        CsvColumns::Unpolarized { .. } | CsvColumns::Scalar { .. } => {
            if matches!(map.columns, CsvColumns::Scalar { .. }) {
                table.power_resolution = PowerResolution::Scalar;
            }
            table.r = it.next();
            table.t = it.next();
        }
        CsvColumns::UnpolarizedLosslessT { .. } => {
            let t = it.next().expect("one column");
            table.r = Some(t.iter().map(|v| (1.0 - v).max(0.0)).collect());
            table.t = Some(t);
            let p = provenance.get_or_insert_with(|| Provenance {
                source_kind: SourceKind::Supplied,
                reference: String::new(),
                url: String::new(),
                license: String::new(),
                retrieved: String::new(),
                range_um: None,
                note: String::new(),
            });
            let note = "R taken as 1 - T: declared lossless at import";
            p.note = if p.note.is_empty() { note.into() } else { format!("{}; {note}", p.note) };
        }
    }
    table.provenance = provenance;
    table.validate()?;
    Ok(table)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn media() -> TableMedia {
        TableMedia { incident: 1.0, exit: 1.0 }
    }

    /// A table whose values are exactly bilinear in `(λ, cos θ)`: the lookup
    /// must give the function itself between the knots.
    fn bilinear() -> (AngularTable, impl Fn(f64, f64) -> [f64; 4]) {
        let f = |l: f64, c: f64| {
            [
                0.1 + 0.2 * l + 0.05 * c + 0.1 * l * c,
                0.05 + 0.1 * l + 0.02 * c + 0.03 * l * c,
                0.6 - 0.2 * l + 0.1 * c - 0.05 * l * c,
                0.7 - 0.1 * l + 0.05 * c - 0.02 * l * c,
            ]
        };
        let lambda_um = vec![0.4, 0.5, 0.65, 0.7];
        let aoi_deg = vec![0.0, 20.0, 45.0, 60.0];
        let mut cols = [Vec::new(), Vec::new(), Vec::new(), Vec::new()];
        for &l in &lambda_um {
            for &a in &aoi_deg {
                let v = f(l, f64::to_radians(a).cos());
                for (col, x) in cols.iter_mut().zip(v) {
                    col.push(x);
                }
            }
        }
        let [rs, rp, ts, tp] = cols;
        let t = AngularTable {
            lambda_um,
            aoi_deg,
            power_resolution: PowerResolution::Sp,
            rs: Some(rs),
            rp: Some(rp),
            ts: Some(ts),
            tp: Some(tp),
            r: None,
            t: None,
            phase_information: PhaseInformation::Unknown,
            scope: TableScope::Interface,
            side: TableSide::Either,
            media: media(),
            provenance: None,
            out_of_domain: None,
            reflection: ReflectionRole::Path,
        };
        (t, f)
    }

    /// C28: the lookup is exact for data that are bilinear in `(λ, cos θ)`,
    /// at knots and between them.
    #[test]
    fn c28_a_bilinear_fixture_is_reproduced_exactly() {
        let (t, f) = bilinear();
        t.validate().unwrap();
        let mut worst = 0.0f64;
        for l in [0.4, 0.43, 0.5, 0.61, 0.65, 0.699] {
            for a in [0.0, 7.0, 20.0, 33.3, 45.0, 59.0] {
                let c = f64::to_radians(a).cos();
                let got = t.powers_clamped(l, c);
                let want = f(l, c);
                for (g, w) in [got.rs, got.rp, got.ts, got.tp].iter().zip(want) {
                    worst = worst.max((g - w).abs());
                }
            }
        }
        assert!(worst <= 1e-12, "worst {worst:e}");
    }

    /// C28: a vendor table at 45 degrees refuses a ray at 52 under `Strict`
    /// with the sentence the spec names, and clamps it with a flag otherwise.
    #[test]
    fn c28_a_one_angle_table_refuses_by_name() {
        let t = AngularTable::one_angle(vec![0.4, 0.7], 45.0, vec![0.9, 0.1], vec![0.1, 0.9], media());
        t.validate().unwrap();
        let at = |deg: f64| f64::to_radians(deg).cos();
        let strict = t.refuses(Numerics::Strict);
        let err = t.lookup(0.55, at(52.0), TableSide::Front, 1.0, 1.0, strict).unwrap_err();
        assert_eq!(err, "Data supplied at 45 degrees only; this ray is at 52 degrees");
        // Within half a degree it is the data.
        let ok = t.lookup(0.55, at(45.4), TableSide::Front, 1.0, 1.0, strict).unwrap();
        assert!(ok.clamped.is_none());
        assert!((ok.powers.r() - 0.5).abs() < 1e-12);
        // Permissive: the edge, flagged.
        let lax = t.lookup(0.55, at(52.0), TableSide::Front, 1.0, 1.0, t.refuses(Numerics::Permissive)).unwrap();
        assert_eq!(lax.clamped.as_deref(), Some("Data supplied at 45 degrees only; this ray is at 52 degrees"));
        // The table's own Refuse holds under Permissive too.
        let mut firm = t.clone();
        firm.out_of_domain = Some(OutOfDomain::Refuse);
        assert!(firm.refuses(Numerics::Permissive));
        // The wrong side and the wrong media are named.
        let side = t.lookup(0.55, at(45.0), TableSide::Back, 1.0, 1.0, true).unwrap_err();
        assert_eq!(side, "Data supplied for the front side only; this ray meets the back");
        assert!(t.lookup(0.55, at(45.0), TableSide::Front, 1.33, 1.33, true).unwrap_err().contains("measured from n 1.000"));
        assert!(t.lookup(0.8, at(45.0), TableSide::Front, 1.0, 1.0, true).unwrap_err().contains("400 to 700 nm"));
    }

    /// C28: a table written as CSV and read back is the same table.
    #[test]
    fn c28_a_table_round_trips_through_csv() {
        let (mut t, _) = bilinear();
        t.provenance = Some(Provenance {
            source_kind: SourceKind::Supplied,
            reference: "vendor datasheet".into(),
            url: String::new(),
            license: String::new(),
            retrieved: "2026-10-08".into(),
            range_um: None,
            note: String::new(),
        });
        let back = parse_csv(&t.to_csv(), &CsvMap::for_table(&t)).unwrap();
        assert_eq!(back, t);
    }

    /// A vendor's file: nm, percent, one angle, T only, a header and a
    /// comment; R is declared 1 - T and the provenance says so.
    #[test]
    fn a_vendor_csv_reads_with_its_units() {
        let text = "# FB550-10 at 45 degrees\nWavelength (nm);Transmission (%)\n540;5\n550;80\n560;4.5\n";
        let map = CsvMap {
            lambda_column: 0,
            lambda_unit: LambdaUnit::Nm,
            aoi: CsvAoi::Fixed(45.0),
            columns: CsvColumns::UnpolarizedLosslessT { t: 1 },
            value_unit: ValueUnit::Percent,
            side: TableSide::Front,
            scope: TableScope::FiniteAssembly,
            media: media(),
            provenance: None,
            out_of_domain: None,
            reflection: ReflectionRole::Path,
        };
        let t = parse_csv(text, &map).unwrap();
        assert_eq!(t.lambda_um, vec![0.54, 0.55, 0.56]);
        assert_eq!(t.aoi_deg, vec![45.0]);
        assert!((t.t.as_ref().unwrap()[1] - 0.8).abs() < 1e-15);
        assert!((t.r.as_ref().unwrap()[1] - 0.2).abs() < 1e-15);
        assert!(t.provenance.unwrap().note.contains("1 - T"));
    }

    /// Bad data are refused, not repaired.
    #[test]
    fn invalid_tables_are_refused() {
        let ok = AngularTable::one_angle(vec![0.4, 0.7], 0.0, vec![0.5, 0.5], vec![0.5, 0.5], media());
        ok.validate().unwrap();
        let mut over = ok.clone();
        over.t = Some(vec![0.6, 0.5]);
        assert!(over.validate().unwrap_err().contains("exceeds 1 at 400 nm"));
        let mut order = ok.clone();
        order.lambda_um = vec![0.7, 0.4];
        assert!(order.validate().is_err());
        let mut short = ok.clone();
        short.r = Some(vec![0.5]);
        assert!(short.validate().unwrap_err().contains("1 values for 2 wavelengths"));
        let mut extra = ok.clone();
        extra.rs = Some(vec![0.5, 0.5]);
        assert!(extra.validate().unwrap_err().contains("carry no rs"));
        let gap = "lambda_um,aoi_deg,r,t\n0.4,0,0.1,0.9\n0.5,0,0.1,0.9\n0.4,10,0.1,0.9\n";
        let map = CsvMap { aoi: CsvAoi::Column(1), ..CsvMap::for_table(&ok) };
        let map = CsvMap { columns: CsvColumns::Unpolarized { r: 2, t: 3 }, ..map };
        assert!(parse_csv(gap, &map).unwrap_err().contains("no row for 500 nm at 10 degrees"));
    }

    /// C28: a table the transfer matrix writes reproduces the matrix at its
    /// knots (to 1e-9; it is exact).
    #[test]
    fn c28_a_solver_table_is_the_solver_at_its_knots() {
        use crate::thinfilm::{coating_eval, film_coating, CoatingDesign, DesignKind};
        let c = film_coating(
            CoatingDesign { kind: DesignKind::LongPass, lambda_nm: 560.0, aoi_deg: 45.0, periods: 6, n_high: 2.35, n_low: 1.46, adapt: None },
            1.0,
            1.52,
        );
        let ctx = FilmContext::default();
        let lambdas: Vec<f64> = (0..31).map(|i| 0.45 + 0.01 * i as f64).collect();
        let angles = [0.0, 15.0, 30.0, 45.0, 60.0];
        let t = AngularTable::from_coating(&c, 1.0, 1.52, &lambdas, &angles, &ctx).unwrap();
        t.validate().unwrap();
        let mut worst = 0.0f64;
        for &l in &lambdas {
            for &a in &angles {
                let cos = f64::to_radians(a).cos();
                let (v, _) = coating_eval(&c, 1.0, 1.52, l, cos, &ctx).unwrap();
                let p = t.powers_clamped(l, cos);
                for (g, w) in [(p.rs, v.r_s), (p.rp, v.r_p), (p.ts, v.t_s), (p.tp, v.t_p)] {
                    worst = worst.max((g - w).abs());
                }
            }
        }
        assert!(worst <= 1e-9, "worst {worst:e}");
        assert_eq!(t.provenance.as_ref().unwrap().source_kind, SourceKind::Computed);
    }
}
