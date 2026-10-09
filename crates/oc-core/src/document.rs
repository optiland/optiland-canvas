//! One way in to an `.ocanvas` document, whatever version it is.
//!
//! The browser saves v2. Old files and older tools write v1. Every entry point
//! that takes a document (validate, normalize, load, export) reads it here, so
//! none of them can quietly assume a version and fail on the other one with a
//! parse error that says nothing about why.
//!
//! Failing to read a document is three different problems, and a caller acts
//! on each differently, so they stay apart:
//!
//! - [`DocumentError::NotADocument`]: the bytes are not JSON, or JSON that is
//!   not an `.ocanvas` document. Wrong file.
//! - [`DocumentError::UnsupportedVersion`]: an `.ocanvas` document of a version
//!   this build does not read. Newer tool, or a file from the future.
//! - [`DocumentError::InvalidScene`]: the right kind of document at a version
//!   this build reads, whose content does not hold up: a field of the wrong
//!   type, a non-finite number, a reference to an object that is not there.
//!
//! Whether a scene that reads fine can also become a sequential `Optic` is a
//! fourth question, and a separate module's (`oc-trace::seq_export`).

use crate::migrate_v1::{migrate_v1_to_v2, MigrationReport};
use crate::scene::{Scene, OCANVAS_FORMAT};
use crate::scene3::{Scene3, SceneError, OCANVAS_V2_VERSION};

/// The document versions this build reads.
pub const READABLE_VERSIONS: [u32; 2] = [1, OCANVAS_V2_VERSION];

/// Why a document could not be read. See the module docs.
#[derive(Debug, Clone, PartialEq)]
pub enum DocumentError {
    NotADocument(String),
    UnsupportedVersion { found: u32 },
    InvalidScene(String),
}

impl std::fmt::Display for DocumentError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NotADocument(why) => write!(f, "not an .ocanvas document: {why}"),
            Self::UnsupportedVersion { found } => write!(
                f,
                "an .ocanvas version {found} document; this build reads versions {:?}",
                READABLE_VERSIONS
            ),
            Self::InvalidScene(why) => {
                write!(f, "an .ocanvas document whose scene is invalid: {why}")
            }
        }
    }
}

impl std::error::Error for DocumentError {}

/// A document as read, in the version it was written in.
#[derive(Debug, Clone)]
pub enum Document {
    V1(Scene),
    V2(Scene3),
}

/// Read a document of any version this build supports.
///
/// The header decides, and it is read before anything else, so a wrong or
/// future version is reported as that rather than as whatever the typed parse
/// trips over first. A document with no header at all is v1, which is what
/// files written before the header existed are.
pub fn read_document(json: &str) -> Result<Document, DocumentError> {
    let value: serde_json::Value = serde_json::from_str(json)
        .map_err(|e| DocumentError::NotADocument(format!("not JSON ({e})")))?;
    let Some(obj) = value.as_object() else {
        return Err(DocumentError::NotADocument(
            "the top level is not a JSON object".into(),
        ));
    };
    let format = match obj.get("format") {
        None => OCANVAS_FORMAT,
        Some(serde_json::Value::String(s)) => s.as_str(),
        Some(other) => {
            return Err(DocumentError::NotADocument(format!(
                "`format` is {other}, not a string"
            )))
        }
    };
    if format != OCANVAS_FORMAT {
        return Err(DocumentError::NotADocument(format!(
            "`format` is {format:?}, expected {OCANVAS_FORMAT:?}"
        )));
    }
    let version = match obj.get("version") {
        None => 1,
        Some(v) => v
            .as_u64()
            .and_then(|v| u32::try_from(v).ok())
            .ok_or_else(|| {
                DocumentError::NotADocument(format!("`version` is {v}, not a version number"))
            })?,
    };
    match version {
        1 => Scene::from_json(json)
            .map(Document::V1)
            .map_err(|e| DocumentError::InvalidScene(e.to_string())),
        OCANVAS_V2_VERSION => Scene3::from_json(json)
            .map(Document::V2)
            .map_err(|e| match e {
                SceneError::UnsupportedVersion { found } => {
                    DocumentError::UnsupportedVersion { found }
                }
                SceneError::WrongFormat { found } => DocumentError::NotADocument(format!(
                    "`format` is {found:?}, expected {OCANVAS_FORMAT:?}"
                )),
                other => DocumentError::InvalidScene(other.to_string()),
            }),
        found => Err(DocumentError::UnsupportedVersion { found }),
    }
}

impl Document {
    /// The version the document was written in.
    pub fn version(&self) -> u32 {
        match self {
            Self::V1(_) => 1,
            Self::V2(_) => OCANVAS_V2_VERSION,
        }
    }

    /// The document re-emitted in its own version with the current header:
    /// normalizing never migrates, so a v1 file stays readable by v1 tools.
    pub fn to_json(&self) -> String {
        match self {
            Self::V1(s) => s.to_json(),
            Self::V2(s) => s.to_json(),
        }
    }

    /// The canonical 3D scene: a v2 document as it is, a v1 document through
    /// the migration, with the migration's report.
    pub fn into_scene3(self) -> Result<(Scene3, Option<MigrationReport>), DocumentError> {
        match self {
            Self::V2(s) => Ok((s, None)),
            Self::V1(s) => migrate_v1_to_v2(&s)
                .map(|(s3, report)| (s3, Some(report)))
                .map_err(|e| DocumentError::InvalidScene(e.to_string())),
        }
    }
}
