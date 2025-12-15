
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use semver::Version;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PackageManifest {
    pub package: PackageConfig,
    pub dependencies: HashMap<String, Dependency>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PackageConfig {
    pub name: String,
    pub version: Version,
    pub authors: Vec<String>,
    pub description: Option<String>,
    pub license: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum Dependency {
    Simple(String), // "1.0.0"
    Detailed {
        version: String,
        path: Option<String>,
        git: Option<String>,
    },
}
