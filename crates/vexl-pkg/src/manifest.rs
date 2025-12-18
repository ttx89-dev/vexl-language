//! VEXL Package Manifest
//!
//! Defines the Vexl.toml package manifest format and parsing.
//! Specifies package metadata, dependencies, and build configuration.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::path::Path;

/// Package manifest (Vexl.toml)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Manifest {
    /// Package metadata
    pub package: PackageMetadata,
    /// Build configuration
    #[serde(default)]
    pub build: BuildConfig,
    /// Dependencies
    #[serde(default)]
    pub dependencies: HashMap<String, Dependency>,
    /// Development dependencies
    #[serde(default)]
    pub dev_dependencies: HashMap<String, Dependency>,
    /// Optional features
    #[serde(default)]
    pub features: HashMap<String, Vec<String>>,
    /// Workspace configuration (for workspace root)
    #[serde(default)]
    pub workspace: Option<WorkspaceConfig>,
}

/// Package metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PackageMetadata {
    /// Package name
    pub name: String,
    /// Package version
    pub version: String,
    /// Package authors
    #[serde(default)]
    pub authors: Vec<String>,
    /// Package description
    pub description: Option<String>,
    /// Package license
    pub license: Option<String>,
    /// Package repository URL
    pub repository: Option<String>,
    /// Package homepage URL
    pub homepage: Option<String>,
    /// Package keywords
    #[serde(default)]
    pub keywords: Vec<String>,
    /// Package categories
    #[serde(default)]
    pub categories: Vec<String>,
    /// Whether the package is published
    #[serde(default)]
    pub publish: bool,
    /// Minimum VEXL version required
    pub vexl: Option<String>,
}

/// Build configuration
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct BuildConfig {
    /// Build target (cpu, gpu, both)
    #[serde(default)]
    pub target: BuildTarget,
    /// Optimization level
    #[serde(default)]
    pub opt_level: OptLevel,
    /// Debug information
    #[serde(default)]
    pub debug: bool,
    /// Additional compiler flags
    #[serde(default)]
    pub flags: Vec<String>,
    /// GPU compute capability (if GPU target)
    pub gpu_compute_capability: Option<String>,
}

/// Build target
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum BuildTarget {
    Cpu,
    Gpu,
    Both,
}

impl Default for BuildTarget {
    fn default() -> Self {
        BuildTarget::Cpu
    }
}

/// Optimization level
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum OptLevel {
    O0,
    O1,
    O2,
    O3,
}

impl Default for OptLevel {
    fn default() -> Self {
        OptLevel::O2
    }
}

/// Dependency specification
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum Dependency {
    /// Simple version requirement
    Simple(String),
    /// Detailed dependency specification
    Detailed(DependencySpec),
}

/// Detailed dependency specification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DependencySpec {
    /// Version requirement
    pub version: String,
    /// Optional dependency
    #[serde(default)]
    pub optional: bool,
    /// Features to enable
    #[serde(default)]
    pub features: Vec<String>,
    /// Git repository URL
    pub git: Option<String>,
    /// Branch/tag/commit for git dependencies
    pub branch: Option<String>,
    pub tag: Option<String>,
    pub rev: Option<String>,
    /// Path to local dependency
    pub path: Option<String>,
    /// Registry name (for custom registries)
    pub registry: Option<String>,
}

/// Workspace configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkspaceConfig {
    /// Workspace members (package paths)
    pub members: Vec<String>,
    /// Default members for commands
    #[serde(default)]
    pub default_members: Vec<String>,
    /// Workspace-wide dependencies
    #[serde(default)]
    pub dependencies: HashMap<String, Dependency>,
}

impl Manifest {
    /// Load manifest from Vexl.toml file
    pub fn from_file(path: &Path) -> Result<Self, String> {
        let content = fs::read_to_string(path)
            .map_err(|e| format!("Failed to read manifest file: {}", e))?;

        Self::from_str(&content)
    }

    /// Parse manifest from string
    pub fn from_str(content: &str) -> Result<Self, String> {
        toml::from_str(content)
            .map_err(|e| format!("Failed to parse manifest: {}", e))
    }

    /// Save manifest to Vexl.toml file
    pub fn save_to_file(&self, path: &Path) -> Result<(), String> {
        let content = self.to_string()?;
        fs::write(path, content)
            .map_err(|e| format!("Failed to write manifest file: {}", e))
    }

    /// Convert manifest to TOML string
    pub fn to_string(&self) -> Result<String, String> {
        toml::to_string_pretty(self)
            .map_err(|e| format!("Failed to serialize manifest: {}", e))
    }

    /// Get package name
    pub fn name(&self) -> &str {
        &self.package.name
    }

    /// Get package version
    pub fn version(&self) -> &str {
        &self.package.version
    }

    /// Check if this is a workspace root
    pub fn is_workspace(&self) -> bool {
        self.workspace.is_some()
    }

    /// Get workspace members (if this is a workspace)
    pub fn workspace_members(&self) -> Option<&[String]> {
        self.workspace.as_ref().map(|w| w.members.as_slice())
    }

    /// Validate manifest
    pub fn validate(&self) -> Result<(), Vec<String>> {
        let mut errors = Vec::new();

        // Validate package name
        if self.package.name.is_empty() {
            errors.push("Package name cannot be empty".to_string());
        }

        if !is_valid_package_name(&self.package.name) {
            errors.push("Package name contains invalid characters".to_string());
        }

        // Validate version
        if self.package.version.is_empty() {
            errors.push("Package version cannot be empty".to_string());
        }

        if !is_valid_version(&self.package.version) {
            errors.push("Package version is not valid semver".to_string());
        }

        // Validate workspace configuration
        if let Some(workspace) = &self.workspace {
            if workspace.members.is_empty() {
                errors.push("Workspace must have at least one member".to_string());
            }
        }

        if errors.is_empty() {
            Ok(())
        } else {
            Err(errors)
        }
    }

    /// Add a dependency
    pub fn add_dependency(&mut self, name: String, dep: Dependency) {
        self.dependencies.insert(name, dep);
    }

    /// Remove a dependency
    pub fn remove_dependency(&mut self, name: &str) -> Option<Dependency> {
        self.dependencies.remove(name)
    }

    /// Add a dev dependency
    pub fn add_dev_dependency(&mut self, name: String, dep: Dependency) {
        self.dev_dependencies.insert(name, dep);
    }

    /// Get all dependencies (including dev)
    pub fn all_dependencies(&self) -> HashMap<&String, &Dependency> {
        let mut all = HashMap::new();

        for (name, dep) in &self.dependencies {
            all.insert(name, dep);
        }

        for (name, dep) in &self.dev_dependencies {
            all.insert(name, dep);
        }

        all
    }
}

/// Check if package name is valid
fn is_valid_package_name(name: &str) -> bool {
    if name.is_empty() {
        return false;
    }

    // Package names must be lowercase, alphanumeric, with hyphens allowed
    name.chars().all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '-')
        && !name.starts_with('-')
        && !name.ends_with('-')
}

/// Check if version is valid semver
fn is_valid_version(version: &str) -> bool {
    // Simple semver validation (major.minor.patch)
    let parts: Vec<&str> = version.split('.').collect();
    if parts.len() != 3 {
        return false;
    }

    parts.iter().all(|part| part.chars().all(|c| c.is_ascii_digit()))
}

/// Create a new package manifest
pub fn create_package_manifest(name: &str, version: &str, description: Option<&str>) -> Manifest {
    Manifest {
        package: PackageMetadata {
            name: name.to_string(),
            version: version.to_string(),
            authors: vec![],
            description: description.map(|s| s.to_string()),
            license: None,
            repository: None,
            homepage: None,
            keywords: vec![],
            categories: vec![],
            publish: true,
            vexl: Some("0.1.0".to_string()),
        },
        build: BuildConfig::default(),
        dependencies: HashMap::new(),
        dev_dependencies: HashMap::new(),
        features: HashMap::new(),
        workspace: None,
    }
}

/// Create a workspace manifest
pub fn create_workspace_manifest(members: Vec<String>) -> Manifest {
    Manifest {
        package: PackageMetadata {
            name: "workspace".to_string(),
            version: "0.1.0".to_string(),
            authors: vec![],
            description: Some("Workspace root".to_string()),
            license: None,
            repository: None,
            homepage: None,
            keywords: vec![],
            categories: vec![],
            publish: false,
            vexl: Some("0.1.0".to_string()),
        },
        build: BuildConfig::default(),
        dependencies: HashMap::new(),
        dev_dependencies: HashMap::new(),
        features: HashMap::new(),
        workspace: Some(WorkspaceConfig {
            members,
            default_members: vec![],
            dependencies: HashMap::new(),
        }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_manifest_creation() {
        let manifest = create_package_manifest("test-package", "1.0.0", Some("A test package"));

        assert_eq!(manifest.package.name, "test-package");
        assert_eq!(manifest.package.version, "1.0.0");
        assert_eq!(manifest.package.description.as_ref().unwrap(), "A test package");
    }

    #[test]
    fn test_manifest_validation() {
        let mut manifest = create_package_manifest("test-package", "1.0.0", None);

        // Should be valid
        assert!(manifest.validate().is_ok());

        // Test invalid name
        manifest.package.name = "Test Package".to_string();
        assert!(manifest.validate().is_err());

        // Reset and test invalid version
        manifest.package.name = "test-package".to_string();
        manifest.package.version = "invalid".to_string();
        assert!(manifest.validate().is_err());
    }

    #[test]
    fn test_workspace_manifest() {
        let members = vec!["pkg1".to_string(), "pkg2".to_string()];
        let manifest = create_workspace_manifest(members.clone());

        assert!(manifest.is_workspace());
        assert_eq!(manifest.workspace_members().unwrap(), members.as_slice());
    }

    #[test]
    fn test_dependency_management() {
        let mut manifest = create_package_manifest("test", "1.0.0", None);

        let dep = Dependency::Simple("1.0.0".to_string());
        manifest.add_dependency("some-dep".to_string(), dep.clone());

        assert_eq!(manifest.dependencies.len(), 1);
        assert_eq!(manifest.all_dependencies().len(), 1);

        let removed = manifest.remove_dependency("some-dep");
        assert!(removed.is_some());
        assert_eq!(manifest.dependencies.len(), 0);
    }
}
