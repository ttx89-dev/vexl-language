//! VEXL Package Registry Client
//!
//! Handles communication with VEXL package registries for:
//! - Package search and discovery
//! - Package download and installation
//! - Authentication and authorization
//! - Registry management

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use crate::manifest::{Manifest, Dependency};

/// Registry client configuration
#[derive(Debug, Clone)]
pub struct RegistryConfig {
    /// Registry URL
    pub url: String,
    /// Authentication token
    pub auth_token: Option<String>,
    /// Timeout in seconds
    pub timeout: u64,
    /// Cache directory
    pub cache_dir: PathBuf,
}

impl Default for RegistryConfig {
    fn default() -> Self {
        Self {
            url: "https://registry.vexl-lang.org".to_string(),
            auth_token: None,
            timeout: 30,
            cache_dir: dirs::cache_dir()
                .unwrap_or_else(|| PathBuf::from(".cache"))
                .join("vexl"),
        }
    }
}

/// Package registry client
pub struct RegistryClient {
    config: RegistryConfig,
    client: reqwest::Client,
}

impl RegistryClient {
    /// Create a new registry client
    pub fn new(config: RegistryConfig) -> Result<Self, String> {
        let client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(config.timeout))
            .build()
            .map_err(|e| format!("Failed to create HTTP client: {}", e))?;

        // Create cache directory
        if !config.cache_dir.exists() {
            fs::create_dir_all(&config.cache_dir)
                .map_err(|e| format!("Failed to create cache directory: {}", e))?;
        }

        Ok(Self { config, client })
    }

    /// Search for packages
    pub async fn search_packages(&self, query: &str, limit: usize) -> Result<Vec<PackageInfo>, String> {
        let url = format!("{}/api/v1/search?q={}&limit={}", self.config.url, query, limit);

        let mut request = self.client.get(&url);

        if let Some(token) = &self.config.auth_token {
            request = request.header("Authorization", format!("Bearer {}", token));
        }

        let response = request.send().await
            .map_err(|e| format!("Search request failed: {}", e))?;

        if !response.status().is_success() {
            return Err(format!("Search failed with status: {}", response.status()));
        }

        let search_results: SearchResults = response.json().await
            .map_err(|e| format!("Failed to parse search results: {}", e))?;

        Ok(search_results.packages)
    }

    /// Get package information
    pub async fn get_package_info(&self, name: &str, version: Option<&str>) -> Result<PackageInfo, String> {
        let version_part = version.map(|v| format!("/{}", v)).unwrap_or_default();
        let url = format!("{}/api/v1/packages/{}{}", self.config.url, name, version_part);

        let mut request = self.client.get(&url);

        if let Some(token) = &self.config.auth_token {
            request = request.header("Authorization", format!("Bearer {}", token));
        }

        let response = request.send().await
            .map_err(|e| format!("Package info request failed: {}", e))?;

        if !response.status().is_success() {
            if response.status() == reqwest::StatusCode::NOT_FOUND {
                return Err(format!("Package '{}' not found", name));
            }
            return Err(format!("Package info request failed with status: {}", response.status()));
        }

        let package_info: PackageInfo = response.json().await
            .map_err(|e| format!("Failed to parse package info: {}", e))?;

        Ok(package_info)
    }

    /// Download package
    pub async fn download_package(&self, name: &str, version: &str) -> Result<PathBuf, String> {
        let package_info = self.get_package_info(name, Some(version)).await?;

        // Check cache first
        let cache_path = self.config.cache_dir
            .join(format!("{}-{}.vexl", name, version));

        if cache_path.exists() {
            return Ok(cache_path);
        }

        // Download from registry
        let download_url = format!("{}/api/v1/packages/{}/{}/download",
            self.config.url, name, version);

        let mut request = self.client.get(&download_url);

        if let Some(token) = &self.config.auth_token {
            request = request.header("Authorization", format!("Bearer {}", token));
        }

        let response = request.send().await
            .map_err(|e| format!("Download request failed: {}", e))?;

        if !response.status().is_success() {
            return Err(format!("Download failed with status: {}", response.status()));
        }

        let bytes = response.bytes().await
            .map_err(|e| format!("Failed to read download response: {}", e))?;

        // Verify checksum if provided
        if let Some(expected_checksum) = package_info.checksum.as_ref() {
            use sha2::{Sha256, Digest};
            let mut hasher = Sha256::new();
            hasher.update(&bytes);
            let actual_checksum = format!("{:x}", hasher.finalize());

            if actual_checksum != *expected_checksum {
                return Err("Package checksum verification failed".to_string());
            }
        }

        // Save to cache
        fs::write(&cache_path, &bytes)
            .map_err(|e| format!("Failed to save package to cache: {}", e))?;

        Ok(cache_path)
    }

    /// Publish package
    pub async fn publish_package(&self, manifest: &Manifest, package_path: &Path) -> Result<(), String> {
        if manifest.package.publish == false {
            return Err("Package is marked as not publishable".to_string());
        }

        // Read package file
        let package_data = fs::read(package_path)
            .map_err(|e| format!("Failed to read package file: {}", e))?;

        // Calculate checksum
        use sha2::{Sha256, Digest};
        let mut hasher = Sha256::new();
        hasher.update(&package_data);
        let checksum = format!("{:x}", hasher.finalize());

        // Create publish request
        let publish_request = PublishRequest {
            name: manifest.package.name.clone(),
            version: manifest.package.version.clone(),
            description: manifest.package.description.clone(),
            authors: manifest.package.authors.clone(),
            license: manifest.package.license.clone(),
            repository: manifest.package.repository.clone(),
            homepage: manifest.package.homepage.clone(),
            keywords: manifest.package.keywords.clone(),
            categories: manifest.package.categories.clone(),
            dependencies: manifest.dependencies.clone(),
            dev_dependencies: manifest.dev_dependencies.clone(),
            checksum,
            size: package_data.len(),
        };

        let url = format!("{}/api/v1/packages", self.config.url);

        let mut request = self.client.post(&url)
            .json(&publish_request);

        if let Some(token) = &self.config.auth_token {
            request = request.header("Authorization", format!("Bearer {}", token));
        } else {
            return Err("Authentication token required for publishing".to_string());
        }

        let response = request.send().await
            .map_err(|e| format!("Publish request failed: {}", e))?;

        if !response.status().is_success() {
            let error_text = response.text().await
                .unwrap_or_else(|_| "Unknown error".to_string());
            return Err(format!("Publish failed: {}", error_text));
        }

        // Upload package data
        let upload_url = format!("{}/api/v1/packages/{}/{}/upload",
            self.config.url, manifest.package.name, manifest.package.version);

        let response = self.client.put(&upload_url)
            .header("Authorization", format!("Bearer {}", self.config.auth_token.as_ref().unwrap()))
            .body(package_data)
            .send().await
            .map_err(|e| format!("Upload request failed: {}", e))?;

        if !response.status().is_success() {
            return Err(format!("Upload failed with status: {}", response.status()));
        }

        Ok(())
    }

    /// Get user's packages
    pub async fn get_my_packages(&self) -> Result<Vec<PackageInfo>, String> {
        if self.config.auth_token.is_none() {
            return Err("Authentication required".to_string());
        }

        let url = format!("{}/api/v1/user/packages", self.config.url);

        let response = self.client.get(&url)
            .header("Authorization", format!("Bearer {}", self.config.auth_token.as_ref().unwrap()))
            .send().await
            .map_err(|e| format!("User packages request failed: {}", e))?;

        if !response.status().is_success() {
            return Err(format!("User packages request failed with status: {}", response.status()));
        }

        let packages: Vec<PackageInfo> = response.json().await
            .map_err(|e| format!("Failed to parse user packages: {}", e))?;

        Ok(packages)
    }

    /// Delete package
    pub async fn delete_package(&self, name: &str, version: &str) -> Result<(), String> {
        if self.config.auth_token.is_none() {
            return Err("Authentication required".to_string());
        }

        let url = format!("{}/api/v1/packages/{}/{}", self.config.url, name, version);

        let response = self.client.delete(&url)
            .header("Authorization", format!("Bearer {}", self.config.auth_token.as_ref().unwrap()))
            .send().await
            .map_err(|e| format!("Delete request failed: {}", e))?;

        if !response.status().is_success() {
            return Err(format!("Delete failed with status: {}", response.status()));
        }

        Ok(())
    }
}

/// Package information from registry
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PackageInfo {
    /// Package name
    pub name: String,
    /// Package version
    pub version: String,
    /// Package description
    pub description: Option<String>,
    /// Package authors
    pub authors: Vec<String>,
    /// Package license
    pub license: Option<String>,
    /// Package repository
    pub repository: Option<String>,
    /// Package homepage
    pub homepage: Option<String>,
    /// Package keywords
    pub keywords: Vec<String>,
    /// Package categories
    pub categories: Vec<String>,
    /// Dependencies
    pub dependencies: HashMap<String, Dependency>,
    /// Download count
    pub downloads: u64,
    /// Publication date
    pub published_at: String,
    /// Package checksum
    pub checksum: Option<String>,
    /// Package size in bytes
    pub size: Option<usize>,
}

/// Search results
#[derive(Debug, Serialize, Deserialize)]
struct SearchResults {
    pub packages: Vec<PackageInfo>,
    pub total: usize,
}

/// Publish request
#[derive(Debug, Serialize)]
struct PublishRequest {
    pub name: String,
    pub version: String,
    pub description: Option<String>,
    pub authors: Vec<String>,
    pub license: Option<String>,
    pub repository: Option<String>,
    pub homepage: Option<String>,
    pub keywords: Vec<String>,
    pub categories: Vec<String>,
    pub dependencies: HashMap<String, Dependency>,
    pub dev_dependencies: HashMap<String, Dependency>,
    pub checksum: String,
    pub size: usize,
}

/// Authentication token management
pub struct AuthManager {
    config_dir: PathBuf,
}

impl AuthManager {
    /// Create new auth manager
    pub fn new() -> Result<Self, String> {
        let config_dir = dirs::config_dir()
            .unwrap_or_else(|| PathBuf::from(".config"))
            .join("vexl");

        if !config_dir.exists() {
            fs::create_dir_all(&config_dir)
                .map_err(|e| format!("Failed to create config directory: {}", e))?;
        }

        Ok(Self { config_dir })
    }

    /// Save authentication token
    pub fn save_token(&self, registry: &str, token: &str) -> Result<(), String> {
        let token_file = self.config_dir.join(format!("{}.token", registry));

        fs::write(&token_file, token)
            .map_err(|e| format!("Failed to save token: {}", e))
    }

    /// Load authentication token
    pub fn load_token(&self, registry: &str) -> Result<String, String> {
        let token_file = self.config_dir.join(format!("{}.token", registry));

        fs::read_to_string(&token_file)
            .map_err(|e| format!("Failed to load token: {}", e))
    }

    /// Delete authentication token
    pub fn delete_token(&self, registry: &str) -> Result<(), String> {
        let token_file = self.config_dir.join(format!("{}.token", registry));

        if token_file.exists() {
            fs::remove_file(&token_file)
                .map_err(|e| format!("Failed to delete token: {}", e))
        } else {
            Ok(())
        }
    }
}

/// Local package cache
pub struct PackageCache {
    cache_dir: PathBuf,
}

impl PackageCache {
    /// Create new package cache
    pub fn new(cache_dir: PathBuf) -> Result<Self, String> {
        if !cache_dir.exists() {
            fs::create_dir_all(&cache_dir)
                .map_err(|e| format!("Failed to create cache directory: {}", e))?;
        }

        Ok(Self { cache_dir })
    }

    /// Check if package is cached
    pub fn is_cached(&self, name: &str, version: &str) -> bool {
        let cache_path = self.cache_dir.join(format!("{}-{}.vexl", name, version));
        cache_path.exists()
    }

    /// Get cached package path
    pub fn get_cached_path(&self, name: &str, version: &str) -> PathBuf {
        self.cache_dir.join(format!("{}-{}.vexl", name, version))
    }

    /// Cache package
    pub fn cache_package(&self, name: &str, version: &str, data: &[u8]) -> Result<PathBuf, String> {
        let cache_path = self.get_cached_path(name, version);

        fs::write(&cache_path, data)
            .map_err(|e| format!("Failed to cache package: {}", e))?;

        Ok(cache_path)
    }

    /// Clear cache
    pub fn clear_cache(&self) -> Result<(), String> {
        for entry in fs::read_dir(&self.cache_dir)
            .map_err(|e| format!("Failed to read cache directory: {}", e))? {

            let entry = entry.map_err(|e| format!("Failed to read cache entry: {}", e))?;
            let path = entry.path();

            if path.is_file() {
                fs::remove_file(&path)
                    .map_err(|e| format!("Failed to remove cached file: {}", e))?;
            }
        }

        Ok(())
    }

    /// Get cache statistics
    pub fn get_stats(&self) -> Result<CacheStats, String> {
        let mut total_size = 0u64;
        let mut file_count = 0usize;

        for entry in fs::read_dir(&self.cache_dir)
            .map_err(|e| format!("Failed to read cache directory: {}", e))? {

            let entry = entry.map_err(|e| format!("Failed to read cache entry: {}", e))?;
            let path = entry.path();

            if path.is_file() {
                let metadata = path.metadata()
                    .map_err(|e| format!("Failed to get file metadata: {}", e))?;
                total_size += metadata.len();
                file_count += 1;
            }
        }

        Ok(CacheStats {
            total_size,
            file_count,
        })
    }
}

/// Cache statistics
#[derive(Debug)]
pub struct CacheStats {
    /// Total size in bytes
    pub total_size: u64,
    /// Number of cached files
    pub file_count: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_registry_config() {
        let config = RegistryConfig::default();
        assert!(config.url.starts_with("https://"));
        assert_eq!(config.timeout, 30);
    }

    #[test]
    fn test_auth_manager() {
        let auth_manager = AuthManager::new().unwrap();
        assert!(auth_manager.config_dir.exists());

        // Test token operations (would fail without actual token file)
        let save_result = auth_manager.save_token("test-registry", "test-token");
        assert!(save_result.is_ok());

        let load_result = auth_manager.load_token("test-registry");
        assert!(load_result.is_ok());
        assert_eq!(load_result.unwrap(), "test-token");

        let delete_result = auth_manager.delete_token("test-registry");
        assert!(delete_result.is_ok());
    }

    #[test]
    fn test_package_cache() {
        let temp_dir = tempfile::tempdir().unwrap();
        let cache = PackageCache::new(temp_dir.path().to_path_buf()).unwrap();

        // Test cache operations
        assert!(!cache.is_cached("test-pkg", "1.0.0"));

        let test_data = b"test package data";
        let cached_path = cache.cache_package("test-pkg", "1.0.0", test_data).unwrap();

        assert!(cache.is_cached("test-pkg", "1.0.0"));
        assert_eq!(cache.get_cached_path("test-pkg", "1.0.0"), cached_path);

        // Test cache stats
        let stats = cache.get_stats().unwrap();
        assert_eq!(stats.file_count, 1);
        assert_eq!(stats.total_size, test_data.len() as u64);

        // Test cache clearing
        cache.clear_cache().unwrap();
        let stats_after_clear = cache.get_stats().unwrap();
        assert_eq!(stats_after_clear.file_count, 0);
    }
}
