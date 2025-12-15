
use vexl_pkg::{Package, PackageManifest};
use std::fs::File;
use std::io::Write;
use tempfile::tempdir;

#[test]
fn test_package_roundtrip() {
    let dir = tempdir().unwrap();
    let src_dir = dir.path().join("src_pkg");
    std::fs::create_dir(&src_dir).unwrap();

    // Create vexl.toml
    let manifest_content = r#"
    [package]
    name = "test-pkg"
    version = "0.1.0"
    authors = ["Tester"]
    "#;
    let mut f = File::create(src_dir.join("vexl.toml")).unwrap();
    f.write_all(manifest_content.as_bytes()).unwrap();

    // Create a source file
    let mut f = File::create(src_dir.join("lib.vexl")).unwrap();
    f.write_all(b"fn test() {}").unwrap();

    // Pack
    let tar_path = dir.path().join("package.tar.gz");
    Package::pack(&src_dir, &tar_path).unwrap();
    assert!(tar_path.exists());

    // Unpack
    let dest_dir = dir.path().join("dest_pkg");
    std::fs::create_dir(&dest_dir).unwrap();
    Package::unpack(&tar_path, &dest_dir).unwrap();

    // Verify
    assert!(dest_dir.join("vexl.toml").exists());
    assert!(dest_dir.join("lib.vexl").exists());
}
