
use anyhow::Result;
use std::fs::File;
use std::path::Path;
use flate2::write::GzEncoder;
use flate2::Compression;

pub struct Package;

impl Package {
    pub fn pack(source_dir: &Path, output_path: &Path) -> Result<()> {
        let tar_gz = File::create(output_path)?;
        let enc = GzEncoder::new(tar_gz, Compression::default());
        let mut tar = tar::Builder::new(enc);
        
        tar.append_dir_all(".", source_dir)?;
        tar.finish()?;
        
        Ok(())
    }

    pub fn unpack(archive_path: &Path, dest_dir: &Path) -> Result<()> {
        let tar_gz = File::open(archive_path)?;
        let tar = flate2::read::GzDecoder::new(tar_gz);
        let mut archive = tar::Archive::new(tar);
        
        archive.unpack(dest_dir)?;
        
        Ok(())
    }
}
