use serde::{Deserialize, Serialize};
use std::path::PathBuf;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VoiceStudioConfig {
    /// Default audio sample rate (e.g. 24000 Hz or 44100 Hz)
    pub default_sample_rate: u32,
    /// Target loudness in LUFS for output normalization (default -16.0 LUFS)
    pub target_lufs: f32,
    /// Storage directory for voice personas and cloned embeddings
    pub personas_dir: PathBuf,
    /// Temp directory for intermediate dubbing renders
    pub temp_dir: PathBuf,
    /// Maximum allowed speech stretch factor in dubbing (default 1.25)
    pub max_time_stretch: f32,
    /// Maximum allowed speech compress factor in dubbing (default 0.8)
    pub min_time_stretch: f32,
}

impl Default for VoiceStudioConfig {
    fn default() -> Self {
        Self {
            default_sample_rate: 24000,
            target_lufs: -16.0,
            personas_dir: PathBuf::from("personas"),
            temp_dir: PathBuf::from("temp"),
            max_time_stretch: 1.25,
            min_time_stretch: 0.8,
        }
    }
}
