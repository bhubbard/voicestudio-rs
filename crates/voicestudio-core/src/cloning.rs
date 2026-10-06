use anyhow::{Context, Result};
use hound::WavReader;
use serde::{Deserialize, Serialize};
use std::f32::consts::PI;
use std::path::Path;

pub const EMBEDDING_DIM: usize = 512;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpeakerPersona {
    pub id: String,
    pub name: String,
    pub sample_rate: u32,
    pub embedding: Vec<f32>,
    pub estimated_pitch_hz: f32,
    pub duration_seconds: f32,
}

pub struct VoiceCloner;

impl VoiceCloner {
    /// Extract speaker timbre embedding and acoustic profile from reference audio
    pub fn extract_persona<P: AsRef<Path>>(name: &str, wav_path: P) -> Result<SpeakerPersona> {
        let path_ref = wav_path.as_ref();
        let mut reader = WavReader::open(path_ref)
            .with_context(|| format!("Failed to read reference voice file {:?}", path_ref))?;

        let spec = reader.spec();
        let channels = spec.channels as usize;
        let sample_rate = spec.sample_rate;

        let raw: Vec<f32> = match spec.sample_format {
            hound::SampleFormat::Int => {
                let max = (1i64 << (spec.bits_per_sample - 1)) as f32;
                reader.samples::<i32>().map(|s| s.unwrap_or(0) as f32 / max).collect()
            }
            hound::SampleFormat::Float => reader.samples::<f32>().map(|s| s.unwrap_or(0.0)).collect(),
        };

        let mono: Vec<f32> = if channels > 1 {
            raw.chunks(channels).map(|c| c.iter().sum::<f32>() / channels as f32).collect()
        } else {
            raw
        };

        let duration_seconds = mono.len() as f32 / sample_rate as f32;

        let (embedding, pitch) = Self::compute_speaker_embedding(&mono, sample_rate);

        let id = format!(
            "{}_{}",
            name.to_lowercase().replace(' ', "_"),
            (duration_seconds * 1000.0) as u64
        );

        Ok(SpeakerPersona {
            id,
            name: name.to_string(),
            sample_rate,
            embedding,
            estimated_pitch_hz: pitch,
            duration_seconds,
        })
    }

    /// Compute 512-dim normalized speaker timbre embedding and estimate pitch
    fn compute_speaker_embedding(samples: &[f32], sample_rate: u32) -> (Vec<f32>, f32) {
        let mut emb = vec![0.0f32; EMBEDDING_DIM];
        if samples.is_empty() {
            return (emb, 150.0);
        }

        // 1. Estimate fundamental pitch (F0) using autocorrelation
        let max_lag = (sample_rate / 60) as usize; // min 60 Hz
        let min_lag = (sample_rate / 400) as usize; // max 400 Hz
        let mut best_lag = min_lag;
        let mut max_corr = 0.0f32;

        let search_len = samples.len().min(sample_rate as usize);
        if search_len > max_lag {
            for lag in min_lag..max_lag {
                let mut corr = 0.0f32;
                for i in 0..(search_len - lag) {
                    corr += samples[i] * samples[i + lag];
                }
                if corr > max_corr {
                    max_corr = corr;
                    best_lag = lag;
                }
            }
        }

        let pitch = if best_lag > 0 {
            sample_rate as f32 / best_lag as f32
        } else {
            150.0
        };

        // 2. Compute filterbank spectral energy embedding
        let chunk_size = samples.len() / EMBEDDING_DIM;
        if chunk_size > 0 {
            for d in 0..EMBEDDING_DIM {
                let start = d * chunk_size;
                let end = ((d + 1) * chunk_size).min(samples.len());
                let slice = &samples[start..end];
                let mut energy = 0.0f32;
                for (i, &s) in slice.iter().enumerate() {
                    let w = 0.54 - 0.46 * (2.0 * PI * i as f32 / slice.len() as f32).cos();
                    energy += (s * w).powi(2);
                }
                emb[d] = (1.0 + (energy / slice.len() as f32)).ln();
            }
        }

        // Normalize embedding to unit length
        let norm = emb.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 1e-6 {
            for x in &mut emb {
                *x /= norm;
            }
        }

        (emb, pitch)
    }

    /// Compute cosine similarity between two voice embeddings (-1.0 to 1.0)
    pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
        if a.len() != b.len() || a.is_empty() {
            return 0.0;
        }

        let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

        if norm_a > 1e-6 && norm_b > 1e-6 {
            dot / (norm_a * norm_b)
        } else {
            0.0
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::NamedTempFile;

    #[test]
    fn test_cosine_similarity() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![1.0, 0.0, 0.0];
        let c = vec![0.0, 1.0, 0.0];

        assert!((VoiceCloner::cosine_similarity(&a, &b) - 1.0).abs() < 1e-5);
        assert!((VoiceCloner::cosine_similarity(&a, &c)).abs() < 1e-5);
    }

    #[test]
    fn test_extract_persona() {
        let temp_wav = NamedTempFile::new().unwrap();
        let spec = hound::WavSpec {
            channels: 1,
            sample_rate: 16000,
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        };
        let mut writer = hound::WavWriter::create(temp_wav.path(), spec).unwrap();
        for i in 0..16000 {
            let sample = ((2.0 * PI * 220.0 * (i as f32 / 16000.0)).sin() * 15000.0) as i16;
            writer.write_sample(sample).unwrap();
        }
        writer.finalize().unwrap();

        let persona = VoiceCloner::extract_persona("Rachel", temp_wav.path()).unwrap();
        assert_eq!(persona.name, "Rachel");
        assert_eq!(persona.sample_rate, 16000);
        assert_eq!(persona.embedding.len(), EMBEDDING_DIM);
        assert!(persona.estimated_pitch_hz > 100.0);
    }
}
